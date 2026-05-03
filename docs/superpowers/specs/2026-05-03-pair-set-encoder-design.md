# W2.6 v2 重启：PairSetEncoder（bucket + 1-layer transformer pool）

**Status**: v1.0 — 初稿
**Author**: brainstorming session 2026-05-03（W2.6 v1 失败 F27 后重启）
**Branch**: `feature/pair-weighted-pool`（continued from v1 weighted pool 实验）
**Parent spec**: `docs/superpowers/specs/2026-05-01-taac-improvement-plan.md`（v2.2 W2.6 v1 已闭环）
**Predecessor (DEAD)**: `docs/superpowers/specs/2026-05-03-pair-feature-design.md`（v0.2 weighted pool；F27 实证 test -0.0022 ~ -0.0054）
**Goal**: 在 v1 weighted pool 失败后重启 W2.6，验证 **set self-attention pool + bucket-emb dense 注入** 能否带来 ≥ +0.002 信号层增量（同时 val + test 同向涨）

## Changelog

- **v1.0（2026-05-03 下午）**：初稿。v1 paradigm（hard-coded log1p / sigmoid weighted pool）已 F27 闭环。v2 完全换 paradigm：把 dense 从 pool weight 角色改为 token feature 角色，pool 机制改为 BERT-style 1 层 transformer block + mean pool

## 背景

### v1 失败回顾（F27 / parent spec v2.2）

| 实验 | val | test |
|---|---|---|
| baseline (none) | 0.862207 | 0.812282 |
| log1p mode (fid 62-66 only) | 0.862392 (+0.0002) | **0.81004 (-0.0022)** |
| full mode (62-66 log1p + 89-91 sigmoid) | 0.862408 (+0.0002) | **0.80693 (-0.0054)** |
| 边际 sigmoid on fid 89-91 | +0.000016 (零) | **-0.0031** |

**三重失败机制**：

1. **sigmoid 在 [-1,+1] 上对比度太低**：权重最大对比 ~2.5× vs log1p 在 fid 62-66 ~21× 对比；区分度低但仍微扰，等于注入噪声
2. **fid 91 全 0 率 48% 触发非系统性噪声**：一半样本 fallback mean-pool / 一半样本走微扰权重，模型见到的同一 fid 行为不一致
3. **val/test divergence 第 5 次触发**（F19/F21/F23/F25/F27）：dense 数值的 train→test 分布漂移让"基于 dense weight 的 pool 选择"在 test 上失效

**核心病灶**：dense 值直接驱动 pool 权重 → dense 分布漂移 → 输出连续漂移 → test 反向。

### v2 vs v1 paradigm shift

| 维度 | v1（DEAD） | v2（本 spec） |
|---|---|---|
| dense 角色 | **pool weight**（直接驱动 pool 选择）| **token feature**（注入到 token，让 attention 决定权重）|
| dense 注入形态 | hard-coded transform `log1p(v)` / `sigmoid(v)` | learnable bucket embedding `bucket_emb[bucket(v)]` |
| pool 机制 | mean pool weighted by transform(v) | mean pool **after** 1-layer transformer block |
| list 内元素交互 | 无（每元素独立加权）| 有（self-attn 让元素互相看）|
| pool 选择决定者 | dense values（连续漂移敏感）| attention scores（id_emb 主导，dense 辅助）|
| val/test divergence 鲁棒性 | 差（sigmoid 平滑映射 → v 漂 0.1 输出连续漂）| 好（bucket 离散 → 多数样本不跨桶 → emb 不变）|

## 设计

### 范围

- **改**：所有 8 个 paired fid（fid 62-66 + 89-91）的多值 mean-pool 升级为 PairSetEncoder（per-fid 独立模块）
  - v1 把 fid 89-91 单独放在 `full` mode 是因为 sigmoid 不适合 bounded 数据；v2 用 bucket 后这个限制消失，统一处理
- **不改**：
  - fid 15/60/80 多值 int（无 dense 配对，仍 mean-pool）
  - fid 61/87 纯 dense（仍走 user_dense token concat 路径）
  - 整个 user_dense token 输出（dense 数据全部仍保留进 dense token，PairSetEncoder 只是"读"它）

### 架构（PairSetEncoder）

```python
class PairSetEncoder(nn.Module):
    """Per-fid set encoder for paired (int, dense) features.

    Pipeline:
    1. id embedding lookup
    2. bucket(dense) → bucket embedding lookup
    3. add: tokens = id_emb + bucket_emb
    4. 1-layer BERT-style transformer block (attn + FFN, no causal mask)
    5. mean pool over valid (non-padding) positions
    6. Linear(D, D) project
    """
    NUM_BUCKETS = 32

    def __init__(self, fid: int, vocab: int, d_model: int, nhead: int = 4,
                 dim_feedforward: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.fid = fid
        self.d_model = d_model
        self.id_emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.bucket_emb = nn.Embedding(self.NUM_BUCKETS, d_model)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward or 2 * d_model,
            activation='gelu',           # BERT-style
            dropout=dropout,
            batch_first=True,
            norm_first=True,             # pre-norm，更稳
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def _quantize(self, vals: Tensor) -> Tensor:
        """Map dense values to bucket indices [0, NUM_BUCKETS)."""
        if self.fid in PAIR_WEIGHTED_FIDS_COUNT:        # 62-66, heavy-tailed [0, 1.5e9]
            v_pre = torch.log1p(vals.clamp(min=0))      # [0, ~21]
            bucket = (v_pre / 24.0 * self.NUM_BUCKETS).floor()
        else:                                            # 89-91, bounded [-1, +1]
            bucket = ((vals + 1.0) / 2.0 * self.NUM_BUCKETS).floor()
        return bucket.clamp(0, self.NUM_BUCKETS - 1).long()

    def forward(self, ids: Tensor, vals: Tensor) -> Tensor:
        """
        Args:
            ids:  (B, len_f) int64
            vals: (B, len_f) float
        Returns:
            (B, D) pooled tensor
        """
        # Mask
        valid = (ids != 0).float()                       # (B, len_f)
        kpm = (ids == 0)                                  # (B, len_f), True=padding for transformer

        # Inject dense
        bucket = self._quantize(vals)                    # (B, len_f) int64
        tokens = self.id_emb(ids) + self.bucket_emb(bucket)   # (B, len_f, D)

        # 1-layer transformer (BERT-style, no causal)
        tokens = self.transformer(tokens, src_key_padding_mask=kpm)   # (B, len_f, D)

        # Mean pool over valid positions
        masked_sum = (tokens * valid.unsqueeze(-1)).sum(dim=1)        # (B, D)
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1)            # (B, 1)
        pool = masked_sum / denom

        return self.out_proj(pool)                                     # (B, D)
```

### 分桶方案细节

**等宽分桶**（不用 quantile，避免预先扫数据存边界常量）：

| fid | 预变换 | 桶分布 |
|---|---|---|
| 62-66 | `b = log1p(clamp_min(v, 0))` ∈ [0, ~21] | 32 桶等宽分布在 [0, 24]；桶 0 给 v=0；多数数据落在桶 8-22 |
| 89-91 | 不变 ∈ [-1, +1] | 32 桶等宽分布在 [-1, +1]；桶 16 附近是 v=0（fid 91 一半数据）|

**为什么等宽不等频**：
- 简化代码：边界全部公式化算（floor((b/24)*32) 或 floor((v+1)/2*32)），不需要把 quantile 边界存常量
- 32 桶足够细：即使 mass 不均匀，多数桶预期至少 ~100 sample（per-batch），emb 能学
- 易于改造：万一某些桶太空导致 emb 学不动，下版（v2.1）改 quantile（用 EDA 已有的 p10/p25/p50/p75/p90/p99 7 个边界 → 8 桶）

### CLI 设计

**复用 `--pair_weighted_pool` 参数**（避免引入新 flag），新增 `transformer` 值：

```
--pair_weighted_pool {none, log1p, full, transformer}
```

- `none`（默认）= baseline mean-pool，不动 user_int_feats 处理
- `log1p` / `full` = v1 weighted pool 路径，**保留但 deprecated**（CLI help 加 `[deprecated, F27 dead]` 标记，避免误用）
- `transformer` = v2 PairSetEncoder 路径

`run.sh` 通过 `PAIR_WEIGHTED_POOL` env var 控制：

```bash
# v2 (本设计)
PAIR_WEIGHTED_POOL=transformer ./run.sh

# v1 (deprecated, 仅历史复现)
PAIR_WEIGHTED_POOL=log1p ./run.sh
PAIR_WEIGHTED_POOL=full  ./run.sh
```

### Wiring（如何接入 PCVRHyFormer）

**当前 v1 路径（保留不动）**：
- `_build_paired_dense(inputs)` 返回 `{fid_idx: (B, length) precomputed weight}` dict
- `user_ns_tokenizer.forward(int_feats, paired_dense=weights)` 在 tokenizer 内部做 weighted pool

**v2 新增 transformer 路径**：

PCVRHyFormer 持有 `nn.ModuleDict[fid_idx → PairSetEncoder]`，在 forward 里 pre-compute 每个 fid 的 (B, D) 池化输出，传给 tokenizer：

```python
class PCVRHyFormer(nn.Module):
    def __init__(self, ..., pair_weight_mode='none', ...):
        ...
        if pair_weight_mode == 'transformer':
            self.pair_set_encoders = nn.ModuleDict({
                str(fid): PairSetEncoder(fid, vocab=int_vocab[fid], d_model=d_model)
                for fid in PAIR_WEIGHTED_FIDS_COUNT + PAIR_WEIGHTED_FIDS_SCORE
                if fid in user_paired_dense_specs and fid in user_int_specs
            })

    def _build_pair_set_pools(self, inputs: ModelInput) -> Optional[dict]:
        """Pre-compute (B, D) pool per fid for transformer mode."""
        if self.pair_weight_mode != 'transformer':
            return None
        pools = {}
        for fid_str, encoder in self.pair_set_encoders.items():
            fid = int(fid_str)
            fid_idx = self._user_int_fid_to_idx[fid]
            ioff, ilen = self._user_int_idx_to_slice[fid_idx]
            doff, dlen = self._user_paired_dense_specs[fid]
            ids = inputs.user_int_feats[:, ioff:ioff + ilen]
            vals = inputs.user_dense_feats[:, doff:doff + dlen]
            pools[fid_idx] = encoder(ids, vals)         # (B, D)
        return pools

    def forward(self, inputs):
        if self.pair_weight_mode == 'transformer':
            paired_pool = self._build_pair_set_pools(inputs)
            user_ns = self.user_ns_tokenizer(inputs.user_int_feats, paired_pool=paired_pool)
        elif self.pair_weight_mode in ('log1p', 'full'):
            paired_dense = self._build_paired_dense(inputs)
            user_ns = self.user_ns_tokenizer(inputs.user_int_feats, paired_dense=paired_dense)
        else:
            user_ns = self.user_ns_tokenizer(inputs.user_int_feats)
        ...
```

**Tokenizer 接口扩展**（`RankMixerNSTokenizer.forward` 和 `GroupNSTokenizer.forward`）：

```python
def forward(self, int_feats, paired_dense=None, paired_pool=None):
    """
    paired_dense: dict {fid_idx: (B, length)} — v1 weighted pool weights
    paired_pool:  dict {fid_idx: (B, D)} — v2 pre-pooled outputs
    """
    for fid_idx in multi_value_fids:
        if paired_pool is not None and fid_idx in paired_pool:
            pooled = paired_pool[fid_idx]              # (B, D), 直接用
        elif paired_dense is not None and fid_idx in paired_dense:
            # v1 weighted pool 路径（保留）
            ...
        else:
            # baseline mean pool 路径
            ...
```

**好处**：
- v1 路径完全不动（zero regression risk）
- v2 路径独立增加，跟 v1 共存
- tokenizer 接口扩展是 additive（默认 paired_pool=None，旧调用不受影响）

### 等价性 / Fallback / 边界情况

| 情况 | 行为 |
|---|---|
| `pair_weight_mode='none'` | PairSetEncoder 不实例化，paired_pool=None，tokenizer 走 baseline mean-pool（**bit-equivalent baseline**）|
| id=0 (padding 位置) | mask 屏蔽 attention（kpm）+ 屏蔽 mean pool（valid mask）；padding 位置的 bucket 也算了一个值但完全不影响输出 |
| valid len=0 (整行 padding) | denom clamp(min=1) 救回，输出 0 向量；下游 user_ns 收到 0 向量等价于"该 fid 无信号" |
| v=0 (dense 全 0) | fid 62-66 → log1p(0)=0 → bucket 0；fid 89-91 → bucket 16；这些桶高频 emb 学得最稳 |
| v 极端（远超 EDA 观测范围）| clamp 到 [0, 31] 兜底；进入边界桶 0 或 31 |
| **fid 91 全 0 率 48%**（v1 病灶）| v2 这一半样本统一进**桶 16**，emb 学得最稳；从 v1 的"非系统性噪声"变成 v2 的"systematic strong prior"——是优势而非问题 |

## EDA 关键事实（驱动设计的硬数据）

来源：`docs/eda/2026-05-01-data-profile.md` Section 5

| fid | dim | min | max | mean | p50 | p99 | std/mean | all_zero |
|---|---|---|---|---|---|---|---|---|
| 62 | 6 | 0 | 2.7e+08 | 56,721 | 16,058 | 557,127 | 2.51 | 6.27% |
| 63 | 19 | 0 | 3.6e+08 | 20,038 | 5,389 | 198,738 | 3.15 | 6.27% |
| 64 | 26 | 0 | 5.4e+08 | 21,154 | 5,608 | 227,940 | 2.74 | 6.27% |
| 65 | 111 | 0 | 1.5e+09 | 8,928 | 1,638 | 102,612 | 4.15 | 7.17% |
| 66 | 150 | 0 | 1.5e+09 | 15,540 | 1,458 | 263,797 | 3.91 | 7.54% |
| 89 | 10 | -0.91 | +0.92 | ~0 | -0.018 | 0.450 | — | 4.13% |
| 90 | 10 | -0.91 | +0.92 | ~0 | -0.021 | 0.436 | — | 7.64% |
| 91 | 10 | -0.92 | +0.93 | ~0 | -0.004 | 0.468 | — | **48.02%** |

## 参数账

| 模块 | 单 fid 参数 | 8 fid 合计 | 备注 |
|---|---|---|---|
| bucket_emb (32 × 64) | 2,048 | 16,384 | 新增 |
| TransformerEncoderLayer (d=64, nhead=4, ff=128, gelu, dropout=0.1, pre-norm) | ~50,000 | ~400,000 | 新增（占比最大）|
| out_proj Linear(D, D) | 4,160 | 33,280 | 新增 |
| **总增量** | **~56K** | **~450K** | |
| baseline PCVRHyFormer | — | ~100M+ | 对比 |
| **增量占比** | — | **~0.4%** | 安全 |

显存增量预估：
- forward activations 峰值 ≈ B × len_max × D × 4 bytes × 几个 layer 中间态
- B=256, len_max=150 (fid 66), D=64 → 256 × 150 × 64 × 4 = ~10 MB per fid，8 fid ~80 MB
- 微不足道（baseline daily driver 12-13 GB）

## A/B 实验计划

**配置**：

| Run | --pair_weighted_pool | seq encoder | 备注 |
|---|---|---|---|
| Baseline | none | transformer (默认) | 对照（baseline AMP+compile val 0.862207 / test 0.812282）|
| v2 | transformer | transformer | 本设计 |

**协议**：
- 6 epoch（baseline 收敛标准）
- 同 seed / 同 lr / 同 batch size / 同所有其他超参
- 在 baseline 同一 GPU 跑（虚拟化负载相似）
- 监控：每 epoch val AUC + 最终 ckpt 提交一次 test

**判决标准**（吸收 F27 教训）：
- ✅ **接受**：val + test **同向**涨 ≥ +0.002（v2.2 反模式：纯靠 val 必反向）
- ❌ **拒绝（v2 paradigm 也死）**：test 反向 ≥ -0.001 → 整条 W2.6 信息层 leg 关闭，转其他金矿（W1.7 子方案 c / W1.10 / W2.7）
- 🔄 **重审视**：val ↑ test 持平（gap 没拉大）→ 不闭环但需要更多 epoch / 更细超参研究

**为什么 v2 更可能成功（vs v1）**：
1. dense 不再直接驱动 pool 选择 → 绕开 v1 的核心病灶
2. bucket 离散 → 对 v 的分布漂移更鲁棒（多数样本不跨桶）
3. id_emb 在 attention 里主导 → id 集合在 test 上不漂移，比 dense 稳定
4. self-attn 让 list 内元素互相看 → 当前 baseline 完全没有的能力

**为什么 v2 仍可能失败**：
1. self-attn 也能学到 spurious id 组合关联（但不再是 dense 漂移直接坏）
2. 1 层 transformer 可能不够（list 内交互复杂度可能需要 2 层）→ 但 2 层风险更大，先看 1 层
3. bucket 等宽可能某些桶太空学不动 → 改 quantile 是 fallback

## 风险 & 缓解

| 风险 | 缓解 |
|---|---|
| v2 也触发 val/test divergence（第 6 次） | A/B 协议强制看 test；val 涨 test 持平/跌 → 立刻关 W2.6 leg，不再尝试更多变种 |
| 等宽分桶导致某些桶 emb 学不动 | v2.1 fallback 到 quantile 分桶（用 EDA p10/p25/p50/p75/p90/p99 作 7 边界 → 8 桶）|
| 1 层 transformer 不够 / 多层过拟合 | v2 起步用 1 层；如果 v2 持平且诊断为"表达力不够"，v2.1 试 2 层（含 dropout=0.2）|
| 显存溢出（fid 66 dim=150 self-attn） | 实测 ~80 MB 增量微不足道，但 H20 19G 并发 batch 还需复核；先 1 epoch 确认显存稳定再跑全 6 epoch |
| transformer block 训练初期不稳定 | pre-norm（`norm_first=True`）+ gelu + dropout 0.1，BERT-base 同款配置，已验证稳定 |
| nn.TransformerEncoderLayer 跟 baseline torch.compile 的兼容性 | baseline 已开 `--use_compile reduce-overhead` + AMP；TransformerEncoderLayer 是 PyTorch 标准模块，compile 兼容性应 OK；首次 epoch 慢一点正常 |
| v1 path 跟 v2 path 在 model.py 里共存导致代码混乱 | 通过 `pair_weight_mode` dispatch 严格隔离；v1 路径完全不动（zero regression risk）；v2 路径独立模块 |

## Out of Scope（明确不做）

- ❌ **Cross-fid self-attention**（fid 间互相看，方案 F）：v2 先验证 per-fid set 信号；v3 候选
- ❌ **Learnable bucket boundaries**：v2 等宽起步；v2.1 fallback quantile；v3 才考虑可学
- ❌ **PMA / Set Transformer learnable seed query**（方案 G）：v2 用 mean pool 起步；如果 v2 持平且诊断为"pool 表达力不够"，v3 试 PMA
- ❌ **多层 transformer**：v2 = 1 层；v2.1 视情况试 2 层
- ❌ **dense fusion 用 FiLM / concat 等其他方式**：v2 = bucket+add；v2.1 视情况试 FiLM
- ❌ **target × user_seq 类的真实 pair embedding**（v0.3.2 W2.6 原始构想）：跟当前 set encoder 设计完全不同方向；v3 候选
- ❌ **fid 15/60/80 等无 dense pair 的多值 int 用 set encoder**：本期范围只覆盖 8 个有 dense pair 的 fid

## 实施 checklist（写给后续 plan / 实施步骤）

- [ ] `model.py`：新增 `PairSetEncoder` class（class 定义 + forward + _quantize）
- [ ] `model.py`：`PCVRHyFormer.__init__` 当 `pair_weight_mode='transformer'` 时实例化 `nn.ModuleDict` of PairSetEncoder
- [ ] `model.py`：`PCVRHyFormer._build_pair_set_pools(inputs)` 方法
- [ ] `model.py`：`PCVRHyFormer.forward` 加 mode dispatch（transformer / v1 weighted / baseline）
- [ ] `model.py`：`RankMixerNSTokenizer.forward` 加 `paired_pool=None` 参数 + dispatch 逻辑
- [ ] `model.py`：`GroupNSTokenizer.forward` 同上（保持两个 tokenizer API 一致）
- [ ] `train.py`：`--pair_weighted_pool` 加 `transformer` choice + CLI help 标 log1p/full 为 deprecated
- [ ] `run.sh`：添加 `PAIR_WEIGHTED_POOL=transformer` 注释示例
- [ ] `tests/test_pair_weighted_pool.py`：新增 PairSetEncoder 测试（forward shape / mask 处理 / pair_weight_mode='none' 等价 baseline / 极端值 clamp / fid 91 全 0 路径）
- [ ] 本地 demo 跑通（`tools/run_demo.sh` mirror 修改）
- [ ] 平台 6 epoch A/B 跑（baseline none vs transformer）
- [ ] 出结果后更新 parent spec：F28 + W2.6 状态决策
