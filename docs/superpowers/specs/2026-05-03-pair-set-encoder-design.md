# W2.6 v2 重启：PairSetEncoder（bucket + 1-layer transformer pool）

**Status**: v1.2 — reset behavior 设计：id_emb 共享 tokenizer（reset 路径自动覆盖）/ bucket_emb 不 reset（time_embedding 类比）
**Author**: brainstorming session 2026-05-03（W2.6 v1 失败 F27 后重启）
**Branch**: `feature/pair-weighted-pool`（continued from v1 weighted pool 实验）
**Parent spec**: `docs/superpowers/specs/2026-05-01-taac-improvement-plan.md`（v2.2 W2.6 v1 已闭环）
**Predecessor (DEAD)**: `docs/superpowers/specs/2026-05-03-pair-feature-design.md`（v0.2 weighted pool；F27 实证 test -0.0022 ~ -0.0054）
**Goal**: 在 v1 weighted pool 失败后重启 W2.6，验证 **set self-attention pool + bucket-emb dense 注入** 能否带来 ≥ +0.002 信号层增量（同时 val + test 同向涨）

## Changelog

- **v1.2（2026-05-03 下午）**：reset behavior 关键设计修订：
  - 🔧 **id_emb 共享 tokenizer 现有 emb 表**（v1.1 是 PairSetEncoder 自持独立表）：避免双 register；走 tokenizer 已有的 reinit_high_cardinality_params 路径，自动覆盖 reset
  - 🔧 **bucket_emb 不进 reinit 路径**（不是疏忽，是设计）：跟 baseline `time_embedding` 同构——semantic 由 quantize 函数定义、跨 epoch 稳定，不会被模型当 shortcut 记忆，可以累积学习
  - 📝 新增"Reset behavior (F15 interaction)" section，论证 time_embedding 类比 + obs/bucket 估算
- **v1.1（2026-05-03 下午）**：自查后修订 wiring：
  - 🔧 **PairSetEncoder 输出维度 emb_dim 而非 d_model**：tokenizer 内部每 fid 的 mean-pool 输出是 `(B, emb_dim)`（不是 d_model），下游有 `Linear(num_fids × emb_dim → d_model)` 投影负责升维。PairSetEncoder 是 mean-pool 的替换，必须保持同样接口
  - 🔧 **train.py gating 需扩展**：`if args.pair_weighted_pool != 'none'` 需改为允许 `'transformer'` 也填充 `user_paired_dense_specs`（否则 PairSetEncoder 拿不到 dense slice）
  - 🔧 **bucket 上界微调**：fid 62-66 用 24 作 log1p 上界余量，fid 65/66 max log1p≈21 → 桶 28，桶 29-31 留给极端 outliers（保留余量优于撑满）
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

**重要：PairSetEncoder 输出 `(B, emb_dim)`**（不是 d_model），跟 tokenizer 内部 per-fid mean-pool 接口一致。下游有 `Linear(num_fids × emb_dim → d_model)` 投影负责拼接后升维。baseline 默认 emb_dim = d_model = 64，但二者概念不同，PairSetEncoder 必须按 emb_dim 设计。

```python
class PairSetEncoder(nn.Module):
    """Per-fid set encoder for paired (int, dense) features.

    Pipeline:
    1. id embedding lookup (emb_dim)
    2. bucket(dense) → bucket embedding lookup (emb_dim)
    3. add: tokens = id_emb + bucket_emb
    4. 1-layer BERT-style transformer block (attn + FFN, no causal mask)
    5. mean pool over valid (non-padding) positions
    6. Linear(emb_dim, emb_dim) project
    """
    NUM_BUCKETS = 32

    def __init__(self, fid: int, vocab: int, emb_dim: int, nhead: int = 4,
                 dim_feedforward: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.fid = fid
        self.emb_dim = emb_dim
        self.id_emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.bucket_emb = nn.Embedding(self.NUM_BUCKETS, emb_dim)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=emb_dim,             # transformer "d_model" = our emb_dim
            nhead=nhead,
            dim_feedforward=dim_feedforward or 2 * emb_dim,
            activation='gelu',           # BERT-style
            dropout=dropout,
            batch_first=True,
            norm_first=True,             # pre-norm，更稳
        )
        self.out_proj = nn.Linear(emb_dim, emb_dim)

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
            (B, emb_dim) pooled tensor — replaces tokenizer's per-fid mean-pool output
        """
        # Mask
        valid = (ids != 0).float()                       # (B, len_f)
        kpm = (ids == 0)                                  # (B, len_f), True=padding for transformer

        # Inject dense
        bucket = self._quantize(vals)                    # (B, len_f) int64
        tokens = self.id_emb(ids) + self.bucket_emb(bucket)   # (B, len_f, emb_dim)

        # 1-layer transformer (BERT-style, no causal)
        tokens = self.transformer(tokens, src_key_padding_mask=kpm)   # (B, len_f, emb_dim)

        # Mean pool over valid positions
        masked_sum = (tokens * valid.unsqueeze(-1)).sum(dim=1)        # (B, emb_dim)
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1)            # (B, 1)
        pool = masked_sum / denom

        return self.out_proj(pool)                                     # (B, emb_dim)
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

PCVRHyFormer 持有 `nn.ModuleDict[fid_idx → PairSetEncoder]`，在 forward 里 pre-compute 每个 fid 的 `(B, emb_dim)` 池化输出，传给 tokenizer。**关键复用**：现有的 `_paired_count_idx_to_slice` / `_paired_score_idx_to_slice` 已经把 `fid_idx → (doff, dlen)` 映射好（dense slice），v2 直接复用，不重复维护。`(ioff, ilen)` int slice 通过 `user_int_feature_specs[fid_idx]` 拿到（PCVRHyFormer 已持有）。

```python
class PCVRHyFormer(nn.Module):
    def __init__(self, ..., pair_weight_mode='none', ...):
        ...
        # v1 modes ('log1p', 'full') 已有路径不动
        # v2 mode ('transformer') 新增 PairSetEncoder dict
        if pair_weight_mode == 'transformer' and (
            self._paired_count_idx_to_slice or self._paired_score_idx_to_slice
        ):
            int_vocab_per_fid = ...   # 从 user_int_feature_specs 拿到 vs (vocab) per fid
            self.pair_set_encoders = nn.ModuleDict({
                str(fid_idx): PairSetEncoder(
                    fid=user_int_fids[fid_idx],
                    vocab=int_vocab_per_fid[fid_idx],
                    emb_dim=emb_dim,
                )
                for fid_idx in (
                    list(self._paired_count_idx_to_slice.keys()) +
                    list(self._paired_score_idx_to_slice.keys())
                )
            })

    def _build_pair_set_pools(self, inputs: ModelInput) -> Optional[dict]:
        """Pre-compute (B, emb_dim) pool per fid for transformer mode."""
        if self.pair_weight_mode != 'transformer':
            return None
        if not hasattr(self, 'pair_set_encoders'):
            return None
        pools = {}
        # 复用已有的 slice 映射（COUNT 和 SCORE 合并迭代）
        all_slices = {**self._paired_count_idx_to_slice, **self._paired_score_idx_to_slice}
        for fid_idx, (doff, dlen) in all_slices.items():
            ioff, ilen = self.user_int_feature_specs[fid_idx]   # (vs, offset, length)
            ids = inputs.user_int_feats[:, ioff:ioff + ilen].long()
            vals = inputs.user_dense_feats[:, doff:doff + dlen]
            pools[fid_idx] = self.pair_set_encoders[str(fid_idx)](ids, vals)   # (B, emb_dim)
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
    paired_pool:  dict {fid_idx: (B, emb_dim)} — v2 pre-pooled outputs
    """
    for fid_idx in multi_value_fids:
        if paired_pool is not None and fid_idx in paired_pool:
            fid_emb = paired_pool[fid_idx]              # (B, emb_dim), 直接用
        elif paired_dense is not None and fid_idx in paired_dense:
            # v1 weighted pool 路径（保留）
            fid_emb = _pool_multivalue(emb_all, vals, paired_dense[fid_idx])
        else:
            # baseline mean pool 路径
            fid_emb = _pool_multivalue(emb_all, vals, None)
```

**train.py CLI gating 扩展**（重要修订）：

`train.py:300` 当前是 `if args.pair_weighted_pool != 'none':` 来填充 `user_paired_dense_specs`。v2 加入 `'transformer'` 后，这个 gating 仍然正确（`'transformer' != 'none'` → 填充 specs），不需要修改逻辑，**只需要把 `'transformer'` 加入 CLI choices**：

```python
# train.py:204
parser.add_argument('--pair_weighted_pool', type=str, default='none',
                    choices=['none', 'log1p', 'full', 'transformer'],
                    help='... transformer = v2 PairSetEncoder (set self-attn + bucket-emb dense)')
```

**好处**：
- v1 路径完全不动（zero regression risk）
- v2 路径独立增加，跟 v1 共存
- 复用已有的 `_paired_*_idx_to_slice` 映射（不引入新数据结构）
- tokenizer 接口扩展是 additive（默认 `paired_pool=None`，旧调用不受影响）

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

baseline emb_dim = d_model = 64（默认值，二者数值相等但概念独立）。所有 PairSetEncoder 内部组件按 emb_dim 算：

| 模块 | 单 fid 参数 | 8 fid 合计 | 备注 |
|---|---|---|---|
| bucket_emb (32 × emb_dim) | 2,048 | 16,384 | 新增 |
| TransformerEncoderLayer (d=emb_dim, nhead=4, ff=128, gelu, dropout=0.1, pre-norm) | ~50,000 | ~400,000 | 新增（占比最大）|
| out_proj Linear(emb_dim, emb_dim) | 4,160 | 33,280 | 新增 |
| **总增量** | **~56K** | **~450K** | |
| baseline PCVRHyFormer | — | ~100M+ | 对比 |
| **增量占比** | — | **~0.4%** | 安全 |

如果将来调高 emb_dim（比如 emb_dim=96 同 F26 实验），参数线性增长但仍可控。

## Reset behavior（F15 reinit 交互——关键设计）

### baseline F15 行为回顾

`reinit_cardinality_threshold=0` 是模型不崩的底线（spec parent F15）。每 epoch 末调 `reinit_high_cardinality_params(0)` 把所有 `vs > 0` 的 nn.Embedding 重置成 xavier_normal_。Run Y (threshold=10000) 实证不 reset 低基数 emb → 模型 epoch 2 崩。

但**不是所有低基数 emb 都需要 reset**。baseline `time_embedding` (vocab=65) 始终被显式排除（`model.py:1715` 注释 `# time_embedding is always preserved`），baseline 工作得很好。

### 区分原则：semantic 是任意学的还是数据定义的

| emb 类型 | vocab 量级 | semantic 来源 | 需要 reset？ | 类比 |
|---|---|---|---|---|
| user_int_feats id_emb (gender / age / device 等) | 小（3-100）| **任意学**（id 0 vs id 1 哪个表示啥纯靠数据）| ✅ 必须 | F15 实证 |
| seq emb（用户行为序列 id）| 中-大（100-1M）| **任意学**（同上）| ✅ 必须 | F15 实证 |
| time_embedding | 65 | **数据定义**（bucket k 永远表示同样的时间区间，quantize 函数锁住）| ❌ 不需要 | baseline 实证 |
| **bucket_emb (W2.6 v2)** | 32 | **数据定义**（bucket k 永远表示同样的 v 区间，`_quantize` 函数锁住）| ❌ 跟 time_embedding 同构 | 本设计 |

### 为什么 bucket_emb 跟 time_embedding 同构

两者都用 quantize 函数把连续值映射到固定语义的桶：
- `time_embedding`: `floor((delta_seconds / max_window) * NUM_BUCKETS)` → bucket k 表示固定时间区间
- `bucket_emb` (COUNT 路径): `floor(log1p(v) / 24 * 32)` → bucket k 表示固定 v 范围
- `bucket_emb` (SCORE 路径): `floor((v + 1) / 2 * 32)` → bucket k 表示固定 v 范围

bucket 编号有内在排序（bucket 5 永远比 bucket 4 表示"更大"的 v / 更久之前的事件）。模型**不能**靠 bucket emb 来"记住特定用户"——bucket 是按 v 切的，不是按用户切的。

### 跟 gender_emb 不同构（为什么 F15 trap 不适用）

F15 失败机制：低基数 emb 被许多 user 共享 → 不 reset → dense 参数把它当 shortcut 记忆 → 跨 epoch 累积 → val 越涨 test 越坏。

bucket_emb **抗这个机制**因为：
1. 每 user 不止 1 次 lookup（fid 66 dim=150 → 一个 user 一次 forward 触 150 个桶 lookup）；不是"per-user 1 个 emb 当身份卡"
2. semantic 锁定（quantize 决定哪些 v 落进哪个桶），不能任意调 bucket emb 内容来 fit 单个样本
3. 桶共享性：bucket 16 在 fid 91 上是 48% user 共享的（v=0 落桶），是"群体共享 prior"，不是"个体 shortcut"

### obs/bucket 估算（说明每 epoch 学得动）

fid 62（dim=6, val 集 907K rows）：
- 总 lookup ≈ 907K × 6 = **5.4M**
- 平摊 32 桶 ≈ **170K obs/bucket**

fid 66（dim=150）：
- 总 lookup ≈ 907K × 150 = **136M**
- 平摊 32 桶 ≈ **4.25M obs/bucket**

参考：
- baseline NS emb（保留的）每 fid 约 1K-3K obs/id（spec 附录 D.9）
- F21 失败的复活方案 fid 29 进 reinit 范围后 55 obs/id 学不动
- bucket_emb 远超学习门槛 → 累积学习是合理的

### id_emb 共享 tokenizer：让现有 reset 路径自动覆盖

PairSetEncoder.id_emb 是**任意学 semantic**（同 gender_emb），按 F15 必须 reset。最干净的实现是**共享 tokenizer 现有的 emb 表**（不是 PairSetEncoder 自己再建一份）：

```python
# PCVRHyFormer.__init__:
shared_emb = self.user_ns_tokenizer.embs[real_idx]  # tokenizer 已建的表
self.pair_set_encoders[str(fid_idx)] = PairSetEncoder(
    fid=fid, vocab=int(vs), emb_dim=emb_dim,
    id_emb_module=shared_emb,  # 共享，不是复制
)
```

PairSetEncoder 内部用 list 包裹避免双 register（不然 optimizer 看到同一参数两次）：

```python
class PairSetEncoder(nn.Module):
    def __init__(self, ..., id_emb_module=None):
        super().__init__()
        if id_emb_module is None:
            self.id_emb = nn.Embedding(vocab + 1, emb_dim, padding_idx=0)
            self._shared_id_emb = None
        else:
            self.id_emb = None  # 不持有 ParameterContainer
            self._shared_id_emb = [id_emb_module]  # list 绕过 nn.Module 注册
        self.bucket_emb = nn.Embedding(self.NUM_BUCKETS, emb_dim)  # 自有，不 reset

    def _id_emb_lookup(self, ids):
        return (self._shared_id_emb[0] if self._shared_id_emb else self.id_emb)(ids)
```

**好处**：
1. 一份 emb 表 / fid（省 ~5M 参数 vs 自持）
2. v1 weighted-pool 路径用的是 tokenizer 那张；v2 transformer 路径也是同一张 → 模式之间公平对比
3. tokenizer 已有的 `reinit_high_cardinality_params` 自动覆盖 id_emb reset，不需要改 reinit 代码
4. forward / weighted-pool / set-encoder 三条路径 emb lookup 一致

### 总结：v2 各组件 reset 行为

| 组件 | reset 每 epoch？| 理由 |
|---|---|---|
| `id_emb`（tokenizer 共享）| ✅ 是 | 任意 semantic，跨 epoch 不 reset 会被 dense 当 shortcut 记忆（F15）|
| `bucket_emb`（PairSetEncoder 自有）| ❌ 否 | 数据定义 semantic（quantize 锁住），跟 time_embedding 同构 |
| `transformer` block 权重 | ❌ 否 | dense 参数累积学习（baseline transformer / RankMixer 一致）|
| `out_proj` Linear | ❌ 否 | 同上 |

### 风险

如果 bucket_emb 不 reset 实测仍引入 val/test divergence，回退方案是把它加进 reinit 路径（vs > 0 触发，等同 id_emb 处理）。但理论上不应触发，因为 bucket semantic 锁死无法 shortcut。

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
