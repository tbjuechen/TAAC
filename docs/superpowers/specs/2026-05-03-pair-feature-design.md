# W2.6 重写：(int, dense) 平行对的 value-加权 pool 设计

**Status**: v0.2 — fid 89-91 sigmoid 路径加入；CLI 扩展为 {none, log1p, full}
**Author**: brainstorming session 2026-05-03
**Branch**: `feature/pair-weighted-pool`（cut from `main` @ `ba8bded`）
**Parent spec**: `docs/superpowers/specs/2026-05-01-taac-improvement-plan.md`（v2.0 W2.6 重定义）
**Goal**: 把 user_int_feats 中 8 个多值离散特征（fid 62-66 + 89-91）的 mean-pool 升级为由对齐 dense 数组驱动的加权 pool（per-fid transform：62-66 log1p / 89-91 sigmoid），验证 +0.0015 量级增益

## Changelog

- **v0.2（2026-05-03 晚）**：用户改主意，**fid 89-91 sigmoid 路径并入本期**（不再是 W2.6.5 单独任务）。设计调整：
  - 新增 `PAIR_WEIGHTED_FIDS_SCORE = [89, 90, 91]`（旧 `PAIR_WEIGHTED_FIDS` 保留为 COUNT 的 alias）
  - CLI 扩展为 `--pair_weighted_pool {none, log1p, full}`：log1p = 仅 62-66；full = 62-66 log1p + 89-91 sigmoid
  - **API 重构**：`paired_dense` 语义从"原始 dense 值"改为"已计算好的权重"；helper 不再做 transform，调用方（PCVRHyFormer.\_build\_paired\_dense）按 mode 分派 log1p / sigmoid 后传入。weight_mode 参数从 helper 移除（None vs not None 即可表达）。**好处**：未来加 transform（abs / FiLM / learnable）只改调用方，不动 helper / tokenizer 接口
  - 测试从 9 → 14：新增 sigmoid happy path、负值仍贡献验证、PCVRHyFormer uniform/log1p/full 三模式 dispatch 测试
  - run.sh 新增 `PAIR_WEIGHTED_POOL=full` 选项注释
  - **A/B 计划升级**：现在是 baseline / log1p / full **三档 A/B**（见下方 A/B 节）
- **v0.1（2026-05-03）**：初稿。EDA 锁定 fid 62-66 = 重尾正值（log1p 适合）/ fid 89-91 = 双边 bounded（log1p 不适合，本期保持 uniform）。范围确认 A1（只动 int 侧 pool，dense 侧不动）

## 背景

### 数据结构（官方信息）

`user_dense_feats` 共 10 列：
- **fid 61, 87**：纯用户级 embedding（SUM / LMF4Ads），不与离散特征配对
- **fid 62-66, 89-91**：与 `user_int_feats_{62-66, 89-91}` 同 fid 同长度，**逐元素对齐**

官方语义说明（用户引官方文档）：

> The 1st element in user_int_feats_62 (value 1) denotes a specific entity or category, while the 1st element in user_dense_feats_62 (value 10.5) provides some statistics for that element, such as a dwell time, a score/probability.

即 dense 数组是离散数组每个元素的"强度/统计"标签。

### 当前处理方式（baseline 缺陷）

`model.py` 的 `GroupNSTokenizer.forward` (line 1057-1063) 与 `RankMixerNSTokenizer.forward` (line 1170-1174)：
- 多值 int feat 对 `(B, length)` 做 shared embedding lookup → `(B, length, emb_dim)`
- 用 `mask = (vals != 0).float()` 做**均匀** mean-pool

dense 部分则被全部 concat 进单一 `user_dense` token（918-d → projection）。

**结果**：(int, dense) 配对的对齐信号被两个独立路径分别消化：
1. int 侧丢掉了 "哪个元素更重要" 的位置权重信息（每个元素权重 = 1/n）
2. dense 侧保留了原始数值，但失去了"这个数值对应哪个 entity id"的对应关系

## EDA 关键发现（驱动设计的硬事实）

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

**二分**：
- **fid 62-66**：min=0（全非负）、max/p99 ≈ 500-5000×（极重尾）、p99/p50 ≈ 30-180× → log1p 数学上适合
- **fid 89-91**：min ≈ -0.92 / max ≈ +0.92、居中分布、bounded ~[-1, +1] → log1p 会截掉负值，**不适合**

**注**：fid 62-66 数值形态像累计计数 / dwell time / probability sum；fid 89-91 形态像 cosine sim / 归一化分数。但比赛匿名特征，**语义不能确证**——驱动决策的是分布形态，不是语义猜测。

## 设计

### 范围（A1 决策）

- **改**：`user_int_feats_{62, 63, 64, 65, 66}` 的多值 mean-pool 升级为 `user_dense_feats_{62-66}` 驱动的 log1p-weighted pool
- **不改**：
  - fid 89-91（log1p 不适用，且 fid 91 一半样本全 0；本期保持 uniform mean-pool）
  - fid 15/60/80 多值 int（无 dense 配对）
  - fid 61/87 纯 dense
  - **整个 user_dense token 的输出**（A1 决策：dense 数据全部保留进 dense token，weighted pool 只是"读"它，不"夺走"）

### 公式

对每个 paired fid `f ∈ {62, 63, 64, 65, 66}`：

```
ids   = int_feats[:, off_f : off_f + len_f]              # (B, len_f), int64
vals  = dense_feats[:, doff_f : doff_f + len_f]          # (B, len_f), float
emb_i = E_f(ids)                                          # (B, len_f, D)
mask  = (ids != 0).float()                                # (B, len_f)
w     = log1p(clamp_min(vals, 0)) * mask                  # (B, len_f)
W     = w.sum(dim=1, keepdim=True)                        # (B, 1)

pool  = where(W > eps,
              (emb_i * w.unsqueeze(-1)).sum(dim=1) / W,
              # fallback: dense 全 0 但 id 有效 → 退化到 mean-pool
              (emb_i * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp(min=1))
```

`eps = 1e-8`。Fallback 路径保证：
- dense 全 0 但 id 有效 → 等价于当前 mean-pool
- id 也全 0（unknown user）→ 退化到零向量（`mask.sum=0`，但被 clamp(min=1) 救回，乘 0 mask 后仍 = 0）

### 与当前 mean-pool 的等价性

当 `weight_mode='uniform'`（CLI 默认）：跳过整个 weighted 分支，沿用当前代码路径，**逐 bit 等价于 baseline**。这是 A/B 严格对照的前提。

### Wiring

**改动文件**：仅 `model.py`、`train.py`，**不动** `dataset.py`。

1. **`model.py: GroupNSTokenizer.forward`** 与 **`RankMixerNSTokenizer.forward`**
   - 增加可选参数 `paired_dense: Optional[Dict[int, Tensor]]`（key=fid_idx 在 feature_specs 中的位置，value=`(B, len_f)` dense 切片）
   - 内部循环遍历 fid 时，若 `fid_idx ∈ paired_dense and weight_mode == 'log1p'`，走 weighted pool 分支；否则原 mean-pool

2. **`model.py: PCVRHyFormer.forward`**（line 1637, 1680）
   - 在调用 `user_ns_tokenizer` 前，从 `inputs.user_dense_feats` 切出 fid 62-66 的对应段，组装 `paired_dense` dict
   - 切片所需 `(doff, dlen)` 由 `dataset.py` 已建好的 `user_dense_schema` 提供（通过 model 构造时传入的 `user_dense_specs`）
   - 注意：`PCVRHyFormer.__init__` 需要新增 `user_paired_dense_specs` 参数，包含 fid → (doff, dlen) 映射

3. **`train.py`**
   - 新增 CLI flag `--pair_weighted_pool {none,log1p}`，默认 `none`
   - 从 `dataset.user_dense_schema` 提取 fid 62-66 的偏移信息，构造 `user_paired_dense_specs` 传入 model
   - 配对 fid 列表硬编码 `PAIR_WEIGHTED_FIDS = [62, 63, 64, 65, 66]`

### CLI / 配置

```
--pair_weighted_pool {none,log1p}   # default: none
```

- `none`：完全不计算/传递 paired_dense，tokenizer 走原路径（baseline）
- `log1p`：fid 62-66 启用 weighted pool；fid 89-91 即使配对也走 uniform（硬编码不参与）

不引入 per-fid 配置 CLI——5 个 fid 处理方式一致（log1p），简单优于灵活。未来若 fid 89-91 也想试 sigmoid/abs，再扩展。

## A/B 计划

### Run X — baseline
```
--pair_weighted_pool none
```
（与 main 当前行为完全等价；可选跳过此 run，复用最近一次干净 baseline）

### Run Y — treatment
```
--pair_weighted_pool log1p
```

### 隔离
- `CUDA_VISIBLE_DEVICES` 隔离到不同 GPU
- 不同 `TRAIN_CKPT_PATH / LOG_PATH / TF_EVENTS_PATH` env var
- 同一份代码，仅 CLI flag 切换

### 判定（按 CLAUDE.md A/B 启发法）

3 epoch 诊断（≥ 0.0015 量级足够分辨）：

| Δval | Δtest | 判定 |
|---|---|---|
| ≥ +0.0015 | ≥ +0.0015 | **同向涨** → 推进 6 epoch 完整训练 + 上 test 校准 + 进事实表 F# |
| ≥ +0.0015 | ≤ -0.0010 | **val/test divergence** → 同 F19/F21/F23 模式，**封死回归**，不投后续算力 |
| 双向 \|Δ\| ≤ 0.001 | — | **中性** → 登记低 ROI，回退 baseline |
| Δval < 0 | — | **负向** → 立即停，记录 |

**test 提交预算**：完整训练后消耗 1 次（每日 3 次配额的 1/3）

### 实施时机

- **必须等**：E4 激进版结果出来（CLAUDE.md 当前最高优先级）
- **不冲突**：本设计改 NSTokenizer，与 LongerEncoder（W1.7）、emb_skip 复活（W1.10）、时间特征（W2.7）均独立——E4 之后可与其他金矿候选并行（不同 GPU）

## 非目标（明确不做）

- **方案 B（FiLM 调制）/ 方案 C（concat-MLP 融合）**：先看方案 A 增量，A 涨了再决定值不值得加
- **fid 89-91 weighted 处理**：本期 uniform，等 A 跑通后单开任务（W2.6.5？）
- **dense token 重组（A2/A3 路径：dense 抽出去专职供 pair）**：A1 决策——保留信号冗余
- **item 侧多值配对**：schema 显示 item_dense 为空，无配对可做
- **W2.6 v2.0 spec 文本里的 "target × seq binary indicator" 路径**：那是另一个题目（target 与 user 历史 item_id 的交互），跟 (int, dense) 平行对建模不冲突。本期把 W2.6 重定义为 "(int, dense) 平行对"；binary indicator 如要做单立 W2.6.5 任务

## 风险 / 已知未确证

1. **fid 62-66 真实语义未确证**：分布形态像 dwell time / count，但匿名特征不能确证。log1p 在"非负重尾"下数学合理，若实际是别的（如 z-score 异常值之类），log1p 仍可压缩但非最优。**Mitigation**：A 跑通后，方案 B (FiLM) 不假设语义、能让模型自己学非线性变换，作为下一步保险

2. **val/test divergence 风险**：F19/F21/F23 三次 baseline 改动都掉进了这个陷阱（val ↑ test ↓）。**Mitigation**：A/B 判定表强制要求双向同向涨；不能纯靠 val 判定

3. **fid 65/66 max 高达 1.5e+09**：log1p 后 max 权重 ≈ log(1.5e9) ≈ 21.1。pool 后 LayerNorm 会压回——量级 OK，但如果 pool 输入分布漂移过大，LayerNorm 后续模块的输入分布也会动。**Mitigation**：3 epoch 诊断阶段观察 train loss 曲线是否平稳

4. **fid 62-66 全 0 率 6-7%**：fallback 分支必然被高频触发。**Mitigation**：编写单元测试覆盖 (a) 全 dense 0 但 id 有效、(b) id 也全 0、(c) 部分位置 dense=0（id 非 0）三种情况

5. **fid 91 不在改动范围内**（48% 全 0），无影响

## 期望增益

| 情形 | Δtest |
|---|---|
| 乐观（weighted pool 解锁了 dense token 没吃干净的对齐信号） | +0.002 ~ +0.005 |
| 现实预期（baseline 已用 dense token 吃了大部分信号） | +0.0010 ~ +0.0030 |
| 悲观（dense token 已基本吃尽） | ≤ +0.001，归类低 ROI |
| 反向（val/test divergence） | val ↑ test ↓，封死 |

参考点：spec v2.0 W2.6 binary 占位预期 +0.0005 ~ +0.005；本设计是更细粒度的 weighted 信号，理论上不低于 binary。

## 单元测试要点

新写 `tests/test_pair_weighted_pool.py`（或在已有测试目录增量）：

1. **等价性测试**：`weight_mode='uniform'` 输出与原 mean-pool 数值逐 bit 一致
2. **加权正确性**：构造已知 (ids, vals)，手算 log1p-weighted pool，对比模型输出
3. **fallback 触发**：dense 全 0 但 id 非 0 → 输出等于 mean-pool 结果
4. **零向量边界**：id 全 0 → 输出全零
5. **Mask 正确性**：部分 padding（id=0 在尾部）→ pool 不被 padding 污染
6. **数值稳定**：fid 65/66 量级（vals max=1.5e+09）→ 输出无 inf/nan

## 实施步骤（待 writing-plans skill 细化）

1. 写公式 + 单元测试（先红后绿）
2. 改 `model.py` 两个 NSTokenizer
3. 改 `model.py: PCVRHyFormer.forward` wiring
4. 改 `model.py: PCVRHyFormer.__init__` 接受 `user_paired_dense_specs`
5. 改 `train.py` 加 CLI flag + 构造 paired_dense_specs
6. demo 数据上跑 1 个 mini step 验证 forward 不崩
7. 准备 A/B run.sh 配置（two scripts or env var switch）
8. 等 E4 出 → kick A/B 3-epoch 诊断
9. 出结果 → 更新本 spec changelog + 主 spec 加 F# + CLAUDE.md 状态同步

## 引用

- 主 spec：`docs/superpowers/specs/2026-05-01-taac-improvement-plan.md` (v2.0)
- EDA：`docs/eda/2026-05-01-data-profile.md` Section 5（user_dense per fid）
- 现状代码：`src/model.py:1034-1067 (GroupNSTokenizer)` / `src/model.py:1148-1189 (RankMixerNSTokenizer)` / `src/model.py:1637, 1680 (PCVRHyFormer.forward)`
- Schema：`src/taac2026_schema.json`（mock，仅参考；线上 schema 由平台 data_dir 提供）
- A/B 启发法：CLAUDE.md "算力 & 性能基线" 节
- 反模式 / 历史失败教训：CLAUDE.md F19 / F21 / F23（val/test divergence）
