# Warmup-LR Design — feature/warmup-lr

**Date**: 2026-05-02
**Branch**: `feature/warmup-lr`
**Status**: Approved (design phase)
**Parent spec**: `2026-05-01-taac-improvement-plan.md` (W2.x 正则化层)

## 一句话目标

给 dense AdamW 加 **linear warmup + cosine decay**，做成默认开启的开关；
baseline 现在是常数 lr，没有 LR scheduler 任何形式；
本支只验 schedule 形状，**不动** sparse Adagrad、reinit、peak_lr 任何其它东西。

## 动机

- baseline (v0.3.2 daily driver, AMP+compile) val 0.862207 / test 0.812282，按 spec v0.3.2 已在过拟合悬崖上。
- W2.1 正则化层期望 +0.0005~0.002，warmup 是其中影响实施面最小的一项。
- 主要预期收益：
  - 早期 dense 梯度爆炸风险下降（current `clip_grad_norm_(1.0)` 是事后兜底）
  - 让 sparse embedding 在 dense 大幅更新前先稳定（Adagrad lr=0.05 vs AdamW lr=1e-4 量级差 500×）
  - cosine 后段把有效 lr 降下来，给 reinit-after-epoch-1 的高基数 embedding 留 fine-tune 空间

## 非目标

- ❌ 不改 sparse Adagrad lr / 不给 sparse 加 schedule
- ❌ 不动 `reinit_cardinality_threshold`（v0.3.2 F15 已禁碰）
- ❌ 不 sweep peak_lr（dense lr 仍 1e-4），这支只看 schedule 形状是否单独有效
- ❌ 不引入 LR scheduler 抽象（不抽象成可插拔模块），单纯一个 LambdaLR

## CLI 设计（`train.py` 新增 3 个 flag）

| flag | type | default | 含义 |
|---|---|---|---|
| `--warmup_steps` | int | `500` | linear ramp 0→peak 步数。**`0` = 关闭整个 schedule**，回到 baseline 常数 lr 行为 |
| `--cosine_total_epochs` | float | `8.0` | cosine 终点位置 = `epochs × len(train_loader)` 步。8 epoch 对齐典型 early-stop 收敛点 |
| `--cosine_min_lr_ratio` | float | `0.1` | cosine 谷底 lr = `peak × ratio`。peak=1e-4 时谷底=1e-5。到达谷底后保持不再降 |

`--warmup_steps 0` 即"开关关"：scheduler 不构建、不调 step、零行为差异。
`run.sh` 不改，所以默认 ON。

## 行为规范

设 `peak = args.lr`、`W = warmup_steps`、`T = cosine_total_epochs × len(train_loader)`、`r = cosine_min_lr_ratio`。
**约束**：`T > W`，否则 raise（配置错误，不静默退化）。

LR multiplier `f(step)`：

```
if step < W:                            f = step / W
elif step < T:  progress = (step - W) / (T - W)
                f = r + (1 - r) * 0.5 * (1 + cos(pi * progress))
else:                                   f = r
```

应用到 dense AdamW；sparse Adagrad 不变。

## 实现位点

### `trainer.py`

- `__init__` 新增参数：`warmup_steps`、`cosine_total_epochs`、`cosine_min_lr_ratio`
- `__init__` 末尾按上面公式构建 `torch.optim.lr_scheduler.LambdaLR(self.dense_optimizer, lr_lambda=...)`
- 当 `warmup_steps == 0`：`self.lr_scheduler = None`，所有调用点用 `if self.lr_scheduler is not None` 守卫
- `_train_step` 中：在 `self.scaler.step(self.dense_optimizer)` 之后、`self.scaler.update()` 之前调 `self.lr_scheduler.step()`（PyTorch 在用 GradScaler 时官方推荐这样：先 unscale + step optimizer，再 step scheduler）
- TensorBoard：每 `--eval_every_n_steps` 或每 epoch 末（沿用现有 cadence）记 `LR/dense = self.dense_optimizer.param_groups[0]['lr']`

### `train.py`

- 3 个 flag 加入 argparse
- 透传给 trainer 构造
- `total_steps` 在 train.py 里算（需要 `len(train_loader)`）传给 trainer，避免 trainer 反算

### `run.sh`

不改。默认 warmup ON。

## A/B 协议（睡前提交版）

| Run | 命令差异 | 期望 |
|---|---|---|
| W (warmup ON) | 默认 `--warmup_steps 500 --cosine_total_epochs 8.0 --cosine_min_lr_ratio 0.1` | val ≥ 0.862207 + ε（ε 待观察）|
| N (baseline 对照) | `--warmup_steps 0` | val ≈ 0.862207（再现 daily driver baseline）|

隔离方式：`CUDA_VISIBLE_DEVICES` + 独立 `TRAIN_CKPT_PATH / TRAIN_LOG_PATH / TRAIN_TF_EVENTS_PATH`。

按 spec v0.3 启发法："3 epoch 足够诊断 ≥ 0.0015 量级差异"。
但 warmup 收益本身可能 < 0.0015，所以**这次跑满 6 epoch（不要中断）**。
单次 ≈ 23 min × 6 ≈ 2.3h × 2 runs = ~5h，半夜跑可。

成功判据：
- W val 高于 N **≥ 0.0008**（差距明显大于跑间噪声）
- W 训练曲线显示 epoch 1 末 val 不低于 N epoch 1 末 val 太多（warmup 不应严重拖慢早期收敛）
- W 无 NaN、无 loss 爆炸

## 风险与已知 corner case

1. **`len(train_loader)` 于 IterableDataset**：现有 dataset 已正确实现 `__len__`（tqdm 用了），可信。
2. **`torch.compile` 与 `LambdaLR` 兼容性**：scheduler 不在编译图内（只改 optimizer.param_groups[0]['lr']），不会触发重编译。低风险。
3. **AMP scaler.step 跳过 optimizer 时 scheduler.step 仍会推进**：当前 trainer scaler enabled=False（bf16 不需要 scaling），scaler 永远是 pass-through，不会跳过 optimizer step。所以 scheduler.step 无条件调用安全。
4. **early stop 在 `T` 之前触发**：cosine 只走完一部分。可接受 —— 这是设计选择（`T=8` 故意略大于典型 stop 点）。
5. **🟡 cosine decay × per-epoch sparse reset 交互**（v0.2 补充分析）：
   - 当前 baseline `reinit_sparse_after_epoch=1` + `reinit_cardinality_threshold=0` 表现：每 epoch 末 95/96 个 sparse embedding 被 wipe，Adagrad 状态清零，sparse_lr=0.05 不变。Schedule 不影响 sparse，只影响 dense。
   - 实际 lr 轨迹（W=500, T=8 epoch≈28320 步, r=0.1, 3540 steps/epoch）：
     - epoch 1 末 9.7e-5；epoch 4 末 5.6e-5；epoch 6 末 2.4e-5；epoch 8 末 1e-5
   - **风险**：late epoch（7-8）reset 之后，sparse 用 Adagrad lr=0.05 一轮内能重新学到 representation，但 dense lr ≤ 1.4e-5 几乎不动，没法 refine 头部去 match 新 sparse → 那一轮 val 大概率持平甚至倒退。
   - **反向考虑**：低 lr 本身也是 anti-overfit 手段（与 reset 同向），两者也可能互补而非互冲。
   - **决策**（2026-05-02）：本次 A/B **接受这个交互风险不做缓解**，原因：
     1. baseline early-stop 多落在 epoch 5-6，那时 lr 还有 2.4-3.9e-5 还能动
     2. 想以最干净的形式测整套 schedule（warmup+cosine）这一组合的整体效果
     3. 如果失败再单独剥离 warmup→constant 测，不浪费今晚 GPU 时间
   - **诊断信号**（看 TB 时重点关注）：
     - W vs N 的 early-stop epoch（W 早停说明 cosine 拖死了后期）
     - W 的 best AUC 落在哪个 epoch（如果在 epoch 1-2，就是后期反向走的证据）
     - W 在 epoch 5-6 之后 val 是否 plateau / 倒退

## Out-of-scope follow-ups

- 如果 W 跑赢，下一步可考虑：
  - sweep `warmup_steps ∈ {200, 1000}`
  - sweep `cosine_min_lr_ratio ∈ {0.0, 0.01, 0.1, 0.3}`
  - 加 sparse Adagrad warmup（与本设计正交）
- 如果 W 跑输或持平：放弃这条路，不要进一步加 schedule 复杂度，转 W2 信息层（W1.7/W1.10/W2.6）。

## Spec changelog

- v0.1 (2026-05-02): 初版，与 user brainstorm 确认（schedule shape = warmup→cosine、scope = AdamW only、warmup_steps=500、cosine_total_epochs=8、min_lr_ratio=0.1）
- v0.2 (2026-05-02): 风险章节补充 cosine decay × per-epoch sparse reset 交互分析。决策保持原计划不缓解（接受风险，便于干净诊断），加诊断信号 checklist。代码与 CLI 默认值不变。
