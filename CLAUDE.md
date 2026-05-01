# TAAC 2026 PCVR Baseline

**Comp**: WWW 2026 Tencent Advertising Algorithm Competition — "Towards Unifying Sequence Modeling and Feature Interaction for Large-scale Recommendation"
**Task**: PCVR (Post-Click Conversion Rate) binary classification, metric = AUC
**Deadline**: 2026-05-23 AOE
**Goal**: 单模型 test AUC 从 0.811492 → ≥ 0.825（top1 当前 0.82879）
**Constraint**: 单模型 only（ensemble 比赛规则疑似禁止）

## 主要 spec（先读这个）

`docs/superpowers/specs/2026-05-01-taac-improvement-plan.md` —— 当前 v0.3，包含 W1-W3 完整作战图、关键事实表（F1-F17）、附录 A-D。**任何后续工作都从这份 spec 起步**，不要重新规划。

## 仓库布局

```
src/                  ← 平台上传的代码（除 taac2026_schema.json）
├── model.py          PCVRHyFormer 架构（multi-domain seq encoder + RankMixer + dual optimizer）
├── dataset.py        IterableDataset，按 row group 切 train/val
├── trainer.py        训练循环 + AMP + reinit
├── train.py          CLI 入口
├── run.sh            平台启动脚本
├── utils.py          EarlyStopping / 日志
├── ns_groups.json    NS grouping 配置（当前 run.sh 禁用它）
└── taac2026_schema.json   ⚠️ 本地 codex 生成的 mock，**不要上传**

data/demo/            1000 行 demo 数据 + mock schema.json（codex 生成，不是线上 schema）
docs/superpowers/specs/   spec 文档主目录
output/                本地 ckpt / log / tensorboard（gitignored）
```

## 算力 & 性能基线

- **硬件**：2×H20-20%×19G，**time-slicing 虚拟化**（半夜独占可冲到 500%，白天稳定 20%）
- **每日 test 提交配额**：3 次，**省着用**
- **Baseline 速度**（v0.3.2 实测）：
  - 原始 fp32：52 min/epoch、15-16G 显存、val 0.862173 / test 0.811492
  - AMP only：34 min/epoch（-35%）、10-11G、val 0.862202
  - compile only：20 min/epoch（-62%）、15-16G、val 0.862024
  - **AMP+compile（daily driver）：23 min/epoch（-56%）、12-13G、val 0.862207 / test 0.812282**（test +0.00079 比原 baseline）
- **A/B 启发法**：6 epoch 收敛 ≈ 每 epoch +0.001；**3 epoch 足够诊断 ≥ 0.0015 量级差异**，可砍 50% 算力
- **实验预算**：理论 30-50 次完整训练；考虑虚拟化高峰损耗按 **15-25 次** 算

## 当前活跃工作

**Branch**：`feature/compile-training`（已合并 AMP / compile / tf32）
**最高优先级**：W1.0.2（dense_wd）+ W1.0.3（修 LongerEncoder）+ W1.10 spot-check
**已完成**：W1.0.1 reinit A/B（threshold=10000 掉 0.0015 → 保留默认 0）
**待跑实验**：
1. AMP+compile 叠加 baseline（确认两个一起开不冲突）
2. 用 1 次每日 test 提交验证 AMP+compile 不掉点
3. W1.10 高基数特征复活 A/B（W1.0.3 修完后并行进行）

## 已知 baseline 问题（v0.3.2 spec F15-F17）

未修，影响后续实验解读：

1. ✅🔥 **F15 / W1.0.1 已结，结论比预期重**：A/B 实验显示 threshold=10000 不只是"略差 0.0015"——Run Y **best val 卡在 epoch 2（0.857339），epoch 3 起反向下降直到 EarlyStopping**。Run X（默认 0）val 6 epoch 单调涨到 0.862207。**reinit threshold=0 不是"刻意激进的优化"，是模型不崩的底线**。**绝对不要碰这条线**。CLI help（`train.py:158`）写的"0=never reset"是错的，但代码行为对，留着别动。
2. **F16 / W1.0.3**：`model.py:691` LongerEncoder 取序列尾部，但序列倒序（pos 0=最近），实际取的是**最老**而非最新。当前 transformer 没触发，W1.7 走 longer 必须先修。
3. **F17 / W1.0.2**：`trainer.py:87` AdamW 没暴露 dense `weight_decay`（用 PyTorch 默认 0.01）。W2.1 扫参前要加 `--dense_weight_decay` CLI。

## v0.3.2 战略提示

- **baseline 已在过拟合悬崖上**——靠 reinit threshold=0 续命
- W2 策略：**信息层（W1.7 长序列、W1.10 复活高基数、W2.6 target∈seq 交互特征）优先于正则化层（W2.1-2.5）**
- 正则化层每个 trick 期望收益降到 +0.0005~0.002（不是原来的 0.001~0.003）
- 不要往 baseline 身上再压重正则化（dropout 0.3 / wd 1e-2 大概率推过悬崖）

## 数据关键事实

- 序列**倒序**：pos 0 = 最近（数据分析报告确认）
- train 集**无 OOV**（vocab 按 train max+1 建），test 集**有 OOV**
- demo 数据 row group 时间范围全部重叠 → 按 RG 切 ≈ 按行随机切；**线上数据待 W1.9 验证**
- 4 个被 `emb_skip_threshold=1M` 跳过的高基数 seq 特征（fid 29/34/47/69 vocab 1M-86M），forward 返回零向量。复活方案见 spec 附录 D。
- 线上 schema 已收到（v0.2），ts_fid 设置正确（39/67/27/26），list dim 比 demo 大 1.4-2.3×

## 工程规范

- **平台上传**：只上传改过的文件（当前 = `src/trainer.py` + `src/train.py` + `src/run.sh`）。**不上传** `taac2026_schema.json` / `__pycache__/` / `output/`
- **schema 来源**：`train.py:227` 默认从 `data_dir/schema.json` 读，平台用平台自己的 schema；本地 `src/taac2026_schema.json` 仅作参考
- **W1.10 hash trick 实施位置**：在 `dataset.py` 里覆盖 schema vocab + 应用 modulo，**不要改本地 schema 文件**
- **A/B 实验隔离**：用 `CUDA_VISIBLE_DEVICES` + 不同 `TRAIN_CKPT_PATH/LOG_PATH/TF_EVENTS_PATH` env var

## 反模式（不要做）

- ❌ **模型 scaling**（d_model / num_layers）：当前 val/test gap 0.051 是过拟合，加大模型只会更糟
- ❌ **碰 reinit_cardinality_threshold**：v0.3.2 已证它是模型不崩的底线，往任何方向改（10000、100、1000000）都会让训练 epoch 2 后崩
- ❌ **再往 baseline 身上压重正则化**（dropout=0.3、wd=1e-2 等极端值）：已在过拟合悬崖上，更强正则会推过悬崖
- ❌ **白天跑长实验**：被抢卡严重，跑实验尽量挪到半夜
- ❌ **并行 A/B 跑同一张物理卡**：用 `CUDA_VISIBLE_DEVICES` 隔离到不同 GPU
- ❌ **改 `taac2026_schema.json` 来做特征工程**：它不上传，改了没用

## 协作约定

- spec 是单一事实源，对话冲突时以 spec 为准
- spec 改动按 v0.X 版本递增，加 changelog 条目
- 关键事实新增进事实表 F# 编号，不要散落在正文
- 实验结果出来后**先更新 spec**再继续做下一个实验
