# TAAC 2026 PCVR Baseline

**Comp**: WWW 2026 Tencent Advertising Algorithm Competition — "Towards Unifying Sequence Modeling and Feature Interaction for Large-scale Recommendation"
**Task**: PCVR (Post-Click Conversion Rate) binary classification, metric = AUC
**Deadline**: 2026-05-23 AOE
**Goal**: 单模型 test AUC 从 0.811492 → ≥ 0.825（top1 当前 0.82879）
**Constraint**: 单模型 only（ensemble 比赛规则疑似禁止）

## 主要 spec（先读这个）

`docs/superpowers/specs/2026-05-01-taac-improvement-plan.md` —— 当前 **v2.0**，包含 W1-W3 完整作战图、关键事实表（**F1-F24**）、附录 A-D。**任何后续工作都从这份 spec 起步**，不要重新规划。

**EDA 报告**：`docs/eda/2026-05-01-data-profile.md` + `.json`（907K val rows 全数据画像，4 方向决策依据齐全）

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

## 当前活跃工作（v2.0 / 5/3 早状态）

**Branch**：`main`（`feature/longer-gather-fix` 已实现 W1.0.3 修复，**未 push**，等 E4 出结果一并 push）
**最高优先级**：**等 E4 激进版（top_k=100, seq 512/512/1024/2048）跑完** → 决定 W1.7 longer 路径生死
**已完成**（按时间）：
- W1.0.1 reinit A/B（threshold=10000 掉 0.0015 → 保留默认 0）
- AMP+compile 验证通过（test=0.812282 +0.00079）
- EDA 全量画像（907K rows，13 节报告，4 方向决策齐全）
- ❌ warmup+cosine 实验：val ↑ 0.0012 / test ↓ 0.0015（F19，反向）
- ❌ hyformer_blocks=4 实验：val -0.0007 / per-epoch 2x（F20，反向）
- ❌ emb_skip_threshold=6M 实验：val ↑ 0.0006 / **test ↓ 0.0021** / GRAM 24G（F21，反向，发现 reinit × obs/row 关键约束）
- ✅ **W1.0.3 LongerEncoder bug fix**（feature/longer-gather-fix，未 push）：修了 2 个 bug（gather 方向 + mask layout 反转，F22）
- ❌ **W1.7 cap 512 longer 实验**：E2 val 0.861586 (-0.0006) / **test 0.794199 (-0.0181)**（F23，反向，longer 主场劣势）
- 📝 **F24 LongerEncoder bug 2 causal mask 记录**（不修；baseline `--seq_causal` 默认 False 不触发；未来开 causal 前必查）
- 📝 **吸纳用户 5/2 笔记**：W2.7 时间特征建模（信息层第 4 金矿候选）+ W2.8 LR base 扫加入 W2 任务表

**待跑实验**（按 ROI 排序）：
1. **等 E4 激进版结果**（跑中）：top_k=100, seq 512/512/1024/2048；longer encoder 真正主场测试，按 F23 矩阵分级解读
2. **E4 出后立刻 push `feature/longer-gather-fix`** + spec 加 F25
3. **W2.6 pair 特征深化 brainstorm**（5/2 晚原计划）：从 binary 占位升级
4. **W1.10 emb_skip 复活 hash trick**（独立于 W1.7 可并行）：fid 69 hash 171K → fid 47 hash 100K → fid 29 freq truncate top-100K
5. **W2.7 时间特征建模 brainstorm**（v2.0 新增）：label_time / seq ts / time_bucket 用法深化；xhs 暗示 +1%
6. **W2.8 LR base 扫**（v2.0 新增）：dense lr {1e-4, 1.82719e-4 xhs, 3e-4}；3 次 train 半天

## 已知 baseline 问题（v2.0 spec F15-F24）

1. ✅🔥 **F15 / W1.0.1 已结**：reinit threshold=0 是模型不崩的底线（Run Y threshold=10000 best val 卡在 epoch 2，patience 耗尽 EarlyStopping）。**绝对不要碰这条线**。CLI help（`train.py:158`）写的"0=never reset"是错的，但代码行为对，留着别动。
2. ✅ **F16 / W1.0.3 已修**（v0.3.5）：`feature/longer-gather-fix` 修了 LongerEncoder gather 方向（取头=最新）+ mask layout 反转（`pos >= n_valid`）；新增 `--longer_gather_side` CLI（tail 保留旧行为做 A/B）；下游 wiring 复核通过；详见 F22
3. **F17 / W1.0.2 未修**：`trainer.py:87` AdamW 没暴露 dense `weight_decay`（用 PyTorch 默认 0.01）。W2.1 扫参前要加 `--dense_weight_decay` CLI（但 v0.3.3 W2.1 已降级，不急）。
4. ✅ **F18 EDA 完成**：Direction 1 (UNK/OOV) 死、Direction 3 (长序列) 锁定为最大金矿、Direction 2 (emb_skip) 4 fid 全有信号
5. ✅🔴 **F19 warmup+cosine 失败**：val/test 已 divergence，**纯靠 val 涨不能再判定 trick 有效**
6. ✅🔴 **F20 hyformer_blocks=4 失败**：depth scaling 反模式实证，这条路闭环
7. ✅🔴 **F21 emb_skip_threshold=6M 失败**：raise threshold 路径死，但发现 **reinit × obs/row 关键约束**——复活方案必须满足 obs/row ≥ 1000（spec 附录 D.9）
8. ✅ **F22 W1.0.3 修复完成**（v0.3.5）：见上 F16；同时修了 mask layout 反转隐藏 bug
9. ✅🔴 **F23 W1.7 cap 512 longer 失败**：E2 val 0.861586 (-0.0006) / **test 0.794199 (-0.0181)**；val/test gap 拉大 +0.0171；诊断为 longer encoder 设计代价（50-query bottleneck），不是 fix bug；E4 激进版（top_k=100, 长 cap）跑中验证长 cap 是否摊薄
10. **F24 LongerEncoder bug 2 causal mask 记录**（v2.0 新增，不修）：`causal=True` 路径 mask 实现疑似反——`generate_square_subsequent_mask` vs reverse-time 序列（pos 0=最近）语义错位。当前 baseline `--seq_causal=False` 不触发，但 W1.7 子方案 c（transformer + 长 cap）/ 未来 causal longer 试点前必查 `model.py:777-799`

## v2.0 战略提示

- **baseline 已在过拟合悬崖上**——靠 reinit threshold=0 续命（v0.3.2 已证）
- **val 已偏离 test 分布**（F19 + F21 + F23 三证据）：W2 扫参不能纯靠 val，必须周期性消耗 test 提交校准
- **复活高基数特征前先算 obs/row**（v0.3.4 新约束）：reinit threshold=0 强制要求复活后表的 obs/row ≥ 1000；hash trick 100K-200K 桶刚好满足
- **LongerEncoder cap 512 是劣势主场**（v0.3.5 新事实）：第一个 block 后 50-query 永久压缩 512 tokens，下游再不见原序列；cap 必须拉到 ≥ 1024 才有机会反超 transformer
- W2 优先级（v2.0 强化，4 金矿候选）：
  - 🔥 **信息层（最高）**：W1.7 长序列（E4 验证中）→ W1.10 emb_skip hash trick → W2.6 pair 深化 → **W2.7 时间特征**（xhs +1%，spec 之前缺位的最大金矿候选）
  - **超参扫（中）**：W2.8 LR base 扫（xhs 1.82719e-4 提示）
  - **正则化层（最低）**：W2.1-2.5 期望收益 +0.0005~0.002（多次反向实测后进一步打折）
  - ❌ **死路**：depth scaling、OOV→UNK 改造、warmup+cosine LR schedule、raise emb_skip_threshold 直接复活全表、cap 512 + LongerEncoder
- 不要往 baseline 身上再压重正则化（dropout 0.3 / wd 1e-2 大概率推过悬崖）

## 数据关键事实

- 序列**倒序**：pos 0 = 最近（数据分析报告 + EDA pos[0] median diff 验证 ✓）
- train 集**无 OOV**（vocab 按 train max+1 建），test 集**有 OOV**；**v0.3.3 EDA 实测 val 集所有 fid OOV < 1%**（Direction 1 死）
- demo 数据 row group 时间范围全部重叠 → 按 RG 切 ≈ 按行随机切；**线上数据待 W1.9 验证**
- 4 个被 `emb_skip_threshold=1M` 跳过的高基数 seq 特征（fid 29/34/47/69 vocab 1M-86M），forward 返回零向量。**v0.3.4 复活路径**：fid 69 hash 171K（obs/row=1046）/ fid 47 hash 100K（obs/row=3148）/ fid 29 freq truncate top-100K（obs/row=3147）/ fid 34 信号弱保 skip。详见 spec 附录 D.2.5 + D.9。**❌ raise threshold 直接全表已 F21 闭环**
- **序列截断浪费严重（v0.3.3 EDA 实测）**：seq_d cap=512 vs p99=3962 → **90.5% 截断率 / 1.79B tokens 被扔**；seq_a 73% / seq_b 65% / seq_c 34% 截断；当前 baseline 只看到用户 10-30% 历史
- 线上 schema 已收到（v0.2），ts_fid 设置正确（39/67/27/26），list dim 比 demo 大 1.4-2.3×

## 工程规范

- **平台上传**：只上传改过的文件（当前 = `src/trainer.py` + `src/train.py` + `src/run.sh`）。**不上传** `taac2026_schema.json` / `__pycache__/` / `output/`
- **schema 来源**：`train.py:227` 默认从 `data_dir/schema.json` 读，平台用平台自己的 schema；本地 `src/taac2026_schema.json` 仅作参考
- **W1.10 hash trick 实施位置**：在 `dataset.py` 里覆盖 schema vocab + 应用 modulo，**不要改本地 schema 文件**
- **A/B 实验隔离**：用 `CUDA_VISIBLE_DEVICES` + 不同 `TRAIN_CKPT_PATH/LOG_PATH/TF_EVENTS_PATH` env var

## 反模式（不要做）

- ❌ **模型 scaling**（d_model / num_layers / num_blocks）：v0.3.3 F20 实证 hyformer_blocks=4 val ↓ 0.0007 / per-epoch 2x；当前 val/test gap 0.051 是过拟合，加大模型只会更糟
- ❌ **碰 reinit_cardinality_threshold**：v0.3.2 F15 已证它是模型不崩的底线，往任何方向改（10000、100、1000000）都会让训练 epoch 2 后崩
- ❌ **再往 baseline 身上压重正则化**（dropout=0.3、wd=1e-2 等极端值）：已在过拟合悬崖上，更强正则会推过悬崖
- ❌ **warmup+cosine LR schedule**：v0.3.3 F19 实证 val ↑ 0.0012 但 test ↓ 0.0015，gap 拉大 +0.0028
- ❌ **纯靠 val 涨判定 trick 有效**：v0.3.3 F19 实证 val/test 已 divergence，必须周期性消耗 test 提交校准
- ❌ **OOV → UNK 改造**：v0.3.3 F18 EDA 实测所有 fid val OOV < 1%，无 ROI
- ❌ **raise emb_skip_threshold 直接复活全表**：v0.3.4 F21 实证 val ↑ test ↓ / GRAM 24G；reinit × obs/row 约束（spec 附录 D.9）证明全表复活的 obs/row 必然 < 100，模型学不动 → 必须走 hash trick / freq truncate 中转
- ❌ **cap 512 + LongerEncoder**：v0.3.5 F23 实证 test -0.0181；longer encoder 第一个 block 后 50-query 永久压缩到固定长度，cap 512 是 transformer 主场尺寸，longer 设计 ceiling 必然更低 → longer 路径必须搭配长 cap (≥1024)
- ❌ **白天跑长实验**：被抢卡严重，跑实验尽量挪到半夜
- ❌ **并行 A/B 跑同一张物理卡**：用 `CUDA_VISIBLE_DEVICES` 隔离到不同 GPU
- ❌ **改 `taac2026_schema.json` 来做特征工程**：它不上传，改了没用

## 协作约定

- spec 是单一事实源，对话冲突时以 spec 为准
- spec 改动按 v0.X 版本递增，加 changelog 条目
- 关键事实新增进事实表 F# 编号，不要散落在正文
- 实验结果出来后**先更新 spec**再继续做下一个实验
