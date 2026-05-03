# TAAC 2026 PCVR Baseline 改进计划（W1-W2）

**Status**: v2.1 — W1.7 longer encoder 路径**整体闭环失败**：E4 激进版（top_k=100, seq 2048）val **0.861047 (-0.0012)**，比 E2 cap 512 还差 → 拉 cap 不能救 longer，4 重不利机制（信息瓶颈、head gather 自指退化、top_k/cap 比例下降、val/test divergence 即"近因偏置毒药"）见 F25 诊断；W1.7 唯一剩下的子方案 c = transformer + 长 cap（O(L²) 显存待验证）；F26 tokenizer 手工划分（group + query 3 + d_model 96）val 微跌但多变量混淆，待 test 决策
**Author**: brainstorming session 2026-05-01
**Branch**: this doc lives on `main`; implementation work happens on feature branches
**Deadline**: 提交截止 2026-05-23 AOE
**Goal**: 单模型 test AUC 从 0.811492 → ≥ 0.825（接近 / 超过 top1 当前 0.82879）

## Changelog

- **v2.1（2026-05-03 上午）**：W1.7 longer 路径整体闭环 + tokenizer 实验记录：
  - 🔴 **F25 W1.7 E4 longer + cap 2048 失败**：8 epoch val **0.861047 (-0.0012)** / test 未提交（决策已清晰，省配额）/ 19 min/epoch / 12-13G。**比 E2 cap 512 还差**！4 重失败机制深度诊断：(a) 信息瓶颈无法绕开（block 1 cross-attn 一次摘要后下游永远 K 维）；(b) head gather 自指退化（query=最新 K 跟 key 前 K 重合，cross-attn 退化成准 self-attn）；(c) E4 top_k/cap 比例 4.9% < E2 的 9.7%，信息密度反而更低；(d) val/test divergence 加剧——longer 是"近因偏置放大器"，PCVR test 时间漂移让这个偏置变成毒药
  - 🔄 **W1.7 子方案最终裁剪**：(a) cap 512 longer = 死（F23）；(b) 长 cap longer = 死（F25）；(c) **transformer + 长 cap = 唯一剩下，未试**；longer 路径**整体闭环**
  - 🆕 **F26 Tokenizer 手工划分实验（待 test）**：group NS tokenizer + num_queries=3 + d_model=96，5 epoch val **0.861597 (-0.0006)** / test 待提交 / 14-15G / 26 min/epoch。**3 变量同改**（tokenizer type + queries + d_model）val 持平无法判决，**值得花 test 配额校准**——是 spec 之前完全没覆盖的新路径
  - 📝 **W1.0.3 fix 分支 merge 进 main**（独立 commit）：feature/longer-gather-fix 默认 run.sh = baseline 一致（`SEQ_ENCODER_TYPE=transformer` default），可安全 merge；提供 env var 切换 W1.7 实验
  - 📝 **反模式新增**：❌ longer encoder 路径整体（cap 512 / 长 cap 都死）
  - 📝 **接续动作**：longer 路径 close；W1.7 转 transformer + 长 cap 试点（要先验证显存）；W2.7 时间特征 brainstorm 提优先级
- **v2.0（2026-05-03 早）**：阶段性整理 + 吸纳用户笔记新议题。v0.3.x 5 次小迭代积累的内容稳定化；从用户 5/2 笔记里挑出 spec 缺位的 4 个议题加入：
  - 🆕 **F24 bug 2 causal mask（记录不修）**：用户 5/2 review 时发现 LongerEncoder 在 `causal=True` 路径下 mask 实现疑似有误。**当前 baseline `--seq_causal` 默认 False 不触发**，所以暂不修；但记录在事实表，避免 W1.7 子方案 c（transformer + 长 cap，可能要开 causal 稳定训练）或未来要试 causal longer 时踩坑
  - 🆕 **W2.7 时间特征建模（信息层第 4 金矿候选）**：xhs 暗示这条路 +1% 以上。spec 之前缺位，只在 F11 提到 ts_fid 设置正确 + time_bucket 在用，但具体用法可能很浅。需要 brainstorm 设计：(a) label_time vs row timestamp 差值 (b) 各 seq ts vs row timestamp 的 cross-domain 时间对齐 (c) RoPE / 时间分桶 PE 变体
  - 🆕 **W2.8 LR base value 扫**：xhs 暗示 lr=1.82719e-4 有奇效。F19 闭环的是 schedule（warmup+cosine），不是 base LR。我们 baseline `lr_dense=1e-4 / lr_sparse=0.05` 没扫过 base value。1 次 train 可验证
  - 📝 **W2.6 标记"待 5/2 晚深化"**：用户计划 5/2 晚干 pair 特征深挖；当前 W2.6 是简单 binary 占位，等用户深化后重写
  - 📝 **tokenizer 命名澄清（不写进事实表）**：用户笔记说 baseline 用"onetrans"，但 run.sh 已设 `--ns_tokenizer_type rankmixer`。用户确认是命名混淆——代码叫 rankmixer 但实现思路更像 onetrans，group 才更像 rankmixer。结论：**命名问题不影响行为，spec 不需要追这条**
  - 🔄 **Status 头重构**：从"事件流水"改成"阶段总结 + 当前唯一决策点"
  - 📝 **接续动作微调**：W2.7 / W2.8 加入候选清单
- **v0.3.5（2026-05-02 上午）**：W1.0.3 修复完成 + W1.7 第一轮 longer encoder 实验：
  - ✅ **F22 W1.0.3 修复完成**（`feature/longer-gather-fix` 分支，未 push）：实际改了 **2 个 bug**——
    - Bug A（计划修的）：gather 方向，旧 `start_pos = valid_len - actual_k` 取尾=最老，新 `start_pos = 0`（head 模式）取头=最新
    - Bug B（顺手修的隐藏 bug）：mask layout 跟 token 内容反了，仅在 `valid_len < K` 时触发——旧 `pos < pad_count` 说 padding 在前，但 indices 实际取出 valid 在前的 token；新 `pos >= n_valid` 跟内容对齐
    - 新增 `--longer_gather_side {head,tail}` CLI 参数 + run.sh 通过 `GATHER_SIDE` env var 控制；`tail` 保留旧行为做 A/B
    - 下游 wiring 已复核（MultiSeqHyFormerBlock cross_attn 全靠 mask 屏蔽 padding，不依赖 valid 在哪一头）；RoPE 位置对齐正确
  - 🔴 **F23 W1.7 cap 512 longer encoder 实验失败**：E2 (longer + head + cap 512) val 0.861586 (-0.0006) / **test 0.794199 (-0.0181)** / 7 min/epoch（vs baseline 23min）/ 10G GRAM。val/test gap 0.0499 → 0.0670（+0.0171，是 F19/F21 的 ~10x）。**诊断不是 fix bug**，是 LongerEncoder 设计代价：第一个 block 后用 50-query 永久压缩 512 tokens，cap 512 下设计 ceiling 必然低于 transformer
  - 🔄 **W1.7 子方案分裂**：cap 512 longer 路径已闭环（F23 实证劣势）；剩下两条候选：(a) longer + 长 cap（E4 激进版跑中：top_k=100, seq 512/512/1024/2048）；(b) transformer + 长 cap（attention O(L²)，未试）
  - 📝 **E1 vs E2 对比**：E1 (no fix, tail+反转 mask) val 0.861060 / test 缺；E2 (fix, head, mask 对齐) val 0.861586。差 +0.0005 噪声级，无法只凭 val 数据判断 fix 单独贡献，但代码 review + 下游验证支持 fix 正确
  - 📝 **接续动作更新**：W1.0.3 闭合，W1.7 子任务展开，F23 加进死路警告
- **v0.3.4（2026-05-02 上午）**：emb_skip 复活第一次实验 + 关键约束发现：
  - 🔴 **F21 emb_skip_threshold=6M 失败**：val 0.862207 → **0.862852 (+0.0006)** / test 0.812282 → **0.810162 (-0.0021)** / GRAM **24G**（vs baseline 12-13G）/ 28 min/epoch（+22%）。val/test gap 0.0499 → 0.0527（同 F19 模式）
  - 🆕 **关键约束发现：reinit × emb_skip 的隐藏交互**：F15 reinit threshold=0 每 epoch 重置所有已建 emb；raise threshold 把 fid 29 (5.7M) / fid 34 (1M) 拉进 reinit 范围 → 平均 **55 / 314 obs/row** vs 每 epoch 清零 = 模型根本学不动这些超大表。EDA 推荐 hash 桶 100K-171K（**1k-3k obs/row**，跟 baseline 现有 emb 同量级）正是为了规避这个问题
  - 🔄 **W1.10 路径裁剪**：~~raise emb_skip_threshold~~（路径错，F21 实证）；仍可走的两条 → (a) **hash trick**（fid 69 hash 171K / fid 47 hash 100K，附录 D.4）；(b) **freq truncate + UNK pooling**（fid 29，附录 D.6 改造）。fid 34 EDA 已判信号弱建议保 skip
  - 📝 **附录 D.9 新增**：reinit × obs/row 约束量化分析；新增"复活方案 obs/row sanity check"作为前置门槛
  - 📝 **反模式新增**：❌ raise emb_skip_threshold 直接复活全表（不走 hash / truncate 中转）
- **v0.3.3（2026-05-02 早）**：EDA 完成 + 两个反向实验数据：
  - 🆕 **F18 EDA 完成**：`docs/eda/2026-05-01-data-profile.md`（907K val rows，13 节完整画像）。4 方向决策：
    - **Direction 1 (UNK/OOV) DEAD**：0 fid val OOV ≥ 1%，UNK token 改造无 ROI；W2 条件项里"OOV → UNK 改造"分支可砍
    - **Direction 2 (emb_skip 复活) 信号充足**：fid 69 推荐 hash 171K（99% 覆盖率）/ fid 47 hash 100K / fid 29 freq truncate / fid 34 分布平摊（信号弱）
    - **Direction 3 (长序列) 锁定最大金矿**：seq_d **90.5% 截断率** / **1.79B tokens 被扔**（cap=512 vs p99=3962）；seq_a 65% / seq_b 73% 截断；当前 baseline 只看到用户 10-30% 历史
    - **Direction 4 (per-fid emb_dim) 后置**：低 ROI 工程量，先做信息层
  - 🔴 **F19 warmup+cosine 失败**：val 0.862207 → **0.863439 (+0.0012)** 但 test 0.812282 → **0.810773 (-0.0015)**；val/test gap 0.0499 → **0.0527（恶化 +0.0028）**；含义：~~val ↑ 即有效~~ 不再可用作判据，val 已偏离 test 分布；W2.1 扫参必须靠 1 次 test 提交确认，不能纯 val 选优
  - 🔴 **F20 hyformer_blocks=4 失败**：5 epoch val **0.861496 (-0.0007)** / per-epoch 47min（**2x baseline**）/ GRAM 17G；CLAUDE.md "❌ 模型 scaling" 反模式实证；W2 后续不再做 d_model / num_layers 类实验
  - 🔄 **W1.7 升级为最高优先级**：EDA 已量化潜力（seq_d cap 256→2048 能从 9.5% → ~99% token 覆盖）；前置仍是 W1.0.3 LongerEncoder bug fix
  - 🔄 **W2.1 警告增强**：除了"baseline 在过拟合悬崖"，再加"val 已偏离 test"——扫参时必须周期性消耗 test 提交校准，否则扫到的"最佳"可能是过拟合 val
  - 📝 **接续动作重写**：旧 TODO 大部分已完成，按当前实际状态重排
- **v0.3.2（2026-05-02 凌晨）**：W1.0.1 信号重新解读 + W1.1 完整数据：
  - 🔥 **F15 大升级**：Run Y（threshold=10000）不是"略差 0.0015"，是 **best val 卡在 epoch 2（0.857339），epoch 3 起反向下降直到 EarlyStopping 触发**——意味着低基数 emb 不重置 → epoch 2 后训练发散。reinit threshold=0 不是"刻意激进的优化"，是**模型不崩的底线**
  - 🔄 W2 策略大调整：把"正则化层"（W2.1/2.2/2.3/2.4/2.5）和"信息层"（W1.7 长序列 / W1.10 高基数复活）分开看；**baseline 已在重正则化悬崖边走钢丝**，正则化层边际收益打折，**信息层优先级升级**
  - ✅ W1.1 完全收尾：AMP+compile val=0.862207、**test=0.812282（+0.00079）**、23min/epoch、12-13G
  - 📝 显存账修正：AMP only 10-11G / AMP+compile 12-13G / compile only 15-16G（W1.7 长序列时按这 3 档选）
  - 💡 快 A/B 启发法：6 epoch 收敛 ≈ 每 epoch +0.001 量级；**3 epoch 足够诊断 ≥ 0.0015 量级差异**，A/B 可砍 50% 算力
- **v0.3.1（2026-05-01 深夜）**：W1.0.1 实验结果：
  - ✅ W1.0.1 完成：Run X（默认 threshold=0）vs Run Y（threshold=10000）→ **Run Y val AUC 掉 0.0015**
  - 🔄 F15 重新表述：不是"doc/code 矛盾的 bug"，而是"**doc 写错，code 是刻意的激进 cold restart 设计**"——作者本意就是每 epoch wipe 全部已建 embedding
  - 🔄 W2.5 简化：砍掉 `reinit_cardinality_threshold` 0/1000/10000 扫参（已知 0 胜出），只保留 sparse weight_decay 扫
  - 💡 **新认知**：baseline 已经在"重正则化高原"上——你的 0.051 val/test gap 部分已被这个 trick 压住；W2 推更强正则化（dropout=0.3、wd=1e-2）的边际收益要保守估
  - 📝 train.py:158 CLI help 文档错，建议本地修但**不影响平台行为**
- **v0.3（2026-05-01 晚）**：GPT 第二轮 review 反馈 + AMP/compile 实测数据：
  - 🆕 F15：`reinit_cardinality_threshold=0` 文档与代码不一致——CLI help 写 "0=never reset" 但 `model.py:1498` 实际 `vs > 0` 在 threshold=0 时**重置全部已建 embedding** → 新增 **W1.0.1 A/B** 验证 baseline 真实行为
  - 🆕 F16：LongerEncoder top-k 方向反了——`model.py:691` `start_pos = valid_len - actual_k` 取序列**尾部**，但序列倒序（pos 0=最近），即取的是**最老**而非最新 → 新增 **W1.0.3** 修 bug
  - 🆕 F17：`trainer.py:87` AdamW 没暴露 dense weight_decay 参数 → 新增 **W1.0.2** 加参数（W2.1 扫参前置）
  - ✅ W1.1 AMP/compile 实测：compile only **-62%**（20min/epoch），AMP only -35%（34min/epoch），二者 val AUC 漂移 < 0.0002 全部达标；待补 AMP+compile 叠加 + 1 次 test 提交确认
  - W1.2 实施改法：trainer 不动，evaluate 时 dump (user_id, ts, label, pred) 到 csv，后处理脚本算 3 个切法 AUC
  - W1.6 加分母：当前 `_oob_stats` 只有 count 没分母，要算每特征曝光分母才能得 OOV 率
  - W1.7 长序列改走 longer encoder（前提 W1.0.3 修完），优先 AMP-only 配置（省 33% 显存）
  - W1.10 扩成完整方法论 → 详见**附录 D**（spot-check 脚本 + 复活决策树 + 最简实施路径）
  - 附录 A reinit 范围描述更正：不是"全部 embedding"，是"已建的 seq + NS embedding"
- **v0.2（2026-05-01 中午）**：用户提供线上 schema。验证发现：
  - ✅ ts_fid 在线上每个 domain 都正确设置为 39/67/27/26 → **W1.8 取消**（投入减 0.5 天）
  - ⚠️ 真实数据 list dim 比 demo 大 1.4-2.3× → **W1.7 优先级升级**
  - 🆕 emb_skip_threshold 跳过 4 个高基数 seq 特征（fid 29/34/47/69），疑似可上分 → **新增 W1.10**
- v0.1（2026-05-01 凌晨）：初稿

---

## 关键事实回顾（决定整套计划的输入）

| 编号 | 事实 | 来源 | 决定 |
|---|---|---|---|
| F1 | Val AUC 0.862 / Test AUC 0.811，gap = 0.0507 | 用户 baseline 实验记录 | gap 远大于 baseline-vs-top1 的 0.0173；防过拟合/防漂移是最大杠杆 |
| F2 | Top1 AUC 0.82879，跟 baseline 差 0.0173 | 用户 leaderboard 信息 | 不是"换架构"决胜的赛题 |
| F3 | 性能瓶颈在 GPU compute（forward 28.9% + backward 64.5%），data wait 0.1% | 用户 profiling | 任何加速优化要在 op/计算图层；不要碰 dataloader |
| F4 | 算力 2×H20-20%×19G，每次完整训练 30min/epoch×6 = 3h，22 天 = 30-50 次实验预算 | 用户 | 单次实验信息密度要高 |
| F5 | 序列是**倒序**的（pos 0 = 最近，pos N = 最远） | 数据分析报告 | 任何"recent matters more"加权方向应给小 index 高权重 |
| F6 | 训练集 vocab 按 max+1 开，**train 集无 OOV 但 test 集有** | 数据分析报告 + demo 数据验证 | OOV → 0(padding) 在 test 上等同丢特征 |
| F7 | 报告作者明示"全长序列上分"（baseline 截 256/512，p99 序列长 1800-2900） | 数据分析报告 + demo 数据 | AMP/省显存 → 拉长序列是直接的提分路径 |
| F8 | 报告作者明示"val 切法跟线上对不齐" | 数据分析报告 | 当前 val 不可信 |
| F9 | **demo 数据上每个 row group 时间范围完全重叠**——按 row group 切 ≈ 按行随机切 | 实测 demo | **待真实数据验证**：如果真实数据也是这样，val 就是个随机 holdout |
| F10 | **当前 demo schema.json 是用户自用 codex 生成的 mock**；线上版本已于 2026-05-01 中午收到 | 用户 | F11/F12/F13 据线上 schema 重新核定 |
| F11 | ~~demo schema 把每个 domain 的 ts_fid 设为 null~~（mock 错误）→ **线上 schema 把 ts_fid 正确设置为 seq_a:39 / seq_b:67 / seq_c:27 / seq_d:26**，且这 4 个 fid 的 vocab=0（dataset.py:625 触发清零路径，仅作 timestamp 用） | 线上 schema 实测 | time_bucket embedding 在线上**正常工作**；之前担心的 +0.005~+0.015 上分点不存在；W1.8 取消 |
| F12 | **线上 list dim 比 demo 大 1.4-2.3×**（user_int_feats_15 demo dim=13 / 线上 dim=26；user_int_feats_65 demo 49 / 线上 111；user_int_feats_66 demo 66 / 线上 150） | 线上 schema vs demo schema 对比 | 显存压力比 demo 估的更大；W1.7 必须在真实数据上重做估算 |
| F13 | **`emb_skip_threshold=1e6` 在线上跳过 4 个高基数 seq 特征**：seq_b fid=69 vocab=64.7M、seq_c fid=29 vocab=5.7M、seq_c fid=34 vocab=1.0M（边界）、seq_c fid=47 vocab=86.3M | 线上 schema 实测 | 这 4 个特征 forward 时返回零向量。如果是有意义的 ID（不是 timestamp），用 hash trick 重启它们可能是 +0.005 量级上分点 → 新增 W1.10 |
| F14 | 线上 schema 顶层多了 `"format": "raw_parquet"` 字段，代码当前未读取 | 线上 schema 实测 | 标记，暂不处理 |
| F15 | **`reinit_cardinality_threshold=0` 是模型不崩的底线**（v0.3.2 升级表述）：W1.0.1 实测——Run X（threshold=0）val 6 epoch 单调涨到 0.862207（test 0.812282）；Run Y（threshold=10000）**best val 卡在 epoch 2（0.857339），epoch 3 起反向下降，patience 耗尽 EarlyStopping**。意味着如果不重置低基数 emb（gender/age/device），它们 2 epoch 内被 dense 参数绑死开始记忆 train 集组合，泛化反向衰减。**reinit 范围**：所有已建 seq embedding（不含 emb_skip 的 4 个高基数）+ 所有 user/item NS tokenizer embedding；**reinit 不影响**：time_embedding、dense 参数、emb_skip 的表。CLI help（`train.py:158`）写 "0 = never reset" 是错的，但**代码行为绝对不能改** | `trainer.py:355` + `model.py:1470-1519` + W1.0.1 实验 | baseline 已在"过拟合悬崖"上走钢丝；W2 不要碰 reinit 这条线；正则化层（dropout/wd/EMA/SWA/label smoothing）的边际收益要打折估，**真正杠杆在信息层**（W1.7 长序列、W1.10 复活高基数特征） |
| F16 | **LongerEncoder top-k 方向反了**：`model.py:691` `start_pos = valid_len - actual_k` 取序列尾部；F5 已确认序列倒序 pos 0=最近，所以代码实际取的是**最老**而非最新 token；当前 baseline 用 transformer 没触发，但 W1.7 走 longer encoder 必须先修 | `model.py:670-719` 实测 | W1.7 / W2 长序列实验前置任务 |
| F17 | **dense AdamW 没暴露 weight_decay**：`trainer.py:87` `torch.optim.AdamW(dense_params, lr=lr, betas=(0.9, 0.98))` 未传 weight_decay → 用 PyTorch 默认 0.01；CLI 也只有 `--sparse_weight_decay` | `trainer.py:87` 实测 | W2.1 (dropout, wd) 矩阵扫参前置任务 |
| F18 | **EDA 完成（v0.3.3）**：profile 脚本 `src/profile_data.py` 跑出 907K val rows 全数据画像 → `docs/eda/2026-05-01-data-profile.md`。**4 方向决策**：(1) **Direction 1 UNK/OOV 死**：0 个 fid val OOV ≥ 1%，最高 OOV fid 也只 0.4%——UNK 改造无 ROI；(2) **Direction 2 emb_skip 复活**：fid 69 推荐 hash 171K / fid 47 hash 100K / fid 29 freq truncate（top-100K 覆盖 95%）/ fid 34 分布平摊（信号弱可保 skip）；(3) **Direction 3 长序列锁定最大金矿**：seq_d 当前 cap=512 vs p99=3962 → **90.5% 截断率 / 1.79B tokens 被扔**；seq_a/b/c 截断率 65-73%；当前 baseline 只看到用户 10-30% 历史；(4) **Direction 4 per-fid emb_dim 后置**：低 ROI 工程量；schema health check 全部通过（schema vocab vs 实测 max+1 一致） | `docs/eda/2026-05-01-data-profile.md` + `docs/eda/2026-05-01-data-profile.json` | W1.10 / W1.7 决策依据齐全；W2 条件项里 OOV 分支可砍；W1.7 升级为最高优先级 |
| F19 | **warmup+cosine LR schedule 失败（v0.3.3）**：用户 5/2 凌晨实验，10 epoch，**val 0.862207 → 0.863439 (+0.0012)** 但 **test 0.812282 → 0.810773 (-0.0015)**。val/test gap 0.0499 → **0.0527（恶化 +0.0028）**。诊断：cosine 后段低 LR + 多跑 4 epoch 让模型在 val 分布上过精修，test (OOD) 没受益反而被坑。**含义**：(1) val ↑ **不再**可用作单独判据；(2) val 已偏离 test 分布（EDA 已报 train tail vs val head 时间不重叠）；(3) W2.1 扫参必须周期性消耗 test 提交校准，不能纯 val 选优 | 用户实验 / 5 月 2 日 | W2.1 警告增强；不要再单独靠 val 涨判定 trick 有效 |
| F20 | **hyformer_blocks=4 (depth scaling) 失败（v0.3.3）**：用户 5/2 凌晨实验，5 epoch val **0.861496 (-0.0007)** / per-epoch **47min（2x baseline）** / GRAM 17G（接近 19G 上限）。CLAUDE.md "❌ 模型 scaling" 反模式实证 | 用户实验 / 5 月 2 日 | W2 后续不再做 d_model / num_layers 类实验；这条路闭环 |
| F21 | **emb_skip_threshold=6M 失败 + 关键约束发现（v0.3.4）**：用户 5/2 上午实验，threshold 从 1M 拉到 6M 让 fid 29 (vocab=5.7M) + fid 34 (vocab=1.0M) 全量建表（fid 47 / fid 69 仍被 skip）。结果：6 epoch val **0.862852 (+0.0006)** / test **0.810162 (-0.0021)** / GRAM **24G**（vs baseline 12-13G，**+10G**）/ 28 min/epoch（+22%）。val/test gap 0.0499 → 0.0527（同 F19 模式）。**深层失败原因**：F15 reinit threshold=0 每 epoch 清零所有已建 emb；raise threshold 让 fid 29/34 进入 reinit 范围 → 平均 **fid 29: 55 obs/row**（314M obs / 5.7M rows）/ **fid 34: 314 obs/row** + 每 epoch 重置 = 模型实际学不动这些大表，反而引入了未学好的零向量噪声拖累 test。**重要不否定**：EDA 推荐的 hash 100K-171K 桶平均 obs/row 是 1k-3k（跟 baseline 现有 user/item emb 同量级），仍然可走 | 用户实验 / 5 月 2 日 | W1.10 路径裁剪：raise threshold 死、hash trick / freq truncate 仍可试；新增 obs/row sanity check（附录 D.9）作为复活方案前置门槛 |
| F22 | **W1.0.3 LongerEncoder bug fix 完成（v0.3.5，2 bug）**：`feature/longer-gather-fix` 分支（未 push），改 `model.py:670-720` `LongerEncoder._gather_top_k`：(Bug A) gather 方向 `start_pos = valid_len - actual_k` → `start_pos = 0`（head 模式取最新 K）；(Bug B 隐藏 bug，仅 `valid_len < K` 触发) mask layout 跟 token 内容反了——旧 `pos < pad_count` 说 padding 在前，但 indices 实际取出 valid 在前 → 新 `pos >= n_valid` 跟内容对齐。新增 `--longer_gather_side {head,tail}` CLI（tail 保留旧行为做 A/B）+ run.sh 通过 `GATHER_SIDE` env var 控制。**下游已验证**：`MultiSeqHyFormerBlock` cross_attn 全靠 mask 屏蔽 padding 不依赖 valid 在哪一头；RoPE `q_pos_indices` 跟原始序列位置对齐 ✓ | `feature/longer-gather-fix` 代码 review | W1.7 解锁；longer encoder 路径技术上可走 |
| F23 | **W1.7 cap 512 longer encoder 实验失败（v0.3.5）**：E2 (longer + head + cap=256/256/512/512) 6 epoch val **0.861586 (-0.0006)** / **test 0.794199 (-0.0181)** / 7 min/epoch（vs baseline 23min，-70%）/ 10G GRAM。E1 (longer + tail + cap=512，no fix) val 0.861060 / test 缺；E2-E1 val +0.0005（噪声级，无法判 fix 单独贡献）。**val/test gap 0.0499 → 0.0670（+0.0171，是 F19/F21 量级的 ~10x）**。**诊断**：不是 W1.0.3 fix bug（代码 review + 下游验证通过），是 LongerEncoder 设计代价——第一个 block cross-attn 后用 50-query 永久压缩 512 tokens，下游再也访问不到原序列；cap 512 是 transformer 的主场尺寸，longer 设计 ceiling 必然更低 | 用户实验 5/2 / `feature/longer-gather-fix` 代码 review | cap 512 longer 路径闭环失败；W1.7 转向"longer + 长 cap"（E4 激进版跑中：top_k=100, seq 512/512/1024/2048）或"transformer + 长 cap"（O(L²) 显存代价，未试） |
| F24 | **LongerEncoder bug 2 causal mask（记录不修，v2.0）**：用户 5/2 review LongerEncoder 时发现 `causal=True` 路径下 mask 实现疑似反——具体怀疑点见 `model.py:777-799` self-attn 模式的 `attn_mask = nn.Transformer.generate_square_subsequent_mask(L)` 跟 reverse-time 序列（pos 0=最近）的语义可能错位（"causal"按时间应该 mask 掉未来=更新的 token，但 pos 0 在 reverse-time 下就是最新的，不应被 mask）。**当前 baseline `--seq_causal` 默认 False 不触发**，暂不修；但 W1.7 子方案 c（transformer + 长 cap）/ 未来 causal longer 试点前必须先 review 这部分代码并设计 unit test 验证 | 用户代码 review / `model.py:777-799` | 暂不投入；列入"未来开 causal 前必查"清单 |
| F25 | **W1.7 E4 longer + cap 2048 失败 → longer 路径整体闭环（v2.1）**：8 epoch val **0.861047 (-0.0012)** / test 未提交（省配额）/ 19 min/epoch / 12-13G。**比 E2 cap 512（val 0.861586）还差 -0.0005**！4 重失败机制：**(a) 信息瓶颈**：block 1 cross-attn 一次性把全 cap 压缩到 top_k，下游 N-1 个 block 永远在 K 维上 self-attn，单点失败无救（vs transformer 每个 block 都 refresh 全序列）；**(b) head gather 自指退化**：query=最新 K 个 token 跟 key 前 K 重合，cross-attn 倾向 attend 自己，退化成 self-attn + 一点尾巴，后 L-K 个老 token 被低权重看待；**(c) top_k/cap 比例反而下降**：E2 50/512=9.7%，E4 100/2048=**4.9%**——拉 cap 信息密度反而更低；**(d) val/test divergence 即"近因偏置毒药"**：longer 在 train+val（同时间窗）能学"近期 50 token 模式"，test 时间漂移使该模式失效；transformer 因 self-attn 全连接对最新依赖弱，分布偏移容忍度高 | 用户实验 5/2-5/3 + 架构机制深度诊断 | longer 路径整体死；W1.7 子方案 (a) cap 512 longer ❌（F23）+ (b) 长 cap longer ❌（F25）= 全部闭环；唯一剩下 **(c) transformer + 长 cap**（O(L²) 显存待验证 cap 1024 是否爆 19G） |
| F26 | **Tokenizer 手工划分实验（val 持平，待 test 决策，v2.1）**：group NS tokenizer + num_queries=3 + d_model=96（**3 变量同改** vs baseline 的 rankmixer + 2 query + d_model=64），5 epoch val **0.861597 (-0.0006)** / test 待提交 / 14-15G / 26 min/epoch。val 持平无法独立判决——可能"d_model 加大涨了 + group tokenizer 拖累 = 抵消"，也可能真持平。spec 之前完全没覆盖这条路径（v2.0 把 tokenizer 命名问题略过未扫架构），test 数据有独立信息价值 → **值得花 test 配额校准** | 用户实验 5/2-5/3 | 待 test：≥0.812 → 开新路径继续挖；<0.808 → 闭环组合死路；中间 → 单独扫 d_model=96（rankmixer 不变）排除组合效应 |

---

## 总体战略：三周作战图

```
今天 5/1 ───────────────────────────────────────────► 5/23 截止
       │            │            │            │
       │   W1 (7d)  │   W2 (7d)  │   W3 (7d)  │ buffer
       └────────────┴────────────┴────────────┴────────┘
        基建+诊断    防过拟合战    冲分+定稿
```

- **W1（5/1–5/8）"打地基"**：让单次实验更快（AMP）、更可信（多切法 holdout）、看得清问题（分桶诊断 + 实验追踪 csv） + 修明确的 baseline bug。这一周不指望 AUC 大涨。
- **W2（5/9–5/15）"打主战"**：拿 W1 给的诊断结果挑性价比最高的招做扫参 + A/B。目标：单模型 test AUC ≥ 0.821（+0.010）。
- **W3（5/16–5/22）"冲分"**：把 W2 胜出 trick 叠到最终配方，跑 3-5 个不同 seed 的最终训练，挑最好的提交。**留 1 天 buffer 给翻车**。

---

## W1：基建 + 诊断 + 明确 bug 修复

### W1.0：Baseline 行为澄清（v0.3 新增，最高优先级）

GPT 第二轮 review 暴露 3 处 baseline 行为/代码与文档不一致——必须在跑任何调参实验前先搞清楚，否则后续所有 AUC 数字都建立在错的地基上。

| # | Deliverable | 投入 | 验收标准 | 风险 |
|---|---|---|---|---|
| ✅ W1.0.1 | **`reinit_cardinality_threshold` A/B**（v0.3.2 信号修正）：Run X（threshold=0）val 6 epoch 单调涨到 **0.862207**；Run Y（threshold=10000）**best val 卡在 epoch 2 = 0.857339，epoch 3 起反向下降直到 EarlyStopping 触发**。结论：threshold=0 不是"略好的优化"，是**模型不崩的底线**——保留默认。CLI help 文档错（"0=never reset"），代码行为绝对不能改 | 已投入 4h | ✅ 已得结论，且信号比 v0.3.1 解读强得多 | 决定 W2.5 不再扫 threshold；W2 整体策略调整（信息层 > 正则化层） |
| W1.0.2 | **加 `--dense_weight_decay` CLI 参数**：`train.py` argparse 加该参数（default=0.01 维持 PyTorch 默认行为不破坏 baseline 复现）；`trainer.py:87` 把值传给 AdamW；`run.sh` 显式加 `--dense_weight_decay 0.01` 锁定值 | 1h | baseline 复现 AUC 不变（±0.001） | W2.1 扫参直接起步 |
| ✅ W1.0.3 (v0.3.5) | **修 LongerEncoder top-k 方向 + mask layout**：`feature/longer-gather-fix` 已实施。代码改动（详见 F22）：(Bug A) `start_pos = valid_len - actual_k` → `start_pos = 0`（head 模式）；(Bug B) mask 改 `pos >= n_valid` 跟内容对齐；新增 `--longer_gather_side {head,tail}` CLI。**未写 unit test**（直接 6-epoch 实测验证，E1/E2 数据见 F23）。下游 wiring 复核通过 | 已投入 半天 | ✅ E1/E2 数据出来；longer encoder 路径技术上 work，但 cap 512 下设计劣势导致 -0.018 test 跌（F23 诊断为非 fix bug） | F23 暴露的"cap 512 longer 设计劣势"是 W1.7 的关键变量；E4 激进版跑中验证长 cap 是否摊薄 |

#### W1.0 执行优先级

- ✅ **W1.0.1 已完成（v0.3.1）**：threshold=10000 比默认 0 掉 0.0015 → 保留默认；W2.5 简化
- 剩余 W1.0.2 / W1.0.3 是纯代码改动，1 天内并行做完即可

---

### W1 任务清单

| # | Deliverable | 投入 | 验收标准 | 风险 |
|---|---|---|---|---|
| ✅ W1.1 | **AMP/bf16 + compile 接入（v0.3.2 完全收尾）**：实测数据 — Baseline 52min/epoch (val 0.862173 / test 0.811492)、AMP only 34min（-35%）、compile only 20min（-62%）、**AMP+compile 23min/epoch（-56%）val=0.862207 / test=0.812282（+0.00079）/ 显存 12-13G**。test AUC 微涨说明 bf16 + tf32 的小扰动可能意外兼具轻微正则化作用，跟 baseline 重正则化特性吻合。**W1.1 完全 ✅，AMP+compile 锁定为 daily driver** | 已完成 | val/test 漂移 ✅；step 时间 -56% ✅；test AUC 验证 ✅ | 显存 3 档可选：AMP only 10-11G（最省）/ AMP+compile 12-13G（平衡，daily）/ compile only 15-16G（最快） |
| W1.2 | **多切法 holdout（v0.3 改实施方法）**：trainer 不动，evaluate 时 dump 全 valid 集的 (user_id, timestamp, label, pred) 到 csv；写后处理脚本 `tools/eval_multi_split.py` 算 3 个切法 AUC——`tail_rg`（当前 baseline）、`tail_time`（按 timestamp 取尾 10%）、`user_hash`（`xxhash.xxh64(uid).intdigest() % 10 == 0` 隔离 10%）。**用 xxhash 不用 Python `hash()`**——后者跨进程随机不稳定 | 半天（trainer dump 1h + 脚本 3h） | 跑一次训练后能产出 3 个独立切法的 val AUC，数字稳定可重复 | xxhash 是新依赖（pip install xxhash） |
| W1.3 | **多分桶评估脚本** `tools/eval_breakdown.py`：valid 集按 (a) 高/低频 user (b) 高/低频 item (c) seq 长度 0/短/中/长 (d) 标签 0/1 比例 分桶，报每桶 AUC | 1 天 | 输出一张诊断表，找出过拟合最严重的子群 | 桶切太碎导致每桶样本少 AUC 不可信——按训练集 user/item 频次切，而非 valid 频次 |
| W1.4 | **真实数据 reproduce baseline + AMP**：用 AMP 重跑用户的 0.8615 val + 0.811 test 配置，确认实验环境一致 | 3 小时 | val AUC 在 0.8605-0.8625 区间（±0.001），test AUC 在 0.810-0.813 | AMP 后跑出来跟原始有 gap 说明数值精度损失；启动 fp32 fallback |
| W1.5 | **实验追踪 csv**：`runs/experiments.csv` 每行记录 (run_id, branch, config_diff, val_AUC_taill_rg, val_AUC_tail_time, val_AUC_user_hash, test_AUC_if_have, time, notes) | 1 小时 | 后续每次 train 自动追加一行 | 容易忘记填——建议 train.py 退出时强制 dump 一次 |
| W1.6 | **OOV 频次统计（v0.3 加分母）**：dataset 当前 `_oob_stats` 只记 OOV 计数无总曝光分母 → 加每特征曝光 counter；valid 当前 num_workers=0 单进程没汇总问题，但要在文档里标注"valid 改 num_workers > 0 时必须 manual reduce 各 worker stats"；输出 `oov_rate = oob_count / total_count` 按率排序 | 半天 | 输出按 OOV 率排序的特征列表，含分母 | train 集理论 OOV=0（vocab 按 train max+1 建），重点测**线上完整数据集**含 test 隐式 OOV |
| W1.7 | **长序列显存试点（v2.1 longer 整体闭环，剩 transformer + 长 cap 一条路）**：EDA F18 数据——seq_d 当前 cap=512 vs p99=3962 / **90.5% 截断率 / 1.79B tokens 被扔**。**v2.1 状态**：(子方案 a) cap 512 + longer ❌ F23（test -0.0181）；(子方案 b) 长 cap + longer ❌ F25（val 比 cap 512 更差，4 重机制诊断）；(子方案 c) **transformer + 长 cap = 唯一未试**——attention O(L²) 显存待验证，cap 1024 全 4 domain 大概率爆 19G，应只拉 seq_d 单 domain 试探；前置先用 nvidia-smi 观察显存增长曲线 | 1-2 天 | transformer + cap 1024（仅 seq_d）+ AMP-only 配置不爆 19G；val + test 同向涨 ≥ 0.002 才认 W1.7 信息层路径成立；不行就关 W1.7 整条信息层 leg | flash-attention 可能是回退方案；线上 list dim 比 demo 大 2.3×，显存压力远大于 demo |
| ~~W1.8~~ | ~~修复 schema.json 的 ts_fid~~ | ~~半天~~ | ~~**取消**：线上 schema 已正确设置 ts_fid（F11），time_bucket embedding 在线上正常工作~~ | — |
| **W1.10**（v0.3.4 路径裁剪）| **复活 emb_skip 跳过的高基数 seq 特征**：详细方案见**附录 D**。**v0.3.4 警告**：F21 实验证明 raise emb_skip_threshold（直接全量建表）路径死——fid 29/34 平均 55-314 obs/row 配合 reinit threshold=0 模型学不动 + test ↓ 0.0021。仍可走的路径：(a) **hash trick**（附录 D.4）：fid 69 hash 171K（1042 obs/row）/ fid 47 hash 100K（3140 obs/row）；(b) **freq truncate**（附录 D.6 改造）：fid 29 top-100K + UNK pooling（3140 obs/row）。**前置门槛（附录 D.9）**：任何复活方案必须满足 obs/row ≥ 1000 才值得跑 A/B；fid 34 EDA 信号弱 + 平均 314 obs/row（边缘），保 skip | 1 天（fid 69 + fid 47 hash A/B 各 1 次） | 至少 1 个被复活的特征带 ≥ 0.001 val + ≥ 0 test 提升；其他特征做明确保留/丢弃判断 | reinit × obs/row 约束已在 F21 实证；hash trick 仍可能掉点（dataset 改动 bug、hash 冲撞、特征本身就是噪声等）；A/B 必跑，掉点立即回退 |
| **W1.9** | **真实数据上验证 row group 时间分布（待验证）**：在线上数据上跑 `pf.metadata.row_group(i)` 拿每个 RG 的 timestamp min/max，判断是按时间累积（连续）还是混合抽样（重叠） | 1 小时 | 给出明确判断：当前 val 是时间 holdout 还是随机 holdout | 看真实数据 |

### W1 总投入与执行顺序

- **总投入**：~6 天（v0.3 增加了 W1.0 的 1 天）
- **执行顺序**：
  1. **D1（5/1）**：✅ W1.4（baseline reproduce）+ ✅ W1.1 实测（AMP/compile 单独，待补叠加 + test 提交）
  2. **D2（5/2）**：✅ W1.0.1 reinit A/B（保留默认 0）+ W1.0.2 加 dense_wd（1h）+ W1.0.3 修 LongerEncoder + unit test（半天）+ W1.5 csv 追踪
  3. **D3-D4（5/3-5/4）**：W1.10 高基数特征复活（spot-check + 实施 + A/B）+ W1.9 row group 时间分布检查
  4. **D5-D6（5/5-5/6）**：W1.2 多切法 holdout（dump+后处理）+ W1.3 分桶评估 + W1.6 OOV 统计
  5. **D7（5/7）**：W1.7 长序列试点（依赖 W1.0.3 已修）+ buffer + 把 W1 结论写成一页 `diagnosis.md`

### W1 结束的判定标准

到 5/8 你应该手上有：
- 一份 `diagnosis.md`，回答这些问题：
  1. ✅ AMP+compile 后单次训练时间：~20 min/epoch（v0.3 已知）
  2. ✅ reinit_cardinality_threshold 默认值是合理的（v0.3.1 已知，threshold=10000 掉 0.0015）
  3. 在真实数据上 `tail_rg` / `tail_time` / `user_hash` 三个 val AUC 分别是多少？哪个跟 test_AUC 最接近？（W1.2）
  4. 哪些子群（高/低频 user/item、长/短 seq）val_AUC 最差？（W1.3）
  5. 哪些 fid OOV 率 ≥ 5%？（W1.6）
  6. 4 个 emb_skip 高基数 fid 是哪种类型，复活了几个？（W1.10）
  7. row group 时间分布是连续累积还是混合重叠？（W1.9）
- 一份能跑通 + 加速 + 可信 val 的 baseline 训练管线
- 一个能并行 2 个 run 的 GPU 利用环境
- 修过 LongerEncoder top-k 方向 bug（W1.0.3）+ 暴露 dense_weight_decay 参数（W1.0.2）的代码

---

## W2：防过拟合 / 防漂移核心战

W2 是**基于 W1 诊断结果分支执行**的。结构：必做项 + 条件项。

### v0.3.2 W2 战略调整：信息层优先（v2.0 加 W2.7 时间特征 + W2.8 LR base）

W1.0.1 信号修正后明确：baseline 已在"过拟合悬崖"上走钢丝（reinit threshold=0 是底线）。所以 W2 任务**按以下优先级排序**，不再用原来的扁平列表：

| 层级 | 含义 | 任务 | v2.0 后预期 |
|---|---|---|---|
| **信息层（最高优先）** | 给模型更多/更好的信号 | **W1.7 长序列**（最大金矿，EDA 量化：seq_d 90.5% 截断，1.79B tokens 浪费；E4 跑中）、**W1.10 emb_skip hash trick**（EDA 已给方案 + obs/row 通过门槛）、**W2.6 pair 特征**（5/2 晚深化中）、**W2.7 时间特征建模**（xhs 暗示 +1%，spec 之前缺位） | **+0.003~0.015**（W2.7 单项可能就是 +0.01） |
| **正则化层（最低优先）** | 在已有信号上加防过拟合 | W2.1/2.2/2.3/2.4/2.5 | **+0.0005~0.002**（每个 trick 命中，因已在重正则化高原 + warmup+cosine F19 实测反向） |
| **超参扫（中优先）**| 不需要新建模就能上分 | **W2.8 LR base value 扫**（xhs 暗示 1.82719e-4，3 次 train 即可） | **+0.001~0.003**（如果 xhs 数值能迁移） |
| ❌ **死路** | 已闭环不再做 | F20 hyformer_blocks 类 depth scaling、F18 OOV→UNK 改造（Direction 1 死）、F21 raise emb_skip_threshold、F23 cap 512 + LongerEncoder | — |

→ **W2 算力先砸信息层 4 个金矿候选**：W1.7 长序列单项可能 +0.005~0.010；W1.10 4 fid 复活每个 +0.001~0.003；W2.7 时间特征 xhs 暗示 +0.01；W2.6 pair 5/2 晚深化后估。叠加保守估 **+0.010~0.020**，已超 W2 目标。正则化层退化为"如果还有冗余算力顺手做"。

**v2.0 战略闭环**：
- ✅ 信息层有量化数据支撑（EDA F18）+ 第 4 金矿候选（W2.7 时间特征）
- ✅ 反模式有实证（F19 / F20 / F21 / F23）
- 🔥 当前唯一卡点：W1.7 E4 跑结果（决定 longer 路径生死）

### W2 必做项（不论诊断结果）

| # | Deliverable | 投入 | 验收标准 |
|---|---|---|---|
| W2.1 | **正则化超参矩阵（正则化层）**：5 个对角线组合扫——(dropout, wd) ∈ {(0.05, 1e-4), (0.1, 1e-3), (0.2, 1e-3), (0.3, 1e-2), (0.1, 1e-2)}。**v0.3.2 警告**：F15 显示 baseline 在过拟合悬崖上靠 reinit 续命，更强 dropout/wd 大概率推过悬崖；扫到 0.3/1e-2 极可能像 Run Y 一样 epoch 2 后崩。**v0.3.3 警告增强**：F19 显示 val 已偏离 test（warmup+cosine 实测 val ↑ 0.0012 / test ↓ 0.0015）→ **扫参不能纯靠 val 选优，必须每 2-3 个组合消耗 1 次 test 提交校准最佳配置**，否则可能扫到"过拟合 val 不是 test"的伪最佳点 | 5 次 train ≈ 1 天 + 2-3 次 test 提交 | 找到比 baseline 涨 ≥ 0.001 的最佳点（**v0.3.2 期望降至 +0.0005~0.002**） |
| W2.2 | **EMA（正则化层）**：trainer 维护一份 EMA model（衰减 0.999），evaluate 用 EMA model | 半天 | EMA 比 raw 高 ≥ 0.0005（**v0.3.2 期望降，因已在重正则化高原**） |
| W2.3 | **SWA / Checkpoint averaging（正则化层）**：训练结束前最后 N 个 ckpt state_dict 平均 | 半天 | 平均后比单 ckpt 高 ≥ 0.0005（**v0.3.2 期望降**） |
| W2.4 | **Label smoothing 试点（正则化层）**：BCE label 1.0/0.0 → 0.95/0.05；一次 A/B | 1 次 train | 跟 baseline A/B 比，若 ≥ 0.0005 就保留（**v0.3.2 期望降，且可能跟 reinit 重叠**） |
| W2.5 | **Sparse 控制收紧（v0.3.1 简化）**：~~`reinit_cardinality_threshold` 扫参~~（W1.0.1 已证 0 胜出，砍掉）；只保留 Adagrad weight_decay 扫 {0, 1e-4, 1e-3, 1e-2} | 2-3 次 train | 找到 sparse_wd 最佳点；预期收益 ≤ 0.001（baseline 已在重正则化高原） |
| **W2.6**（v0.3.2 新增 / v2.0 待深化）| **target ∈ user seq 交互特征（信息层）**：当前是简单 binary 占位——对每个 (user, target) 样本，计算 4 个二值特征：target item_id 是否在 user 的 seq_a/b/c/d 里出现过；每个 domain 加 1 个 dense feature，concat 进 user 表示。匿名特征场景下最稳的上分招（不依赖语义）。**v2.0 状态**：用户 5/2 晚计划深挖 pair 特征更广建模（不止 binary 是否出现，可能含频次 / 时间衰减 / 位置等），等深化后重写本任务 | 1 天（dataset 改 + 1 次 train）→ 待重估 | val AUC 涨 ≥ 0.002；最理想 +0.005~0.008 | dataset.py 改动量中等；要确保 train/val 各自独立计算（不能跨切污染） |
| **W2.7**（v2.0 新增）| **时间特征建模（信息层第 4 金矿候选）**：spec 之前缺位，仅在 F11 提到 ts_fid 设置正确 + time_bucket 在用，但具体用法可能浅。xhs 暗示这条路 +1% 以上。**Brainstorm 题目**：(a) `label_time` vs `row_timestamp` 差值（用户行为发生到曝光的时间差，可能是 PCVR 的强信号）；(b) 各 seq ts vs row_timestamp 的 cross-domain 时间对齐（每个 domain 的"最近一次行为发生在多久前"）；(c) 现有 time_bucket（65 桶）是否被各 seq 充分使用 vs 只是 padding；(d) 可学习时间编码 vs RoPE vs 时间分桶 PE。需要先 brainstorm 出 ≥ 1 个具体方案再实施 | brainstorm 半天 + 实施 1-2 天 | val + test 同向涨 ≥ 0.003；理想 +0.008~0.015（xhs 暗示 +1%） | spec 缺位说明前面没投入分析；可能跟 W2.6 pair 特征有重叠（target 跟 seq 的时间距离也是 pair 信号） |
| **W2.8**（v2.0 新增）| **LR base value 扫**：F19 闭环的是 schedule（warmup+cosine 反向），不是 base LR。baseline `lr_dense=1e-4 / lr_sparse=0.05` 没扫过 base value。xhs 暗示 lr=1.82719e-4 有奇效（非常具体的数值，可能反映他们扫过）。**A/B 设计**：dense lr ∈ {1e-4 (baseline), 1.82719e-4 (xhs), 3e-4 (上探)} × sparse lr 不变；只扫 dense 因为 sparse 用 Adagrad 对 lr 不敏感 | 3 次 train ≈ 半天 | val + test 同向涨 ≥ 0.001；F19 教训：必须看 test 不能纯靠 val | 比正则化扫参单点 ROI 高（已知 LR 有具体提示数）；但风险是 1.82719e-4 是别人架构的最优，搬到我们 baseline 不一定迁移 |

### W2 条件项（看 W1 诊断哪个分支命中）

| 触发条件 | 动作 | 投入 |
|---|---|---|
| `val_tail_time_AUC` 比 `val_user_hash_AUC` 低 ≥ 0.005 | 加 sample-weight 按时间衰减；近期数据 oversample | 1 天 |
| `val_user_hash_AUC` 比 `val_tail_time_AUC` 低 ≥ 0.005 | OOV → UNK token 改造（每个 vocab 加 1 个 UNK row，dataset 把 OOV 映射到 UNK 而非 0）；high-cardinality embedding hash trick / vocab cap | 1.5 天 |
| **任何切法上** OOV 触发率 ≥ 5% | 同上 OOV → UNK 改造 | 半天（合并） |
| W1.7 显存试点显示能开更长序列 + 不严重过拟合 | 把 seq_a/b 开到 384 或 512，重跑 baseline | 3 小时 |
| W1.3 分桶诊断显示长尾 user/item 子群 AUC 显著低 | 长尾子群 oversample 或 loss reweight | 1 天 |

### W2 执行节奏（v0.3.2 调整：信息层先行）

- **D1（5/9）**：W2.6 "target ∈ user seq" 交互特征（dataset 改 + 1 次 train） + W1.7 长序列试点（如果 W1.0.3 已修）
- **D2-D3（5/10-5/11）**：5 个正则化组合排队跑（W2.1） + 手写 EMA / SWA 代码
- **D4（5/12）**：拿正则化最佳组合叠加 EMA / SWA 跑一次
- **D5-D6（5/13-5/14）**：根据 W1 诊断分支，做条件项里命中的 1-2 个
- **D7（5/15）**：把 W2 所有胜出项叠到一个 "集大成" 配置，跑一次作为 W3 起点

### W2 结束判定

- 一个比 baseline test AUC 涨 ≥ 0.010 的单模型（保守）；理想 +0.015
- 至少 3 个独立有效的 trick（每个 ≥ 0.002）

### W2 算力预算

- 必做 9-10 次 train + 条件 3-5 次 ≈ 12-14 次完整训练
- AMP 后每次 1.5h，2 卡并行，**日历时间 4-5 天**
- W1 的 AMP 加速没拿到 2× 倍率会压缩 W2，触发 fallback：砍掉 W2 条件项里"长尾 oversample"（最贵）

---

## W3：冲分 + 定稿（计划在 W2 结束后再细化）

占位结构：
- D1-D2：W2 集大成配方 ×3 seed 同时跑
- D3-D4：选最佳 seed，再做 1-2 个 final tuning（lr / epochs / EMA decay）
- D5-D6：3-5 个最终候选，挑分最高的提交
- D7：buffer

注：W3 等 W1/W2 实测结果后再写细节，避免空想。

---

## 跨阶段风险管理

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| AMP 与 torch.compile 冲突 | 中 | 高（W1 加速没拿到 → W2 算力崩溃） | W1.1 优先做"AMP 不带 compile"，确认 AMP 单独 OK 再叠 compile |
| 真实数据 row group 是混合时间（demo 那样） | 中 | 高（当前 val 完全不可信） | W1.9 优先做；如果命中，W1.2 多切法 holdout 是必须 |
| ts_fid 在线上 schema 已正确（用户的 schema 是 mock） | 中 | 低（修不修都没事） | W1.8 等线上 schema |
| W1 诊断显示既不是时间漂移也不是 cold-start，是普通过拟合 | 低 | 中（W2 条件项都不命中） | W2 必做项本身就够，预期收益降到 +0.005-0.010 |
| 最后一周 GPU 排队等 | 中 | 中 | W3 提前 1 天开始 |

---

## 接续动作（v2.0 增量更新——5/3 早状态）

### ✅ 已完成（v0.1 → v0.3.2）

1. 线上 schema 接入（v0.2）
2. AMP+compile 实验（v0.3，daily driver 23 min/epoch）
3. W1.0.1 reinit threshold A/B（v0.3.1 → v0.3.2，threshold=0 是底线）
4. AMP+compile 叠加 + test 提交（v0.3.2，test=0.812282 +0.00079）

### ✅ v0.3.3 新增已完成

5. **EDA 完成**（5/1 晚-5/2）：`src/profile_data.py` 跑出 907K val rows 全画像 → `docs/eda/2026-05-01-data-profile.md`；4 方向决策依据齐全（F18）
6. **warmup+cosine 实验**（5/1 夜）：失败，val ↑ test ↓，gap 拉大 → 闭环（F19）
7. **hyformer_blocks=4 实验**（5/1 夜）：失败，depth scaling 反模式实证 → 闭环（F20）

### ✅ v0.3.4 新增已完成

8. **emb_skip_threshold=6M 实验**（5/2 上午）：失败，val ↑ 0.0006 / test ↓ 0.0021 / GRAM 24G → raise threshold 路径死，但发现 reinit × obs/row 关键约束（F21 + 附录 D.9）

### ✅ v0.3.5 新增已完成

9. **W1.0.3 LongerEncoder bug fix**（`feature/longer-gather-fix`，未 push）：修了 2 个 bug（gather 方向 + mask layout 反转），加 `--longer_gather_side` CLI；下游 wiring 复核通过（F22）
10. **W1.7 第一轮 longer encoder cap 512 实验**（5/2 上午）：E1 (no fix, tail) val 0.861060 / test 缺；E2 (fix, head) val 0.861586 / **test 0.794199 (-0.0181)**；诊断为 longer encoder 设计在 cap 512 主场劣势——50-query bottleneck 永久压缩 512 tokens（F23）

### ✅ v2.0 新增已完成

11. **F24 LongerEncoder bug 2 causal mask 记录**（不修）：用户 5/2 review 发现 `causal=True` 路径下 mask 实现疑似反——pos 0=最近 vs `generate_square_subsequent_mask` 默认按时间正序的语义错位。当前 baseline `--seq_causal` 默认 False 不触发，列入"未来开 causal 前必查"清单
12. **吸纳用户 5/2 笔记差距分析**：W2.7 时间特征建模（信息层第 4 金矿候选） + W2.8 LR base value 扫（xhs 暗示 1.82719e-4）加入 W2 任务表

### ✅ v2.1 新增已完成

13. **W1.7 E4 longer + cap 2048 失败**（5/2-5/3 夜）：8 epoch val 0.861047 (-0.0012) **比 E2 cap 512 还差**；4 重机制深度诊断（信息瓶颈 / head gather 自指 / top_k/cap 比例下降 / 近因偏置毒药）→ longer 路径整体闭环（F25）
14. **F26 Tokenizer 手工划分实验**（5/2-5/3 夜）：group + query 3 + d_model 96，val 0.861597 (-0.0006)；3 变量同改 val 持平无法判决，待 test 校准（值得花配额）
15. **feature/longer-gather-fix merge 进 main**（独立 commit）：分支 run.sh 默认 = baseline transformer，可安全 merge；提供 env var 切换 W1.7 实验

### 🔥 立刻做（5/3，按 ROI 排序）

1. **Tokenizer F26 上 test 校准**（0 算力，1 次 test 提交）：只是提交一个 ckpt，立刻知道 group + query 3 + d_model 96 这个组合到底是真持平还是假持平
2. **W1.7 子方案 c 试点：transformer + cap 1024（仅 seq_d）**（首要算力候选）：先 nvidia-smi 看显存，cap 1024 全 4 domain 大概率爆 19G；建议只拉 seq_d 一家；如果不爆，6 epoch 跑一次看 val + test
3. **W2.6 pair 特征深化 brainstorm + 实施**（feature/pair-weighted-pool 已开新分支）：从 binary 占位升级到含频次 / 时间衰减 / 位置等
4. **W1.10 emb_skip 复活 hash trick 路径 A/B**（独立可并行）：
   - 优先 **fid 69 hash 171K**（obs/row=1046）
   - 然后 **fid 47 hash 100K**（obs/row=3148）
   - 可选 **fid 29 freq truncate top-100K + UNK**
5. **W2.7 时间特征建模 brainstorm**（v2.0 新增）：(a) `label_time` vs `row_timestamp` 差值；(b) seq ts vs row_timestamp 各 domain 时间对齐；(c) 现有 time_bucket 用法是否浅；先 brainstorm 设计再实施

### 🔧 中优先（5/4-5/5）

6. **W2.8 LR base value 扫**（v2.0 新增）：dense lr ∈ {1e-4 baseline, 1.82719e-4 xhs, 3e-4 上探}；3 次 train 半天搞定
7. **W1.0.2 加 `--dense_weight_decay` CLI 参数**（1h）—— W2.1 扫参前置，但 W2.1 优先级在 v0.3.3 已经降低，不急
8. **W1.9 row group 时间分布检查**（1h）—— 验证 train tail vs val head 的时间 gap，跟 F19 / F21 / F23 val/test divergence 对照

### 📝 低优先 / 工程债

9. 修 `train.py:158` CLI help 文档错误（"0=never reset" → "0=most aggressive"）
10. profile_data.py 进度日志 bug（modulo 不对齐，第二次没打 log）
11. **未来开 causal 前必查 F24**（bug 2 causal mask 实现）

### 🚫 不再做（v0.3.3 + v0.3.4 + v0.3.5 + v2.1 闭环）

- ❌ Direction 1 OOV→UNK 改造（F18 实测无 ROI）
- ❌ 任何 d_model / num_layers / num_blocks 类 depth scaling（F20 实证；注意 F26 d_model=96 待 test 决定，不直接套这条）
- ❌ 单纯靠 val 涨判定 trick 有效（F19 / F21 / F23 / F25 四实证 val/test divergence）
- ❌ warmup+cosine LR schedule（F19；注意 W2.8 LR base 扫**只扫 base value 不带 schedule**）
- ❌ **raise emb_skip_threshold 直接复活全表**（F21 实证；obs/row 必然 < 1000 不通过门槛）
- ❌ **LongerEncoder 整条路径**（F23 cap 512 死 + F25 长 cap 也死，4 重机制诊断完整）

### 决策点（W1.7 子方案 c 跑完后）

W1.7 子方案 c (transformer + 长 cap) 是 W1.7 信息层最后一根稻草：
- 涨 ≥ 0.005：W1.7 信息层 leg 活了
- 持平：长序列收益跟 transformer O(L²) 代价抵消
- 跌：W1.7 整条 leg 死，信息层金矿候选从 4 个减到 3 个（W1.10 / W2.6 / W2.7）


---

## 附录 A：当前已知的 baseline 行为细节（写给未来的自己）

- `dataset.py:709-721` train/val 切分按 row_group 顺序切，前 N% train 后 M% val
- `dataset.py:633` 当 ts_fid=null 时 time_bucket 全 0
- `model.py emb_skip_threshold=1e6`：vocab > 1M 的特征不建 embedding，forward 时返回零向量
- `train.py reinit_sparse_after_epoch=1` + `reinit_cardinality_threshold=0`（默认）：每个 epoch 末尾**重置所有已建 embedding**（详见 F15）。**reinit 范围**：所有 seq embedding（不含 emb_skip 跳过的 4 个高基数特征）+ 所有 user/item NS tokenizer embedding；**reinit 不影响**：`time_embedding`、被 `emb_skip_threshold` 跳过的表、所有 dense 参数（transformer / RankMixer / projection / LayerNorm 等）。文档与代码不一致 → 见 F15 / W1.0.1
- **reinit 时序**（trainer.py:303-374）：每 epoch 内 `train → evaluate → save_best_ckpt → reinit`，所以**保存的 checkpoint 永远是 reinit 之前的训练态**，推理用的是已学好的 embedding 不是随机参数
- `train.py loss_type=bce`：默认 BCE，可选 focal_loss（默认未启用）
- 双优化器：sparse Adagrad（lr=0.05）+ dense AdamW（lr=1e-4）
- `clip_vocab=True`：OOV 强制 clip 成 0（= padding token）
- 序列倒序（数据分析报告确认，pos 0 = 最近）
- demo 数据 1000 行 / 5 RG / 12.4% 正样本率 / RG 时间范围全部重叠

## 附录 B：与原始数据格式的对应关系

| 字段类别 | 列数 | 数据类型 | 备注 |
|---|---|---|---|
| ID & Label | 5 | int64 / int32 | user_id, item_id, label_type, label_time, timestamp |
| User Int Features | 46 | 混合（int64 / double / list\<int64\>） | 名字叫 int 不一定是 int |
| User Dense Features | 10 | list\<float\> | fid 62-66/89-91 同时出现在 user_int |
| Item Int Features | 14 | int64 / list\<int64\> | fid 11 是多值 |
| Domain Sequence Features | 45 | list\<int64\>（按域分组） | 4 域 a/b/c/d，每域多列对齐 |

---

## 附录 C：线上 schema 关键结论（2026-05-01 中午对比）

### C.1 ts_fid 设置（**线上正确**，无需修复）

| Domain | ts_fid | 备注 |
|---|---|---|
| seq_a | 39 | features 列表里 vocab=0，dataset.py 触发清零，仅作 time_bucket 输入 |
| seq_b | 67 | 同上 |
| seq_c | 27 | 同上 |
| seq_d | 26 | 同上 |

→ time_bucket embedding 在线上**正常工作**，模型能感知"序列每步距今多久"。我们 v0.1 担心的"被埋特征"在线上不存在。

### C.2 被 emb_skip 跳过的高基数 seq 特征（**待 W1.10 排查**）

`emb_skip_threshold=1e6` 触发的特征（forward 返回零向量）：

| Domain | fid | vocab | 是 ts_fid 吗 | 怀疑 |
|---|---|---|---|---|
| seq_b | 69 | 64,710,562 | 否 | hash 后的 long-tail item id？ |
| seq_c | 29 | 5,764,358 | 否 | hash 后的 ID？ |
| seq_c | 34 | 1,031,305 | 否 | 边界，刚超 threshold |
| seq_c | 47 | 86,335,515 | 否 | **最大**，比 ts_fid 还大 50× |

W1.10 要做的事：spot-check 真实数据上每个 fid 的取值分布。判断标准：
- 值域离散且重复多 → 是 ID 类，用 hash trick（hash 到 100K-1M 之间一个合理 vocab）+ 重新建 embedding
- 值域连续且不重复 → 是数值类（金额/时长/坐标），用 bucket 分桶或 dense 投影
- 全部都是同一个值或几个值 → 实际上没信息，保持 skip

未被 skip 但 vocab 大的 seq 特征（≥ 100K，作为参考但不需要排查）：seq_b fid=74 vocab=476K、fid=76 vocab=132K、fid=88 vocab=200K、seq_c fid=36 vocab=977K（接近 threshold）。

### C.3 真实数据 list dim vs demo dim 对照

显存估算时**必须用线上 dim**：

| 字段 | demo dim | 线上 dim | 备注 |
|---|---|---|---|
| user_int_feats_15 | 13 | **26** | 2× |
| user_int_feats_63 | 11 | 19 | — |
| user_int_feats_64 | 18 | 26 | — |
| user_int_feats_65 | 49 | **111** | 2.3× |
| user_int_feats_66 | 66 | **150** | 2.3× |
| user_int_feats_60 | 2 | 2 | 一致 |
| user_int_feats_62 | 5 | 6 | 一致量级 |
| user_dense_feats_61 | 256 | 256 | 一致 |
| user_dense_feats_87 | 320 | 320 | 一致 |

变长 list 字段的 padding tensor 大小 ≈ `B × dim × emb_dim × 4 bytes`。线上 user_int_feats_66 单字段需 256 × 150 × 64 × 4 ≈ 9.8MB/batch。

### C.4 vocab 数量级速查（线上）

| 类别 | vocab 范围 | 数量 |
|---|---|---|
| User int | 3 - 2,848 | 大部分小词表 |
| Item int | 3 - 23,700 | item fid=11 vocab=21,528（多值），item fid=16 vocab=23,700 是最大单值 |
| Seq features | 5 - 86M | 4 个被 skip（见 C.2），其余在 5K - 477K 量级 |

### C.5 顶层差异

线上 schema 多出 `"format": "raw_parquet"` 字段，代码当前未读取——估计是 starter 框架预留，暂不处理。

---

## 附录 D：emb_skip 特征复活决策树（W1.10 详细方案）

### D.1 emb_skip 与 reinit 在 baseline 下独立，复活时强相互作用（v0.3.4 升级）

容易看混的两个 threshold：

| 参数 | 控制什么 | 时机 | run.sh 当前值 | 文档状态 |
|---|---|---|---|---|
| `--emb_skip_threshold` | 是否**建** embedding 表 | 模型构建时（一次性） | **1,000,000**（显式设） | 准确 |
| `--reinit_cardinality_threshold` | 是否**重置**已建 embedding | 每 epoch 末尾 | **0**（默认值） | **错**（见 F15） |

**baseline 状态**（threshold=1M）：emb_skip 跳过的 4 个表**根本没建出来**——forward 返回零向量（`model.py` 中 `if real_idx == -1`），等价于"特征被删除"。reinit 也碰不到它们。两个机制独立。

**复活后状态**（如 F21 raise threshold to 6M）：被复活的 fid 进入 reinit 范围 → **每 epoch 清零**。如果该 fid 的 obs/row 不够大，模型 1 epoch 内拿不到足够梯度信号学好这些 row → 复活引入的不是有效信号，是未学好的零向量噪声。**这就是 F21 test ↓ 0.0021 的根本原因**。详见 D.9。

### D.2 4 个被 skip 特征 spot-check 脚本

**v0.3.3 状态**：spot-check 脚本已被更全面的 `src/profile_data.py` 替代——profile 跑了全 train+val 数据，给出 4 个 fid 的精确 top-K 覆盖率曲线（不是 1 个 row group 的样本）。spot-check 脚本作为遗留参考，实际决策直接看 D.2.5 EDA 实测数据。

```python
# tools/spotcheck_emb_skip.py
import pyarrow.parquet as pq
import numpy as np

TARGETS = [
    ("domain_b_seq_69", 69, 64_710_562),
    ("domain_c_seq_29", 29, 5_764_358),
    ("domain_c_seq_34", 34, 1_031_305),
    ("domain_c_seq_47", 47, 86_335_515),
]

pf = pq.ParquetFile("path/to/real_data.parquet")
df = pf.read_row_group(0).to_pandas()  # 1 个 RG，几万 ~ 几十万行

for col, fid, vocab in TARGETS:
    flat = np.concatenate(
        [np.asarray(x) for x in df[col].dropna() if len(x) > 0]
    )
    flat = flat[flat > 0]  # 过滤 padding=0
    n_unique = len(np.unique(flat))
    print(f"--- fid={fid} (declared vocab {vocab:,}) ---")
    print(f"  total values:     {len(flat):,}")
    print(f"  unique values:    {n_unique:,}")
    print(f"  unique ratio:     {n_unique / max(len(flat), 1):.4f}")
    print(f"  min / max:        {flat.min():,} / {flat.max():,}")
    print(f"  median:           {np.median(flat):,.0f}")
    print(f"  range / vocab:    {(flat.max() - flat.min()) / vocab:.4f}")
    # top-100 频次占比（看 Zipfian 程度）
    vals, counts = np.unique(flat, return_counts=True)
    top100_share = sorted(counts, reverse=True)[:100]
    print(f"  top-100 freq pct: {sum(top100_share) / len(flat):.2%}")
    print()
```

### D.2.5 v0.3.3 EDA 实测数据（替代 spot-check）

来源：`docs/eda/2026-05-01-data-profile.md` Section 6.2 + Section 10。

**4 个 fid 的 top-K 覆盖率曲线**（全 train+val 实测，905K rows / fid，每个 fid ~3 亿次观测）：

| fid | domain | vocab | total_obs | top1k | top10k | top100k | top1M | top10M |
|---|---|---|---|---|---|---|---|---|
| 69 | seq_b | 64,710,562 | 178,843,744 | 14.08% | 22.87% | 33.27% | 45.81% | 45.87% |
| 29 | seq_c | 5,764,358 | 314,691,684 | 5.02% | 10.99% | 27.71% | 58.92% | 58.92% |
| 34 | seq_c | 1,031,305 | 314,799,897 | **82.74%** | **89.25%** | **95.02%** | **98.74%** | 98.77% |
| 47 | seq_c | 86,335,515 | 314,781,147 | 1.00% | 3.69% | 10.59% | 21.19% | 21.19% |

**EDA Section 10 推荐方案**：

| fid | k@99% | 推荐方案 | 桶大小 / 阈值 | 信号判断 |
|---|---|---|---|---|
| 69 | 171,052 | **A：hash trick** | bucket=171,052 | 0.26% vocab → 99% obs，强 Zipfian |
| 47 | 100,000 | **A：hash trick** | bucket=100,000 | 0.12% vocab → 99% obs，最强 Zipfian |
| 29 | 100,000 | **freq truncate + UNK pooling** | top-100K + 1 个 UNK | 1.7% vocab → 99% obs |
| 34 | 139,000 | **raise emb_skip threshold** | 提到 1,031,306（即覆盖全 vocab）or 保 skip | 13.5% vocab 才到 99% — 分布平摊，Zipfian 弱，hash 收益小；信号弱可保 skip |

**优先级**：
1. **fid 69**（seq_b，最高 ROI）：vocab 64.7M，99% obs 只占 0.26% vocab，hash 到 171K 桶几乎无冲撞
2. **fid 47**（seq_c，第二）：vocab 86.3M，最强 Zipfian，hash 100K 桶
3. **fid 29**（seq_c）：freq truncate 比 hash 更直观（100K 个独立桶 + 1 UNK），实施稍重
4. **fid 34**（seq_c，可选）：信号弱，先观察前 3 个的 A/B 结果再决定

**显存账（v0.3.3 更新）**：
- fid 69 复活：171K × 64 × 4B = 44 MB
- fid 47 复活：100K × 64 × 4B = 26 MB
- fid 29 复活：100K × 64 × 4B = 26 MB（+1 UNK row 忽略）
- 全部 3 个复活：~96 MB（vs baseline 0 MB / vs 不 skip 40+ GB）

### D.3 复活方案决策树

读 spot-check 输出，按下表选方案：

| spot-check 看到的 | 判断 | 方案 | 实施成本 |
|---|---|---|---|
| `unique_ratio > 0.5`，值域稀疏离散，top-100 占比 < 10% | 离散 ID 类（hash 后的 user/item/tag） | **A：hash trick**（D.4） | 1h |
| `unique_ratio < 0.1`，min/max 跨度合理，值连续聚集 | 连续数值伪装成 int（金额 / 时长 / 距离 / score） | **B：bucket 分桶**（D.6） | 半天 |
| `unique_ratio < 0.1`，值连续且需保精度 | 真连续数值 | **C：dense 投影**（D.7） | 半天 |
| 单一值 / max-min < 10 / 纯随机分布 | 无信息 | **D：保持 skip** | 0 |

`top-100 占比` 高（>50%）说明分布极度 Zipfian，hash trick 收益高（高频 ID 不冲撞，冷尾 ID 冲撞也无所谓）。

### D.4 方案 A：hash trick 最简实施路径

**0 改 model.py，只改 schema + dataset：**

1. **改 `taac2026_schema.json`**：把这 4 个 fid 的 vocab 从 86M / 64M / 5.7M / 1M 改成 262144（256K）
2. **改 `dataset.py`**：读这 4 个 fid 时 `value = value % 262144`，加白名单
   ```python
   HASH_REVIVE_FIDS = {69: 262144, 29: 262144, 34: 262144, 47: 262144}
   if fid in HASH_REVIVE_FIDS:
       arr = np.asarray(arr) % HASH_REVIVE_FIDS[fid]
   ```
3. **`run.sh` 不改**：`emb_skip_threshold=1000000` 仍设，但现在 4 个 fid 的 vocab=256K < 1M，自动会被建 embedding 表
4. **A/B 验证**：跑 baseline + 复活两个 6-epoch（用 AMP+compile，2h × 2），对比 val AUC

**显存账**：256K × 64 × 4B = 64MB / 表 × 4 表 = **256MB**（vs 不复活 0MB / vs 不 skip 40+ GB 装不下）

### D.5 方案 A 的 hash 实现选择（如果 modulo 不够）

| hash 方式 | 代码 | 优劣 |
|---|---|---|
| modulo（默认起步） | `id % B` | 最快，0 依赖；如果 ID 是连续整数容易扎堆 |
| xxhash | `xxhash.xxh64(str(id)).intdigest() % B` | 分布最均匀；需 `pip install xxhash` |
| 频次哈希 | top-N 高频 ID 占独立桶，剩余 hash 到剩余桶 | 质量最高；要预统计 train 集频次 |

**先用 modulo**，简单 + 0 依赖；如果 modulo 复活的 A/B 显示掉点（且 spot-check 表明 ID 是连续型），再升级 xxhash。

### D.6 方案 B：bucket 分桶（如果 spot-check 显示是连续数值）

```python
# 训练前一次性预处理，从 train 数据收集所有该 fid 的值
all_values = ...  # 跑一次全量 train 收集
log_values = np.log1p(all_values.astype(np.float64))
boundaries = np.quantile(log_values, np.linspace(0, 1, 100))
np.save(f"boundaries_fid_{fid}.npy", boundaries)

# Dataset 读取时
boundaries = np.load(f"boundaries_fid_{fid}.npy")
bucket_id = np.searchsorted(boundaries, np.log1p(value.astype(np.float64)))
bucket_id = np.clip(bucket_id, 0, len(boundaries) - 1)
```

schema 里这个 fid 的 vocab 改成 100；其余流程同方案 A。

### D.7 方案 C：dense 投影（极少数情况，spot-check 显示精度敏感）

跳出 embedding 体系，直接当 float 用。需在 model.py 加一条新 path：把这个 fid 的值 cast 成 float、log1p、过 `nn.Linear(1, emb_dim)`，再 concat 回 seq token embedding。改动量大于 A/B，**只有 spot-check 强烈暗示是高精度连续数值时才走这条**。

### D.8 验证标准与回退

任一方案 + A/B 跑完后：

| A/B 结果 | 动作 |
|---|---|
| val AUC 涨 ≥ 0.001 **且 test 不掉** | 保留复活 |
| val AUC 持平（±0.001）或 val ↑ test ↓（F19 / F21 模式）| 回退（保留 D：继续 skip）；**v0.3.4 提醒**：纯 val 涨不能判定有效，必须 test 提交校准 |
| val AUC 掉 ≥ 0.001 | 回退；如果是 modulo，先升级 xxhash 再试一次；仍掉就 fallback 到 D |

每个 fid **独立 A/B**——可能 fid=69 复活有效但 fid=47 复活掉点，最终保留前者跳过后者。

### D.9 obs/row sanity check（v0.3.4 新增，复活方案前置门槛）

**核心约束**（F21 实证 + F15 推导）：reinit threshold=0 每 epoch 清零所有已建 emb，所以复活后的表必须满足"每行平均收到足够梯度信号"才能在 1 epoch 内学到东西。**经验门槛：obs/row ≥ 1000**（跟 baseline 现有 user/item emb 同量级）。

**计算方式**（来自 EDA 数据）：

```
obs/row = total_observations(fid) / 复活后表的 row 数
```

**4 个 fid 在不同复活方案下的 obs/row**：

| fid | total_obs | 方案 | 表大小 | obs/row | 评估 |
|---|---|---|---|---|---|
| 69 | 178,843,744 | hash 171K | 171,052 | **1,046** | ✅ 通过门槛 |
| 69 | 178,843,744 | raise threshold（全表） | 64,710,562 | **2.8** | ❌ 远远不够 |
| 47 | 314,781,147 | hash 100K | 100,000 | **3,148** | ✅ 通过门槛 |
| 47 | 314,781,147 | raise threshold（全表） | 86,335,515 | **3.6** | ❌ 远远不够 |
| 29 | 314,691,684 | freq truncate top-100K + UNK | 100,001 | **3,147** | ✅ 通过门槛 |
| 29 | 314,691,684 | raise threshold（全表） | 5,764,358 | **55** | ❌ F21 实证失败 |
| 34 | 314,799,897 | freq truncate top-100K + UNK | 100,001 | **3,148** | ✅ 通过门槛 |
| 34 | 314,799,897 | raise threshold（全表） | 1,031,305 | **305** | ⚠️ 边缘（F21 同时复活，无法单独归因；EDA 已判信号弱） |

**含义**：
1. **直接 raise emb_skip_threshold 的路径全死**：4 个 fid 的全表 obs/row 都远低于 1000 门槛
2. **EDA 推荐的 hash trick / freq truncate 全部通过门槛**：obs/row 都在 1k-3k 量级，跟 baseline 同
3. **fid 34 是边缘 case**：305 obs/row 在门槛附近，加上 EDA 判信号弱，建议保 skip 不再投入 A/B

**新规则**：写复活方案前先算 obs/row，< 1000 直接砍掉这条路径，不要浪费 A/B 配额。


