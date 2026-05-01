# TAAC 2026 PCVR Baseline 改进计划（W1-W2）

**Status**: v0.3.1 — W1.0.1 reinit A/B 实验完成：threshold=10000 比默认 0 **掉 0.0015**，确认作者刻意激进 cold restart 是对的；F15 重新表述（文档错，代码故意），W2.5 砍掉 threshold 扫参
**Author**: brainstorming session 2026-05-01
**Branch**: this doc lives on `main`; implementation work happens on feature branches
**Deadline**: 提交截止 2026-05-23 AOE
**Goal**: 单模型 test AUC 从 0.811492 → ≥ 0.825（接近 / 超过 top1 当前 0.82879）

## Changelog

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
| F15 | **`reinit_cardinality_threshold=0` 是刻意的激进 cold restart 设计**（v0.3.1 重新表述）：CLI help（`train.py:158`）写 "0 = never reset" **是错的**，但代码 `model.py:1498` 的 `vs > 0` 即"重置全部已建 embedding" **是作者本意**。W1.0.1 A/B 实验确认：threshold=10000（保留低基数 emb）比默认 0（全员重置）**掉 0.0015 val AUC**。**reinit 范围**：所有已建 seq embedding（不含 emb_skip 跳过的 4 个高基数）+ 所有 user/item NS tokenizer embedding；**reinit 不影响**：time_embedding、dense 参数、emb_skip 跳过的表 | `trainer.py:355` + `model.py:1470-1519` + W1.0.1 实验 | baseline 已在"重正则化高原"，W2 加更强正则化的边际收益要保守估；CLI help 可本地修但不影响平台行为 |
| F16 | **LongerEncoder top-k 方向反了**：`model.py:691` `start_pos = valid_len - actual_k` 取序列尾部；F5 已确认序列倒序 pos 0=最近，所以代码实际取的是**最老**而非最新 token；当前 baseline 用 transformer 没触发，但 W1.7 走 longer encoder 必须先修 | `model.py:670-719` 实测 | W1.7 / W2 长序列实验前置任务 |
| F17 | **dense AdamW 没暴露 weight_decay**：`trainer.py:87` `torch.optim.AdamW(dense_params, lr=lr, betas=(0.9, 0.98))` 未传 weight_decay → 用 PyTorch 默认 0.01；CLI 也只有 `--sparse_weight_decay` | `trainer.py:87` 实测 | W2.1 (dropout, wd) 矩阵扫参前置任务 |

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
| ✅ W1.0.1 | **`reinit_cardinality_threshold` A/B**（v0.3.1 完成）：Run X（默认 threshold=0）val ≈ 0.862，Run Y（threshold=10000）val 掉 **0.0015** → 结论：作者刻意激进 cold restart 是对的，**保留默认 0**。CLI help 文档错（"0=never reset"），代码行为是对的 | 已投入 4h | ✅ 已得结论 + W2.5 简化 | 决定 W2.5 不再扫 threshold |
| W1.0.2 | **加 `--dense_weight_decay` CLI 参数**：`train.py` argparse 加该参数（default=0.01 维持 PyTorch 默认行为不破坏 baseline 复现）；`trainer.py:87` 把值传给 AdamW；`run.sh` 显式加 `--dense_weight_decay 0.01` 锁定值 | 1h | baseline 复现 AUC 不变（±0.001） | W2.1 扫参直接起步 |
| W1.0.3 | **修 LongerEncoder top-k 方向**：`model.py:691-714` 从"取尾 top_k"改成"取头 top_k"（pos 0=最近），padding mask 相应调整；写一个 unit test（构造 valid_len=1000、top_k=50 的输入，验证返回的是 pos[0:50] 而非 pos[950:1000]） | 半天 | unit test 通过 + 切到 `--seq_encoder_type longer` 训练 6-epoch 不掉于 transformer baseline | 改错直接掉点；先 unit test 后训练 |

#### W1.0 执行优先级

- ✅ **W1.0.1 已完成（v0.3.1）**：threshold=10000 比默认 0 掉 0.0015 → 保留默认；W2.5 简化
- 剩余 W1.0.2 / W1.0.3 是纯代码改动，1 天内并行做完即可

---

### W1 任务清单

| # | Deliverable | 投入 | 验收标准 | 风险 |
|---|---|---|---|---|
| ✅ W1.1 | **AMP/bf16 + compile 接入（v0.3 部分完成）**：实测数据 — Baseline 52min/epoch、AMP only 34min（**-35%**, val AUC +0.00003）、compile only 20min（**-62%**, val AUC -0.000149），二者均通过 ≤0.001 漂移阈值。**结论**：compile 是主加速器，AMP 单独收益受限（embedding op 没拿到 bf16 加速）。**待补**：(a) AMP+compile 叠加跑 1 次 6-epoch 验证不冲突；(b) 用 1 次每日 test 提交确认 test AUC 同步不掉点（不掉超 0.002 即认为 W1.1 完成） | 已半天 + 待补 2-3h | val AUC 漂移 ≤ 0.001 ✅；step 时间 -30%+ ✅；test AUC 验证待补 | AMP+compile 叠加可能冲突 → fallback 到 compile-only |
| W1.2 | **多切法 holdout（v0.3 改实施方法）**：trainer 不动，evaluate 时 dump 全 valid 集的 (user_id, timestamp, label, pred) 到 csv；写后处理脚本 `tools/eval_multi_split.py` 算 3 个切法 AUC——`tail_rg`（当前 baseline）、`tail_time`（按 timestamp 取尾 10%）、`user_hash`（`xxhash.xxh64(uid).intdigest() % 10 == 0` 隔离 10%）。**用 xxhash 不用 Python `hash()`**——后者跨进程随机不稳定 | 半天（trainer dump 1h + 脚本 3h） | 跑一次训练后能产出 3 个独立切法的 val AUC，数字稳定可重复 | xxhash 是新依赖（pip install xxhash） |
| W1.3 | **多分桶评估脚本** `tools/eval_breakdown.py`：valid 集按 (a) 高/低频 user (b) 高/低频 item (c) seq 长度 0/短/中/长 (d) 标签 0/1 比例 分桶，报每桶 AUC | 1 天 | 输出一张诊断表，找出过拟合最严重的子群 | 桶切太碎导致每桶样本少 AUC 不可信——按训练集 user/item 频次切，而非 valid 频次 |
| W1.4 | **真实数据 reproduce baseline + AMP**：用 AMP 重跑用户的 0.8615 val + 0.811 test 配置，确认实验环境一致 | 3 小时 | val AUC 在 0.8605-0.8625 区间（±0.001），test AUC 在 0.810-0.813 | AMP 后跑出来跟原始有 gap 说明数值精度损失；启动 fp32 fallback |
| W1.5 | **实验追踪 csv**：`runs/experiments.csv` 每行记录 (run_id, branch, config_diff, val_AUC_taill_rg, val_AUC_tail_time, val_AUC_user_hash, test_AUC_if_have, time, notes) | 1 小时 | 后续每次 train 自动追加一行 | 容易忘记填——建议 train.py 退出时强制 dump 一次 |
| W1.6 | **OOV 频次统计（v0.3 加分母）**：dataset 当前 `_oob_stats` 只记 OOV 计数无总曝光分母 → 加每特征曝光 counter；valid 当前 num_workers=0 单进程没汇总问题，但要在文档里标注"valid 改 num_workers > 0 时必须 manual reduce 各 worker stats"；输出 `oov_rate = oob_count / total_count` 按率排序 | 半天 | 输出按 OOV 率排序的特征列表，含分母 | train 集理论 OOV=0（vocab 按 train max+1 建），重点测**线上完整数据集**含 test 隐式 OOV |
| W1.7 | **长序列显存试点（v0.3 改 encoder + 前置 bug 修复）**：原计划用 transformer 拉序列，但 attention O(L²) 在 L=2000+ 撞墙——改成走 `--seq_encoder_type longer`（**前提：W1.0.3 已修 top-k 方向 bug**）。在线上数据上二分查找 (seq_a, seq_b, seq_c, seq_d) 各自 OOM 边界；**优先 AMP-only 配置**（不带 compile，省 33% 显存，W1.1 实测）；如果 longer 修完后 AUC 不掉于 transformer baseline 则升级 seq_max_lens | 半天 | 给出 4 个 domain 各自 OOM 边界 + longer encoder 在拉长序列后的 val AUC 对比 | 线上 list dim 比 demo 大 2.3×（user_int_feats_66：66→150），显存压力远大于 demo |
| ~~W1.8~~ | ~~修复 schema.json 的 ts_fid~~ | ~~半天~~ | ~~**取消**：线上 schema 已正确设置 ts_fid（F11），time_bucket embedding 在线上正常工作~~ | — |
| **W1.10**（v0.3 扩） | **排查并复活 emb_skip 跳过的高基数 seq 特征**：详细方案见**附录 D**。三步走——(1) 跑 spot-check 脚本看 4 个 fid 取值分布（1h）；(2) 按附录 D 决策树选复活方案（A: hash trick / B: bucket / C: dense / D: 保持 skip）；(3) 改 schema + dataset 实施 + 跑 1 次 6-epoch A/B 验证 | 1.5 天（spot-check + 实施 + A/B） | 至少 1 个被复活的特征带 ≥ 0.001 val AUC 提升；其他特征做出明确保留/丢弃判断 | 4 个 fid 可能本来就是噪声，复活反而掉点；A/B 必做，掉点立即回退 |
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

### W2 必做项（不论诊断结果）

| # | Deliverable | 投入 | 验收标准 |
|---|---|---|---|
| W2.1 | **正则化超参矩阵**：5 个对角线组合扫——(dropout, wd) ∈ {(0.05, 1e-4), (0.1, 1e-3), (0.2, 1e-3), (0.3, 1e-2), (0.1, 1e-2)}。**v0.3.1 备注**：W1.0.1 显示 baseline 已在重正则化高原（aggressive cold restart 已生效），更强 dropout/wd 边际收益要保守估，扫到 0.3/1e-2 大概率掉点 | 5 次 train ≈ 1 天（AMP 后） | 找到比 baseline 涨 ≥ 0.003 的最佳点（保守预期 +0.001~0.003） |
| W2.2 | **EMA**：trainer 维护一份 EMA model（衰减 0.999），evaluate 用 EMA model | 半天 | EMA model 比 raw model 高 ≥ 0.001（几乎确定的） |
| W2.3 | **SWA / Checkpoint averaging**：训练结束前最后 N 个 ckpt 做 state_dict 平均 | 半天 | 平均后比单个 ckpt 高 ≥ 0.001 |
| W2.4 | **Label smoothing 试点**：BCE label 1.0/0.0 → 0.95/0.05；一次 A/B | 1 次 train | 跟 baseline A/B 比，若 ≥ 0.001 就保留 |
| W2.5 | **Sparse 控制收紧（v0.3.1 简化）**：~~`reinit_cardinality_threshold` 扫参~~（W1.0.1 已证 0 胜出，砍掉）；只保留 Adagrad weight_decay 扫 {0, 1e-4, 1e-3, 1e-2} | 2-3 次 train | 找到 sparse_wd 最佳点；预期收益 ≤ 0.002（baseline 已在重正则化高原） |

### W2 条件项（看 W1 诊断哪个分支命中）

| 触发条件 | 动作 | 投入 |
|---|---|---|
| `val_tail_time_AUC` 比 `val_user_hash_AUC` 低 ≥ 0.005 | 加 sample-weight 按时间衰减；近期数据 oversample | 1 天 |
| `val_user_hash_AUC` 比 `val_tail_time_AUC` 低 ≥ 0.005 | OOV → UNK token 改造（每个 vocab 加 1 个 UNK row，dataset 把 OOV 映射到 UNK 而非 0）；high-cardinality embedding hash trick / vocab cap | 1.5 天 |
| **任何切法上** OOV 触发率 ≥ 5% | 同上 OOV → UNK 改造 | 半天（合并） |
| W1.7 显存试点显示能开更长序列 + 不严重过拟合 | 把 seq_a/b 开到 384 或 512，重跑 baseline | 3 小时 |
| W1.3 分桶诊断显示长尾 user/item 子群 AUC 显著低 | 长尾子群 oversample 或 loss reweight | 1 天 |

### W2 执行节奏

- **D1-D2（5/9-5/10）**：5 个正则化组合排队跑（2 张卡并行 = 2.5 天压成 1.25 天日历）；同时手写 EMA 和 label smoothing 代码
- **D3（5/11）**：拿正则化最佳组合叠加 EMA 跑一次
- **D4-D5（5/12-5/13）**：根据 W1 诊断分支，做条件项里命中的 1-2 个
- **D6-D7（5/14-5/15）**：把 W2 所有胜出项叠到一个 "集大成" 配置，跑一次作为 W3 起点

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

## 接续动作（用户明天的 TODO）

1. ~~拿到线上 schema → 跟 demo schema 对比~~ ✅ 完成（v0.2）
2. ~~AMP+compile 实验跑出公平对比~~ ✅ 完成（v0.3，compile 主加速器 -62%）
3. ~~W1.0.1 reinit threshold A/B~~ ✅ 完成（v0.3.1，threshold=10000 掉 0.0015 → 保留默认）
4. **W1.0.2 加 `--dense_weight_decay` 参数**（1h，W2.1 扫参前置）
5. **W1.0.3 修 LongerEncoder top-k 方向 + unit test**（半天，W1.7 长序列前置）
6. **W1.10 spot-check 4 个被 skip 高基数 seq 特征**（1h，附录 D.2 脚本）→ 据决策树选复活方案
7. **W1.9 row group 时间分布检查**（1h）
8. **AMP+compile 叠加跑 1 次 6-epoch**（2h，确认两个一起开不冲突）+ 用 1 次每日 test 提交确认 test AUC 不掉
9. （可选）顺手修 `train.py:158` CLI help 文档错误描述（"0=never reset" → "0=most aggressive, reset all built embeddings"）
10. 我（assistant）根据 6+7+8 的结果继续迭代 spec 到 v0.4，并开始具体实现

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

### D.1 emb_skip 与 reinit 是两个独立机制（v0.3 新增）

容易看混的两个 threshold：

| 参数 | 控制什么 | 时机 | run.sh 当前值 | 文档状态 |
|---|---|---|---|---|
| `--emb_skip_threshold` | 是否**建** embedding 表 | 模型构建时（一次性） | **1,000,000**（显式设） | 准确 |
| `--reinit_cardinality_threshold` | 是否**重置**已建 embedding | 每 epoch 末尾 | **0**（默认值） | **错**（见 F15） |

emb_skip 跳过的表**根本没建出来**——forward 返回零向量（`model.py` 中 `if real_idx == -1`），等价于"特征被删除"。reinit 也碰不到它们。两个机制完全独立。

### D.2 4 个被 skip 特征 spot-check 脚本

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
| val AUC 涨 ≥ 0.001 | 保留复活 |
| val AUC 持平（±0.001） | 选省显存的（保留 D：继续 skip） |
| val AUC 掉 ≥ 0.001 | 回退；如果是 modulo，先升级 xxhash 再试一次；仍掉就 fallback 到 D |

每个 fid **独立 A/B**——可能 fid=69 复活有效但 fid=47 复活掉点，最终保留前者跳过后者。


