# TAAC 下一步行动路线：文章原则 + v7 架构图

**Status**: v0.1
**Goal**: 把三篇文章的共识和 v7 开源结构图落成当前代码库的执行顺序。

---

## 0. 总判断

接下来不要继续“扫参数”或“裸加深”。三篇文章和 v7 图共同指向一个主线：

> 先修输入表示和异构信息对齐，再做长序列检索式读取，最后才考虑深度 scaling。

当前 baseline 的核心问题不是数据没进模型，而是很多数据进模型前被粗糙处理：

- user dense 918 维被压成 1 个 token。
- user/item int 通过 RankMixer 机械切块，语义不一定对齐。
- multi-hot 全部 mean pooling，丢掉数量、稀有值、局部结构。
- sequence sideinfo 全部 concat-project，没有区分 item/action/stat 角色。
- 长序列目前是固定截断或固定 top-k，模型没有主动选择保留哪些历史。

所以行动路线分 5 个阶段。

---

## Phase 1: Representation Audit

先查清楚 baseline 到底在哪里把信息压没。这个阶段不改训练主链路，只产出审计结果。

### 1.1 RankMixer chunk 审计

目的：

- 看当前 `user_ns_tokens=5`、`item_ns_tokens=2` 时，每个 token 混了哪些 fid。
- 判断强语义 fid 是否被机械切块稀释。

要做：

- 写 `tools/inspect_tokenization.py`。
- 输出 user/item int fid -> offset/length/vocab/chunk_id。
- 输出每个 chunk 的总维度、包含 fid、是否跨过大 multi-hot fid。

验收：

- 得到一张 chunk 表。
- 明确哪些 chunk 语义混杂严重，为 v7 typed tokenization 提供依据。

### 1.2 Dense 分组审计

目的：

- 验证 v7 的 dense group 划分是否适合我们数据。

重点 fid：

- emb group: `61`, `87`
- stat group: `62-66`
- quantile/rank group: `89-91`

要做：

- 输出每组维度、全零率、p50/p99/max、log1p 后分布。
- 对 `62-66` 查 int/dense element-wise 对齐关系。
- 对 `89-91` 查全零 mask、rank vector 趋势和 label proxy。

验收：

- 决定 stat group 是否用 `log1p+clip`。
- 决定 quantile group 是否需要 all-zero mask。

### 1.3 Sequence role 审计

目的：

- 验证 v7 `SemanticSeqEmbedder` 的 item/action/stat 三角色是否成立。

v7 图给的 role：

| Domain | item role | action role | stat role |
|---|---|---|---|
| A | 38 | 40 | 42-45 |
| B | 69 | 68 | 70-79 |
| C | 29 | 28 | 30-37 |
| D | none | 17 | 18-25 |

要做：

- 对 item role fid 38/69/29 查覆盖率、hash 桶 obs/row、topK 覆盖。
- 对 action/stat role 查唯一值、非零率、每值 pos rate。
- 对 `seq_d` 无 item role 做确认：是否真没有候选 item-like fid。

验收：

- 明确 `69/29` 复活方式：hash / truncate / keep skipped。
- 明确 semantic seq 是否先只做 B/C，还是四个 domain 一次做。

---

## Phase 2: DenseGroupProjector

这是第一刀。原因：

- 工程量小于重写 sequence。
- 命中 v7 图最明确的新模块。
- 正好修正当前 `user_dense_feats -> 1 token` 的过度压缩。
- 也承接 W2.6 v1 失败后的新范式：不再 hard-code weighted pool，而是 typed projection。

### 2.1 模块设计

新增 `DenseGroupProjector`：

- `emb_group = fid61 + fid87 -> Linear + LN + SiLU -> 1 token`
- `stat_group = fid62-66 -> log1p/clip -> Linear + LN + SiLU -> 1 token`
- `quantile_group = fid89-91 -> QuantileTrendEncoder -> 1 token`

输出从当前 1 个 dense NS token 变成 3 个 dense NS tokens。

### 2.2 QuantileTrendEncoder

输入：

- fid 89/90/91 reshape 成 `(B, 3, 10)`。

结构：

- `Conv1d -> SiLU -> zero-init Conv1d -> AdaptiveAvgPool1d(1) -> LayerNorm`

注意：

- zero-init 是稳定性设计，初始尽量不破坏 baseline。
- fid 91 全零率高，需要显式 zero mask 或让 encoder 看到 all-zero pattern。

### 2.3 代码改动

预计改：

- `src/model.py`
- `src/train.py`
- `src/infer.py`
- 可能新增 `tools/inspect_tokenization.py`

CLI flags：

- `--dense_group_projector {none,v7}`
- `--stat_dense_transform {raw,log1p,log1p_clip}`
- `--no_quantile_trend`

### 2.4 实验

顺序：

1. baseline
2. dense groups only
3. dense groups + quantile trend

判定：

- 先 3 epoch 看是否训练稳定。
- 完整训练后必须 test 校准，因为 dense 方向已有 val/test divergence 历史。

---

## Phase 3: SemanticSeqEmbedder

第二刀是 sequence 输入语义重构。原因：

- 旧 EDA 已显示长序列和 emb_skip 是大金矿。
- v7 图不是简单拉长序列，而是先把 sequence token 按 role 重新表示。
- 文章强调异构性进入 backbone 前要完成语义对齐。

### 3.1 模块设计

当前：

`all sideinfo embeddings concat -> Linear -> seq token`

改为：

`item_role + action_role + stat_role + time_embedding + domain_type_embedding -> seq token`

每个 role：

- item role: 高基数 item-like fid，走 hash/truncate embedding。
- action role: 低基数 action fid embedding。
- stat role: 多个 stat fid embedding concat/sum 后投影。

### 3.2 第一版范围

不要一口吃全量。建议：

- 先做 `seq_b` 和 `seq_c`，因为 B/C 有最明显的高基数 item role：fid 69/29。
- `seq_a fid38` 已低于 skip threshold，后续再纳入。
- `seq_d` 无 item role，先按 action/stat 做 typed embed。

### 3.3 依赖

必须解决：

- fid 69 hash 桶数。
- fid 29 truncate/hash 方案。
- reinit threshold=0 下 obs/row 必须 >= 1000。

不能做：

- 不能直接 raise `emb_skip_threshold`。

### 3.4 CLI flags

- `--semantic_seq {none,v7}`
- `--semantic_seq_domains seq_b,seq_c`
- `--seq_item_hash_buckets fid69:171052,fid29:100000`
- `--domain_type_embedding`

### 3.5 实验

顺序：

1. seq_b item role hash only
2. seq_c item role hash/truncate only
3. B+C semantic seq
4. all-domain semantic seq

判定：

- val/test 同向才保留。
- 若 val 涨 test 跌，回到审计结果看是否 hash 冲突或 role 划分错。

---

## Phase 4: DomainGate + CrossNet Head

第三刀是融合和 head。原因：

- 前两步产生了更干净的 typed tokens。
- 需要让不同 domain 的信息动态影响 NS tokens。
- v7 图明确有 DomainGate 和 LowRankCrossNet。

### 4.1 DomainGate

设计：

- 每个 domain 做 masked mean summary。
- 用 summary 生成 domain weights。
- weighted domain summary residual 回写到 NS tokens。
- zero-init，初始近似 baseline。

适合单独 A/B，因为风险低。

CLI:

- `--domain_gate`
- `--domain_gate_zero_init`

### 4.2 LowRankCrossNet Head

设计：

- 保留当前 Q output。
- 加入 final NS tokens mean pool。
- 拼成 head input，过 LowRankCrossNet。
- 再进 prediction MLP。

CLI:

- `--head_type {baseline,crossnet}`
- `--crossnet_layers 2`
- `--crossnet_rank_ratio 0.5`

判定：

- 先和 baseline tokenization 搭配试一次，确认 head 本身不崩。
- 最终与 DenseGroupProjector / SemanticSeqEmbedder 组合。

---

## Phase 5: Scaling Reopen

只有前面输入侧至少一个方向有效，才重新打开 scaling。

### 5.1 LayerScale

原因：

- 裸 `hyformer_blocks=4` 已失败。
- LayerScale 是深度稳定器，不是单独提分点。

设计：

- 在 block 内主要残差分支加 learnable gamma。
- init `1e-4`。

实验：

1. 2 blocks + LayerScale，确认不掉点。
2. 4 blocks + LayerScale。
3. 若 4 blocks 稳，再考虑 Stochastic Depth。

### 5.2 暂不做项

- 不单独重试 warmup+cosine：已有 test 反向。
- 不直接上 6-12 层。
- 不迁移 fixed pyramid compression。
- 长序列只考虑 attention-aware / query-aware retrieval。

---

## 总优先级

### P0: 立刻做

1. `tools/inspect_tokenization.py`
2. dense group offset / chunk 审计
3. `DenseGroupProjector`
4. `QuantileTrendEncoder`

### P1: 下一轮做

1. SemanticSeqEmbedder role 审计
2. fid 69 / 29 hash 或 truncate
3. B/C domain typed sequence embedding

### P2: 组合增强

1. DomainGate
2. LowRankCrossNet head

### P3: 重新 scaling

1. LayerScale
2. 4 blocks
3. Stochastic Depth

---

## 最小闭环计划

### Day 1

- 写 tokenization 审计脚本。
- 输出 RankMixer chunk 表和 dense group offset 表。
- 确认 v7 dense 分组是否和 schema 完全一致。

### Day 2

- 实现 DenseGroupProjector + flags。
- CPU/demo shape test。
- 平台跑 3 epoch sanity。

### Day 3

- 跑完整 dense group A/B。
- 如果不崩，提交 test 校准。
- 同步做 seq role 审计。

### Day 4-5

- 实现 SemanticSeqEmbedder 最小版：先 B/C item role。
- 跑 B-only / C-only / B+C 三组短训。

### Day 6+

- 加 DomainGate。
- 加 CrossNet head。
- 组合最佳输入侧方案。

---

## 当前不该做的事

- 不要继续裸扫 dropout、LR、depth。
- 不要再做 hard-coded dense weighted pool。
- 不要直接 raise `emb_skip_threshold`。
- 不要把长序列理解成单纯增大 cap。
- 不要直接照 v7 上 12 层。

---

## 一句话路线

先用 EDA 把 baseline 的 tokenization 错位找出来；第一刀改 dense typed tokens；第二刀改 sequence typed tokens；第三刀加 domain/head 融合；最后才在 LayerScale 保护下重新尝试 scaling。

