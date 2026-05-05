# TAAC v7 开源结构迁移梳理

**Status**: draft v0.2 — DenseGroupProjector 首轮校准失败：val 约 +0.001、test 约 -0.003，触发 dense 侧 val/test divergence。Dense v7 从最高优先级降级，后续先推进 SemanticSeqEmbedder 单独实验。
**Context**: 当前 main 是 PCVRHyFormer baseline：RankMixer NS tokenizer + sequence transformer + HyFormer blocks + AMP/compile。已有实验证明裸 depth scaling、warmup+cosine、LongerEncoder 固定压缩、hard-coded weighted pool 都不稳。

---

## 0. 结构图核心判断

这份 v7 结构不是单个 trick，而是一次 **Typed Heterogeneity Fusion**：

- 输入侧先按语义类型重做 tokenization，让 int / dense / rank vector / sequence item / action / stat 在进 backbone 前完成对齐。
- sequence 侧不再把每个 domain 的所有列简单 concat-project，而是把列拆成 item role / action role / stat role，三路 embedding 后相加。
- backbone 仍保留 Homogeneous Hybrid Block，说明主干不一定要推倒重来。
- backbone 后增加 DomainGate，让不同 domain 的 sequence summary 动态回写到 NS tokens。
- 输出侧从单 MLP head 升级为 TokenSE recalibration + mean pool + LowRankCrossNet + AdvPredHead。

对我们最重要的结论：**优先改输入表示和 domain 融合，先不要裸加深模型。**

---

## 1. 与当前代码的差距

| v7 模块 | 图中作用 | 我们当前状态 | 差距判断 |
|---|---|---|---|
| DenseGroupProjector | user dense 按 emb/stat/quantile 三组投成 3 个 NS tokens | `user_dense_feats` 918 维直接 Linear 成 1 个 token | 高优先级，当前明显过度压缩 |
| QuantileTrendEncoder | 处理 fid 89-91 的 3x10 rank/quantile vector | 直接拼进 dense linear；v1 sigmoid weighted pool 失败 | 高优先级，v1 失败不否定这条 |
| SemanticSeqEmbedder | 每个 domain 按 item/action/stat 三路建模并相加 | 每个 domain 所有 sideinfo embedding concat 后 Linear | 高优先级，当前语义混杂 |
| DomainGate | 根据各 domain summary 动态更新 NS tokens | 无，NS tokens 在 block 后只走 backbone 交互 | 中高优先级，可能改善跨域融合 |
| LowRankCrossNet | head 前显式特征交叉 | 只有 `output_proj + classifier` | 中优先级，工程相对独立 |
| NS recalibration / TokenSE | 对最终 NS tokens 做通道/Token重标定 | 当前无 | 中优先级，可作为轻量 head 改造 |
| Pyramid Compression | 按 block 周期截断序列 | LongerEncoder 固定 top-k 已失败 | 暂缓，除非做 attention-aware 版本 |
| LayerScale / depth scaling | 支撑更深 block | 当前裸 4 blocks 失败 | 后置，先不迁移 |

---

## 2. 第一阶段：输入 Tokenization v7

### 2.1 DenseGroupProjector

图中把 user dense 拆成 3 组：

- emb group: fid 61 (256d) + fid 87 (320d)
- stat group: fid 62-66 raw counts
- quantile group: fid 89-91 rank vectors

当前代码把三类完全不同尺度的 dense 一次性投影成 1 个 NS token。这很可能让大尺度 count 淹没 emb/rank 信号，也让 pair 对齐信息在 projection 阶段静默丢失。

建议实现：

- 新增 `DenseGroupProjector`，输入仍是 `user_dense_feats`。
- 按 schema offset 切出 fid 61/87、62-66、89-91。
- emb group: `Linear(576, d_model) + LN + SiLU`。
- stat group: `log1p + clamp/standardize + Linear(312, d_model) + LN + SiLU`。
- quantile group: reshape `(B, 3, 10)`，走 `QuantileTrendEncoder` 输出 1 token。
- 输出 3 个 dense NS tokens，替代当前 1 个 dense token。

Ablation flags:

- `--dense_group_projector {none,v7}`
- `--no_quantile_trend`
- `--stat_transform {raw,log1p,log1p_clip}`

验收：

- 先只改 dense tokenization，不动 sequence/backbone/head。
- 3 epoch 观察 val 曲线，若不崩再跑完整训练。
- test 校准优先级高，因为 dense 相关实验已有 val/test divergence 历史。

### 2.2 QuantileTrendEncoder

图中结构：

`input (B, 3*10=30) -> reshape (B, 3, 10) -> Conv1d -> SiLU -> zero-init Conv1d -> AdaptiveAvgPool1d(1) -> LayerNorm`

这里像是在捕捉 fid 89/90/91 三组 rank vector 的局部趋势，而不是把它们当 30 个普通 dense 数字。

注意事项：

- 第二层 Conv1d zero-init，初始近似无扰动，避免一上来破坏 baseline。
- fid 91 全 0 率高，必须显式带 missing/all-zero mask，或者让 encoder 看到 `is_zero`。
- 这比 v1 sigmoid weighted pool 更合理，因为它不把 rank vector 硬解释成权重。

---

## 3. 第二阶段：SemanticSeqEmbedder

图中每个 domain 拆成三类角色：

| Domain | item_id | action_type | stat cols |
|---|---|---|---|
| A | item_38 | action_40 | stat_42-45 |
| B | item_69 | action_68 | stat_70-79 |
| C | item_29 | action_28 | stat_30-37 |
| D | none | action_17 | stat_18-25 |

我们的当前 `_embed_seq_domain` 是：

`all sideinfo embeddings concat -> Linear(len(vs)*emb_dim, d_model) + time_embedding`

问题：

- item/action/stat 混在同一投影里，语义密度和稀疏性不同。
- 高基数 item fid 69/29 目前被 `emb_skip_threshold=1M` 直接跳过，正好是 v7 的 sequence item role 主字段。
- stat/action 与 item 的作用不同，简单 concat 让 backbone 隐式学对齐，成本高。

建议实现 `SemanticSeqEmbedder`：

- item role: `Embedding(hash/truncate id) + ID dropout + Linear/LN`。
- action role: `Embedding(action) + Linear/LN`。
- stat role: 多个低中基数 stat embedding concat 或 sum 后 `Linear/LN`。
- 三路相加：`token = item_tok + action_tok + stat_tok + time_emb + domain_type_emb`。
- Domain D 没有 item role，就 item_tok=0 或 learnable missing item token。

关键依赖：

- 必须先做 item fid 69/29 的 hash/truncate 方案，否则 v7 seq item role 缺主干信号。
- fid 47 是否也应该作为 C domain item 辅助，需要 EDA 再判。

Ablation flags:

- `--semantic_seq {none,v7}`
- `--semantic_seq_item_mode {skip,hash,trunc}`
- `--domain_type_embedding`

验收：

- 先只替换 sequence embedding，不改 block。
- 优先在 `seq_b fid69` 和 `seq_c fid29` 上做 hash/truncate 小步迁移。

---

## 4. 第三阶段：DomainGate

图中 DomainGate：

`masked mean per domain -> LN -> Linear(D, nD) -> softmax(zero-init) -> weighted sum + residual -> ns_tokens`

直觉：

- 当前 NS tokens 作为静态上下文去生成 query，但 sequence 读完后没有一个轻量机制把 domain-level summary 回写给 NS。
- DomainGate 相当于让 domain summary 重新校准 NS tokens，尤其适合多 domain 行为强弱不一致的样本。

建议实现：

- 在 block stack 后、head 前加入。
- 对每个 domain 的最终 seq tokens 做 masked mean，得到 `(B, num_domains, D)`。
- 对 NS tokens 做 gating residual：初始 zero-init，保证一开始等价 baseline。

Ablation flags:

- `--domain_gate`
- `--domain_gate_zero_init`

验收：

- 作为独立模块测试，要求初始输出与无 gate 近似一致。
- 适合与 SemanticSeqEmbedder 组合，但第一轮可以单独试。

---

## 5. 第四阶段：Output Head v7

图中输出：

`NS recalibration -> mean pool -> LowRankCrossNet -> AdvancedPredictionHead -> logits`

我们当前 head：

`all_q flatten -> output_proj -> classifier`

差距：

- 当前最终只使用 Q tokens 的 flatten 输出，NS tokens 没有作为 head 的显式输入。
- 显式交叉不足，所有交叉都压在 backbone 隐式完成。

建议：

- 保留当前 Q output，同时把最终 NS tokens mean pool 作为 head 输入。
- 新增 LowRankCrossNet 处理 `[q_output, ns_pool]`。
- rank 可取 `d_model / 2`，层数 2。
- AdvPredHead 可先简化为 `MLP + dropout + logits`，不要一次上太复杂。

Ablation flags:

- `--head_type {baseline,crossnet}`
- `--crossnet_layers 2`
- `--crossnet_rank_ratio 0.5`

---

## 6. 暂缓迁移项

### 6.1 裸 depth scaling

已有 F20：`hyformer_blocks=4` 失败。v7 能做深，是因为 LayerScale、StochDepth、projection 解耦和更合理 tokenization 共同支撑。

结论：

- 不要先做 6-12 层。
- 等 Tokenization v7 + SemanticSeqEmbedder 至少一个有效后，再考虑 LayerScale + 4 blocks。

### 6.2 Warmup + Cosine

已有 F19：val 涨但 test 跌。v7 使用不代表可独立迁移。

结论：

- 暂不作为单独实验。
- 若未来深层模型不稳定，再作为稳定训练组件重试。

### 6.3 Pyramid Compression

当前 LongerEncoder 固定压缩已失败。v7 图里 pyramid compression 是 optional，而且另一篇文章也强调应做 attention-aware top-k。

结论：

- 不迁移 fixed truncation。
- 后续若做长序列，应做 query/attention-aware retrieval。

---

## 7. 推荐实施顺序

### Phase A: EDA / 对齐审计

1. 打印 RankMixer 当前 user/item chunk 到 fid 的映射，确认哪些 fid 被混在一起。
2. 对 fid 62-66、89-91 重新做 dense 分布和 label proxy，决定 log/clip/mask。
3. 对 seq item role fid 69/29/38 做覆盖率、hash 桶 obs/row、label proxy。
4. 对 action/stat role 做每 fid 非零率、唯一值、pos-rate 曲线，确认 role 划分。

### Phase B: 最小可用 v7 tokenization

1. 实现 `DenseGroupProjector`，把 user dense 1 token 改成 3 tokens。
2. 实现 `QuantileTrendEncoder`，先 zero-init，默认打开。
3. 增加 ablation flags，确保 `--no_dense_groups` 可回退 baseline。

### Phase C: SemanticSeqEmbedder

1. 先只对 seq_b/seq_c 的 item role 做 hash/truncate。
2. 替换 `_embed_seq_domain` 为 typed role sum。
3. 保持 backbone/head 不变做 A/B。

### Phase D: DomainGate + CrossNet head

1. 增加 DomainGate，zero-init residual。
2. 增加 LowRankCrossNet head。
3. 组合最佳 tokenization 再做完整训练。

### Phase E: Scaling 再打开

1. 加 LayerScale。
2. 试 4 blocks。
3. 若 4 blocks 稳，再考虑 Stochastic Depth。

---

## 8. 当前最高 ROI 实验

优先级排序：

1. **SemanticSeqEmbedder with item role hash / typed roles**
   - 命中 v7 的 sequence redesign。
   - DenseGroupProjector 已出现 val/test 反向，输入侧主线转向 sequence typed tokenization。
   - 复活 fid 69/29 的方式比简单 emb_skip threshold 更合理。

2. **DenseGroupProjector + QuantileTrendEncoder（降级）**
   - 首轮：val 约 +0.001 / test 约 -0.003。
   - 结论：typed dense projection 能拟合本地 val，但 test OOD 反向；不要与后续方案默认组合。
   - 后续若重启，只做更强约束版本：zero-init residual、drop quantile、或只保留 emb group。

3. **DomainGate**
   - zero-init，风险低。
   - 可能改善 multi-domain 融合。

4. **CrossNet head**
   - 与 backbone 解耦，容易 A/B。

5. **LayerScale + 4 blocks**
   - 仅在前面至少一个输入侧改造有效后再做。

---

## 9. 需要避免的误读

- v7 不是让我们直接堆深度；它先重做了输入语义对齐。
- v7 不是证明 warmup+cosine 一定有效；我们已有 test 反向证据。
- v7 不是简单“用更多 token”；关键是 typed tokens，而不是机械增 token。
- v7 的 DenseGroupProjector 不是 v1 weighted pool；它是把 dense 子模态分组投影，不是用 dense 值硬当权重。
- v7 的 sequence item role 依赖高基数 fid 的合理复活，不能直接 raise `emb_skip_threshold`。

---

## 10. 下一步具体动作

1. 写 `tools/inspect_tokenization.py`：
   - 输出 user/item RankMixer chunk 对应 fid。
   - 输出 user dense fid offset。
   - 输出 v7 dense group 切片维度检查。

2. 在 `model.py` 增加：
   - `DenseGroupProjector`
   - `QuantileTrendEncoder`
   - CLI flags wiring

3. 第一轮 A/B：
   - baseline
   - dense groups only
   - dense groups + quantile trend

4. 若 dense groups 不崩，再推进 `SemanticSeqEmbedder`。
