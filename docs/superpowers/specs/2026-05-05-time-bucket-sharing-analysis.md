# Time Bucket Sharing Analysis

Date: 2026-05-05

Question: should the four sequence domains share one recency time embedding and one set of bucket boundaries?

## Current Setup

- Recency channel: shared `time_embedding`, bucket id from `row_timestamp - seq_timestamp`.
- Current boundaries: one global 64-edge hand-written grid from 5s to 365d.
- Delta channel: W2.7 already uses per-domain `delta_embeddings`, but still one global log2 boundary set.

## Existing Real-Data Evidence

Source: `docs/eda/2026-05-01-data-profile.{md,json}`, Section 8.

| domain | tokens | recency p50 | recency p99 | pos0 p50 | poslast p50 | top shared bucket | empty shared buckets | entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seq_a | 202,543,655 | 29.8d | 172.6d | 11.8h | 87.5d | 15.53% | 0 | 3.80 |
| seq_b | 184,483,697 | 21.5d | 174.6d | 12.4h | 130.3d | 15.35% | 0 | 3.93 |
| seq_c | 314,799,897 | 211.3d | 624.1d | 30.3h | 628.4d | 55.73% | 0 | 2.39 |
| seq_d | 441,416,299 | 5.6d | 47.5d | 46.9m | 11.8d | 20.62% | 12 | 3.92 |

Shared recency bucket JS divergence, bits:

| pair | JS divergence |
|---|---:|
| seq_a / seq_b | 0.0495 |
| seq_a / seq_c | 0.4123 |
| seq_a / seq_d | 0.3055 |
| seq_b / seq_c | 0.4229 |
| seq_b / seq_d | 0.2440 |
| seq_c / seq_d | 0.7459 |

## Interpretation

Sharing one recency embedding is probably too coarse.

The strongest evidence is not just that p50 differs. It is that the same bucket ids have very different semantics across domains:

- `seq_c` is a long-horizon sequence. Its p50 is already around 211 days, and 55.7% of all tokens collapse into one top bucket.
- `seq_d` is a short-horizon sequence. Its p50 is 5.6 days, pos0 median is under 1 hour, and 12 shared buckets are unused.
- `seq_a` and `seq_b` are much closer to each other, so sharing between only those two is defensible, but sharing all four is not.

This means the current shared embedding asks one vector table to learn incompatible meanings:

- a late bucket can mean normal middle-mass behavior for `seq_c`;
- the same late bucket can mean rare stale behavior for `seq_d`;
- a very early bucket is useful for `seq_d`, but almost absent in `seq_c`.

The model can partly recover this because sequence domain tokens are separate elsewhere, but the additive time vector itself is domain-blind. That is avoidable leakage of representational capacity.

## Recommendation

Do this in two separate A/B steps:

1. **Per-domain recency embedding, same global boundaries**
   - Replace shared `time_embedding` with `time_embeddings[domain]`.
   - Keep current `BUCKET_BOUNDARIES`.
   - This isolates the cleanest hypothesis: same bucket id, domain-specific meaning.
   - Parameter cost is tiny: `4 * 65 * d_model`, about 16.6k params at `d_model=64`.

2. **Per-domain recency boundaries**
   - Only try after step 1 shows non-negative movement.
   - Generate boundaries from `tools/time_bucket_eda.py` quantile candidates.
   - This is a data pipeline + checkpoint compatibility change, so it is a higher-blast-radius second move.

For delta buckets, keep W2.7 as-is for now:

- delta embedding is already per-domain;
- boundaries are log2 and less semantically tied to recency horizon;
- changing both recency and delta boundaries at once makes attribution muddy.

## EDA Tool

Added `tools/time_bucket_eda.py`.

Run on real data:

```bash
python tools/time_bucket_eda.py \
  --data-dir "$TRAIN_DATA_PATH" \
  --schema-path "$TRAIN_DATA_PATH/schema.json" \
  --max-rows 200000 \
  --out-md output/time_bucket_eda.md \
  --out-json output/time_bucket_eda.json
```

Use `--max-rows 0` for full scan if IO budget allows.

The tool reports:

- per-domain recency quantiles;
- per-domain adjacent delta quantiles;
- current shared bucket occupancy, empty buckets, entropy, top bucket share;
- pairwise JS divergence over shared recency bucket distributions;
- quantile-derived candidate per-domain recency boundaries.

## Proposed Experiment Matrix

| run | recency embedding | recency boundaries | delta |
|---|---|---|---|
| baseline | shared | global | off or current baseline |
| W2.7 current | shared | global | per-domain emb, global log2 boundaries |
| W2.7.1 | per-domain | global | unchanged |
| W2.7.2 | per-domain | per-domain quantile | unchanged |

Decision rule:

- If W2.7.1 improves, keep per-domain recency embedding.
- If W2.7.1 is flat but W2.7.2 improves, the problem was mostly boundary resolution.
- If both are flat or negative, the sequence encoder already extracts enough domain context and the extra time channel is redundant.
