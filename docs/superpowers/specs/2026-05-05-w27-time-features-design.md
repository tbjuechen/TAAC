# W2.7 Time Features Design Spec — delta-t Bucket Embedding

**Status**: design / pending implementation
**Branch**: `feature/time-features`
**Date**: 2026-05-05
**Parent spec**: `docs/superpowers/specs/2026-05-01-taac-improvement-plan.md` v2.2 (W2.7 entry)
**EDA**: `docs/eda/2026-05-01-data-profile.md` Section 8 (Sequence timestamp per domain)

---

## 1. Goal

Add a **second time channel** to seq token embeddings, encoding **adjacent-token time gaps** (`delta_t`) — a "user behavior pacing" signal currently absent from baseline.

**Context**:
- Baseline only models `time_diff = row_ts − seq_ts[i]` (recency: "how long ago"), via 65 hand-crafted buckets and 1 shared `nn.Embedding(65, d_model)` (`dataset.py:625-668`, `model.py:1395, 1589`).
- Baseline does **not** model `delta_t = seq_ts[i] − seq_ts[i+1]` (engagement: "how dense user actions are").
- Intuition: 2s gap = misclick / fast scroll; 10min gap = deep browse; multi-day gap = cross-session return. PCVR signal differs across regimes.

**Success criteria** (gated, see §6):
- Phase 1 (val, 3 epoch): `val Δ ≥ +0.001`
- Phase 2 (test, 6 epoch): `val Δ ≥ +0.001 AND test Δ ≥ +0.0008` (same direction)

**Expected gain**: +0.001 ~ +0.005 (W2.7 alone). Reasoning:
- xhs hint suggests time-features ≈ +1%, but xhs likely covers (a) absolute periodic time, (b) delta, (c) decay-weighted pool. (b) and (c) overlap with W2.6 v1 — already failed (F27). (a) is left for a separate spec. So W2.7 v1 captures only one of three xhs sub-signals.

---

## 2. Architecture

### 2.1 Overview

```
PCVRHyFormer._embed_seq_domain(domain, seq, time_bucket_ids, delta_bucket_ids):
    cat_emb   = concat(sideinfo_embs)                                # baseline
    token_emb = gelu(seq_proj(cat_emb))                              # baseline
    if num_time_buckets > 0:
        token_emb = token_emb + time_embedding(time_bucket_ids)      # baseline (shared)
    if num_delta_buckets > 0:
        token_emb = token_emb + delta_embeddings[domain](delta_bucket_ids)  # NEW
    return token_emb
```

### 2.2 Component table

| Component | Shape | Params | Status |
|---|---|---|---|
| Baseline `time_embedding` (shared) | `nn.Embedding(65, 96)` | 6,240 | unchanged |
| **NEW `delta_embeddings` (per-domain × 4)** | `4 × nn.Embedding(33, 96)` | **12,672** | new |

### 2.3 Invariants (do NOT break)

1. **Baseline `time_embedding` unchanged** — single-variable A/B
2. Delta channel uses `padding_idx=0` → zero-vector for "no neighbor" / padding (semantically: "missing previous action")
3. Delta channel goes through reinit-preserve path (vocab = 33, never reinitialized), same as baseline `time_embedding`
4. Delta channel uses sparse Adagrad path (matches existing `time_embedding` — `get_sparse_params` collects all `nn.Embedding`)
5. Default OFF (`--use_delta_buckets=False`); existing checkpoints / experiments unaffected

### 2.4 Orthogonality with other W2 candidates

| Other spec | Conflict? | Reason |
|---|---|---|
| W1.7 (longer seq, transformer + cap 1024) | none | delta is computed in dataset, model touches token embedding only |
| W1.10 (emb_skip hash trick) | none | delta operates on `ts_*` fids (vocab=0), independent of sideinfo fids |
| W2.6 v2 (pair features) | **partial** | W2.6 v2 candidate (γ) explicit pair embedding may also use (seq_ts, row_ts). **Mitigation**: W2.7 owns token-level time channels; W2.6 v2 should focus on `(int_fid, dense_fid)` pairs only |
| W2.8 (LR sweep) | none | sequential, not parallel |

---

## 3. Data Flow (`dataset.py`)

### 3.1 New constant `DELTA_BOUNDARIES`

Co-locate with existing `BUCKET_BOUNDARIES` (`dataset.py:110`):

```python
# Delta-t bucket boundaries: log2 spacing from 1s to ~34 years.
# 31 boundaries -> bucket ids in [1..32]; padding=0; total 33 slots.
DELTA_BOUNDARIES = np.array([
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
    4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288,
    1048576, 2097152, 4194304, 8388608, 16777216, 33554432,
    67108864, 134217728, 268435456, 536870912, 1073741824,
], dtype=np.int64)
NUM_DELTA_BUCKETS = len(DELTA_BOUNDARIES) + 2  # +1 padding, +1 upper-bound = 33
```

### 3.2 Bucket physical meaning (selected)

| bucket | delta_t range | semantic |
|---|---|---|
| 0 | — | padding / no neighbor |
| 1 | < 1s | extreme repeat |
| 3 | 2–4s | misclick |
| 8 | 64–128s | ~1–2 min (deep-browse candidate) |
| 11 | 512–1024s | ~8–17 min |
| 14 | 4096–8192s | ~1–2 h |
| 17 | 32768–65536s | ~9–18 h (overnight) |
| 20 | 262144–524288s | ~3–6 d (cross-week) |
| 23 | 2.1M–4.2M s | ~24–48 d |
| 26 | 16.8M–33.5M s | ~half year–1 year |
| 32 | > 1.07e9 | overflow sentinel (data anomaly indicator) |

### 3.3 Compute logic (after existing time_bucket fill, ~`dataset.py:667`)

Re-uses `ts_padded` (already padded for time_bucket) and `_buf_seq_db` buffer (new, mirrors `_buf_seq_tb`):

```python
delta_bucket = self._buf_seq_db[domain][:B]
delta_bucket[:] = 0
if ts_ci is not None:
    # forward diff: delta_t[i] = ts[i] - ts[i+1]  (reverse-order: pos 0 = most recent)
    delta_t = np.zeros((B, max_len), dtype=np.int64)
    delta_t[:, :-1] = ts_padded[:, :-1] - ts_padded[:, 1:]
    delta_t = np.maximum(delta_t, 0)  # safety: out-of-order data noise
    raw = np.clip(
        np.searchsorted(DELTA_BOUNDARIES, delta_t.ravel()),
        0, len(DELTA_BOUNDARIES),
    )
    buckets = raw.reshape(B, max_len) + 1  # shift to [1, 32]
    pair_valid = (ts_padded[:, :-1] > 0) & (ts_padded[:, 1:] > 0)
    buckets[:, :-1][~pair_valid] = 0
    buckets[:, -1] = 0  # pos last has no neighbor
    delta_bucket[:] = buckets
result[f'{domain}_delta_bucket'] = torch.from_numpy(delta_bucket.copy())
```

### 3.4 Buffer pre-allocation (`__init__`)

Mirrors `_buf_seq_tb`:

```python
self._buf_seq_db: Dict[str, np.ndarray] = {}
for domain in seq_domains:
    self._buf_seq_db[domain] = np.zeros((max_batch, max_len[domain]), dtype=np.int64)
```

### 3.5 `ModelInput` NamedTuple (model.py:13-19)

Add field:

```python
class ModelInput(NamedTuple):
    ...
    seq_time_buckets: dict
    seq_delta_buckets: dict  # NEW: {domain: tensor [B, L]}
```

### 3.6 trainer.py — build `ModelInput`

Wherever `ModelInput(...)` is constructed, add:

```python
seq_delta_buckets={d: batch[f'{d}_delta_bucket'].to(device) for d in seq_domains},
```

---

## 4. Model Changes (`model.py`)

### 4.1 `__init__` — new param + embedding creation

```python
def __init__(
    self,
    ...
    num_time_buckets: int = 65,
    num_delta_buckets: int = 0,   # NEW: 0 = disabled (default)
    ...
):
    ...
    self.num_time_buckets = num_time_buckets
    self.num_delta_buckets = num_delta_buckets

    if num_time_buckets > 0:
        self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)

    if num_delta_buckets > 0:
        self.delta_embeddings = nn.ModuleDict({
            d: nn.Embedding(num_delta_buckets, d_model, padding_idx=0)
            for d in self.seq_domains
        })
```

### 4.2 `init_weights` — xavier + zero-pad (model.py:~1486)

```python
if self.num_time_buckets > 0:
    nn.init.xavier_normal_(self.time_embedding.weight.data)
    self.time_embedding.weight.data[0, :] = 0

if self.num_delta_buckets > 0:
    for emb in self.delta_embeddings.values():
        nn.init.xavier_normal_(emb.weight.data)
        emb.weight.data[0, :] = 0
```

### 4.3 `reinit_high_cardinality_params` — preserve count (model.py:~1542)

```python
if self.num_time_buckets > 0:
    skip_count += 1
if self.num_delta_buckets > 0:
    skip_count += len(self.delta_embeddings)
```

### 4.4 `_embed_seq_domain` signature + body

```python
def _embed_seq_domain(
    self,
    seq, sideinfo_embs, proj, is_id, emb_index,
    time_bucket_ids,
    delta_bucket_ids,    # NEW [B, L]
    domain_name,         # NEW (routes to delta_embeddings[domain_name])
):
    ...
    token_emb = F.gelu(proj(cat_emb))
    if self.num_time_buckets > 0:
        token_emb = token_emb + self.time_embedding(time_bucket_ids)
    if self.num_delta_buckets > 0:
        token_emb = token_emb + self.delta_embeddings[domain_name](delta_bucket_ids)
    return token_emb
```

### 4.5 `forward` and `predict` call sites (model.py:~1670, ~1719)

Both symmetric; pass new kwargs through:

```python
for domain in self.seq_domains:
    tokens = self._embed_seq_domain(
        inputs.seq_data[domain],
        self._seq_embs[domain], self._seq_proj[domain],
        self._seq_is_id[domain], self._seq_emb_index[domain],
        inputs.seq_time_buckets[domain],
        inputs.seq_delta_buckets[domain],   # NEW
        domain,                              # NEW
    )
```

### 4.6 `train.py` — CLI flag

```python
parser.add_argument('--use_delta_buckets', action='store_true', default=False,
                    help='Enable per-domain delta-t bucket embedding (W2.7). '
                         'Bucket count is determined by dataset.NUM_DELTA_BUCKETS.')
parser.add_argument('--no_delta_buckets', dest='use_delta_buckets', action='store_false')
```

```python
"num_delta_buckets": NUM_DELTA_BUCKETS if args.use_delta_buckets else 0,
```

---

## 5. Decisions Recap

| Axis | Decision | Alternatives rejected |
|---|---|---|
| Direction | adjacent-token delta_t in seq | absolute periodic time / per-domain time_emb / Fourier features (parked for v2 / separate spec) |
| Encoding | additive discrete bucket (β) | substitute (α, loses recency) / dense scalar (γ, F27 paradigm risk) |
| Boundaries | log2, 32 buckets | reuse `BUCKET_BOUNDARIES` (a, replicates b63 hot/cold issue) / EDA-driven (c, +0.5d work) |
| Granularity | per-domain (4 emb) | shared (β2, ignores 4-domain time-scale variance) / dual per-domain (γ, multi-variable) |
| Edge handling | `padding_idx=0` for "no neighbor" | sentinel bucket 33 (b, +1 row, marginal info) / row_ts fill (c, redundant with baseline) |
| Default | OFF (`--use_delta_buckets=False`) | ON-by-default (would break existing experiment baselines) |

---

## 6. Validation Strategy

### 6.1 Single-variable A/B

| Run | `--use_delta_buckets` | other params |
|---|---|---|
| **A: baseline** | `False` | reuse latest `main` ckpt metrics, no rerun |
| **B: W2.7 v1** | `True` | identical to A: AMP+compile, dense_wd default, reinit_threshold=0, same seed |

### 6.2 Decision gates

**Phase 1 — val sanity (3 epoch)**:
- `val Δ < +0.001` → W2.7 v1 dies; record F28; **do NOT consume test submit**; pivot to next candidate (W2.6 v2 brainstorm or W1.7 sub-c)
- `val Δ ≥ +0.001` → proceed to Phase 2

**Phase 2 — test calibration (6 epoch + 1 test submit)**:
- `val Δ ≥ +0.001 AND test Δ ≥ +0.0008` (same direction) → ✅ merge to main, record F28 success
- `val ↑ AND test ↓` → val/test divergence #6; record F28 failure; v1 paradigm dead
- `val ↓` → impossible (Phase 1 gate would have caught)

### 6.3 Resource budget

- 3 epoch run ≈ 70 min (with AMP+compile, ~23 min/epoch)
- 6 epoch run ≈ 140 min
- 1 test submit (out of daily quota of 3)
- Total: ~3.5 h compute + 1 test submit (only if Phase 1 passes)

### 6.4 Sanity checks (run within 30 min of first run start)

After 1 batch on a single domain:
1. `f'{domain}_delta_bucket'` shape == `[B, max_len]` ✓
2. bucket-0 ratio ≈ 1/L per row (last position + padding); typically 50%-90%
3. Histogram of buckets > 0 — should NOT have a single bucket > 30% (would indicate boundary mismatch like baseline `seq_c b63 = 55.7%`)
4. **Critical**: max delta_t ≤ ts span (~926 days = 8e7s). If samples land in bucket 31/32, **stop and investigate** — likely indicates ts unit mismatch or different epoch base. delta_emb would otherwise learn pure noise.

### 6.5 Logging

`trainer.py` startup banner (when enabled):
```python
if model.num_delta_buckets > 0:
    logging.info(f"W2.7 delta_buckets enabled: NUM_DELTA_BUCKETS={NUM_DELTA_BUCKETS}, "
                 f"per-domain (4 × {NUM_DELTA_BUCKETS} × {d_model}), "
                 f"+{4 * NUM_DELTA_BUCKETS * d_model} params")
```

---

## 7. Risks & Mitigations

| ID | Risk | Likelihood | Mitigation |
|---|---|---|---|
| R1 | per-domain emb overfits 4 separate distributions; seq_d (short, dense delta) data-starved | low | vocab=33 with 907K rows ≫ 1000 obs/bucket threshold (CLAUDE.md reinit constraint); fallback: degrade to shared if seq_d gradient norm < 1/10 of others |
| R2 | log2 too coarse at low end (0-1min has only 6 buckets vs baseline 12) | medium | 6 buckets in 1-32s region matches PCVR <30s "bounce" granularity in literature; if v1 fails, v2 candidate = non-uniform boundaries (densify low end) |
| R3 | conflict with W2.6 v2 (γ) explicit pair emb | low | W2.7 owns token-level time; W2.6 v2 should focus on (int_fid, dense_fid) — coordinated via CLAUDE.md anti-pattern note |
| R4 | val/test divergence #6 (F19/F21/F23/F25/F27 history) | medium | new signal channel ≠ substitution → lower divergence prior than F19 (warmup), F23 (longer), F27 (weighted pool); but still gated by §6.2 Phase 2 mandatory test calibration |
| R5 | dataset CPU overhead | low | additional 1 diff + 1 searchsorted per (batch, domain); est. <5% prep time; baseline 23min/epoch unaffected |
| R6 | tensor ts unit mismatch (epoch base / timezone) | high if data anomaly | §6.4 check 4 catches this; halt-and-investigate path |

---

## 8. Open Questions (parked for v2)

| Q | Trigger | v2 candidate |
|---|---|---|
| log2 vs EDA-driven boundaries | v1 val Δ ∈ (0, +0.001) marginal | EDA delta_t distribution per domain → custom boundaries |
| per-domain vs shared comparison | v1 success | ablation: rerun with shared single emb |
| sparse Adagrad vs dense AdamW for delta_emb | gradient norm anomaly | trainer explicit whitelist split |
| extend to row-level absolute time (hour-of-day, dow) | v1 success | separate spec W2.7.2 |

---

## 9. Non-goals

1. NOT touching baseline `time_embedding` (stays shared 65 buckets)
2. NOT introducing RoPE / Fourier features (parked)
3. NOT time-decay weighted pool (F27 dead paradigm)
4. NOT row absolute periodic time (separate spec)
5. NOT modifying demo schema or `taac2026_schema.json` (not uploaded)
6. NOT changing sparse/dense optimizer split (would add second variable)

---

## 10. Acceptance Checklist (pre-merge to main)

- [ ] Unit: `len(DELTA_BOUNDARIES) == 31` and `NUM_DELTA_BUCKETS == 33`
- [ ] Unit: toy ts seq → expected bucket ids (pos last = 0, padding = 0, normal token ∈ [1, 32])
- [ ] forward does not OOM at `batch=512, seq_d cap=512`
- [ ] `--use_delta_buckets=False` produces a model identical to current `main` HEAD (param diff = ∅)
- [ ] `--use_delta_buckets=True` 3-epoch val Δ ≥ +0.001 (Phase 1)
- [ ] `--use_delta_buckets=True` 6-epoch test Δ ≥ +0.0008 same direction (Phase 2)
- [ ] Update parent spec with F28 (success or failure record)
- [ ] Update CLAUDE.md: append F28 to fact table; if failed, add anti-pattern entry

---

## 11. Implementation order (preview for writing-plans)

1. `dataset.py`: `DELTA_BOUNDARIES`, `NUM_DELTA_BUCKETS`, `_buf_seq_db`, compute logic, output key
2. `model.py`: `ModelInput` field, `__init__`, `init_weights`, `reinit_*`, `_embed_seq_domain`, forward / predict call sites
3. `trainer.py`: `ModelInput` construction, startup banner
4. `train.py`: CLI flags, `model_args` wiring
5. Unit tests (acceptance items 1-2)
6. Local sanity run on demo data (acceptance items 3-4 + sanity §6.4)
7. Platform Phase 1 (3 epoch val); decision gate
8. Platform Phase 2 (6 epoch + test submit) if Phase 1 passes
9. Update spec + CLAUDE.md F28
