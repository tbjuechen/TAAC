"""TAAC data profile script — single-pass data analysis (no model forward).

Output strategy (designed for the contest platform's stdout-only retrieval):
  - **stdout**  : ~675-line markdown report + 1 trailing line
                  ``PROFILE_JSON_DUMP=<json>`` carrying the raw stats
                  for offline parsing. Capped at ≤1000 lines.
  - **stderr**  : progress / warnings via ``logging``; doesn't pollute stdout.
  - **disk**    : optional ``${TRAIN_LOG_PATH}/profile_data.json`` (best effort,
                  not relied upon).

Covers 13 sections (general EDA + 4-direction decision tables); see plan
``docs/superpowers/specs`` and the in-repo plan file for the full spec.

Usage (driven by ``run.sh`` with ``TAAC_MODE=profile``):

    TRAIN_DATA_PATH=/path/to/parquet_dir \\
    TRAIN_LOG_PATH=/path/to/logs \\
    TAAC_MODE=profile bash run.sh

Env vars (mirroring train.py):
  TRAIN_DATA_PATH  Parquet directory (must contain *.parquet + schema.json)
  TRAIN_LOG_PATH   Optional log dir; if set, also dumps profile_data.json there
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================================
# Constants
# ============================================================================

# Time-bucket edges (mirrors dataset.py's BUCKET_BOUNDARIES). Used purely for
# Section 8's bucket-occupancy histogram; we don't enforce embedding ranges.
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1

# Per-fid frequency tracking strategy thresholds.
EXACT_FREQ_VOCAB_LIMIT = 100_000   # ≤ this: use exact dict
MISRA_GRIES_K = 100_000             # heavy-hitter capacity for high-card features
RESERVOIR_SIZE = 10_000             # for percentile estimation
TOPK_REPORT = (10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000)
TOPK_DECISION = (1_000, 10_000, 100_000, 1_000_000, 10_000_000)
COVERAGE_TARGETS = (0.5, 0.9, 0.95, 0.99, 0.999)
EMB_DIM_DEFAULT = 64                # current uniform emb_dim used by the model

# Heuristic thresholds for Section 9 (UNK/OOV) recommendation.
OOV_NOOP_RATE = 0.001          # < 0.1% → no-op
OOV_LOW_RATE = 0.01            # < 1%  → single-bucket UNK
# (anything >= 1% → multi-bucket hash)


# ============================================================================
# Streaming statistics primitives
# ============================================================================


class Welford:
    """Numerically-stable online mean / variance (single dim)."""

    __slots__ = ("n", "mean", "M2", "min_v", "max_v")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_v = float("inf")
        self.max_v = float("-inf")

    def update_array(self, arr: np.ndarray) -> None:
        if arr.size == 0:
            return
        a = arr.astype(np.float64, copy=False).ravel()
        # finite mask
        mask = np.isfinite(a)
        a = a[mask]
        if a.size == 0:
            return
        new_n = a.size
        new_mean = float(a.mean())
        new_var_sum = float(((a - new_mean) ** 2).sum())  # M2 of just this chunk
        if self.n == 0:
            self.n = new_n
            self.mean = new_mean
            self.M2 = new_var_sum
        else:
            delta = new_mean - self.mean
            tot = self.n + new_n
            self.mean += delta * new_n / tot
            self.M2 += new_var_sum + (delta * delta) * self.n * new_n / tot
            self.n = tot
        mn = float(a.min())
        mx = float(a.max())
        if mn < self.min_v:
            self.min_v = mn
        if mx > self.max_v:
            self.max_v = mx

    def to_dict(self) -> Dict[str, float]:
        var = self.M2 / self.n if self.n > 1 else 0.0
        return {
            "n": self.n,
            "mean": self.mean,
            "std": math.sqrt(max(var, 0.0)),
            "min": self.min_v if self.n else 0.0,
            "max": self.max_v if self.n else 0.0,
        }


class ReservoirSample:
    """Reservoir sampling for percentile estimation. Memory O(size)."""

    __slots__ = ("size", "buf", "n", "rng")

    def __init__(self, size: int = RESERVOIR_SIZE, seed: int = 42) -> None:
        self.size = size
        self.buf: List[float] = []
        self.n = 0
        self.rng = random.Random(seed)

    def update_array(self, arr: np.ndarray) -> None:
        if arr.size == 0:
            return
        a = arr.astype(np.float64, copy=False).ravel()
        a = a[np.isfinite(a)]
        for v in a:
            self.n += 1
            if len(self.buf) < self.size:
                self.buf.append(float(v))
            else:
                j = self.rng.randint(0, self.n - 1)
                if j < self.size:
                    self.buf[j] = float(v)

    def percentile(self, p: float) -> Optional[float]:
        if not self.buf:
            return None
        return float(np.percentile(np.asarray(self.buf), p * 100))

    def percentiles(self, ps: Iterable[float]) -> Dict[str, Optional[float]]:
        if not self.buf:
            return {f"p{int(p*100)}": None for p in ps}
        arr = np.asarray(self.buf)
        out = {}
        for p in ps:
            out[f"p{int(round(p*1000))/10}"] = float(np.percentile(arr, p * 100))
        return out


class MisraGries:
    """Approximate top-K tracker (lazy-prune variant of Misra-Gries).

    Algorithm: keep an unbounded Counter; when its size exceeds 2*K, prune
    by retaining only the top-K by current count. This is O(N + (N/K) K log K)
    overall. For typical heavy-hitter distributions (a few hot ids dominate
    the long tail), the top-K stays stable between prunes.

    Memory bound: 2*K entries (≤12 MB at K=100k).
    """

    __slots__ = ("k", "prune_at", "counts", "total")

    def __init__(self, k: int = MISRA_GRIES_K) -> None:
        self.k = k
        self.prune_at = 2 * k
        # value -> count
        self.counts: Dict[int, int] = {}
        self.total = 0

    def update_array(self, arr: np.ndarray) -> None:
        if arr.size == 0:
            return
        a = arr.ravel()
        a = a[a != 0]
        if a.size == 0:
            return
        vals, freqs = np.unique(a, return_counts=True)
        # Bulk dict update via direct loop (numpy doesn't speed up dict ops).
        d = self.counts
        for v, f in zip(vals.tolist(), freqs.tolist()):
            self.total += f
            cur = d.get(v)
            if cur is None:
                d[v] = f
            else:
                d[v] = cur + f
        if len(d) > self.prune_at:
            self._prune()

    def _prune(self) -> None:
        # Keep top-K entries by count.
        items = sorted(self.counts.items(), key=lambda x: -x[1])
        self.counts = dict(items[: self.k])

    def topk(self, k: Optional[int] = None) -> List[Tuple[int, int]]:
        items = sorted(self.counts.items(), key=lambda x: -x[1])
        if k is None:
            return items
        return items[:k]


class FreqStat:
    """Per-fid frequency tracker. Uses exact dict if vocab is small, else
    Misra-Gries.
    """

    __slots__ = (
        "vocab", "exact", "counts", "mg", "total", "obs_min", "obs_max",
        "non_pad_total", "row_count", "row_with_value_count",
    )

    def __init__(self, vocab: int) -> None:
        self.vocab = int(vocab)
        # exact flag also gates when vocab=0 (no info → exact, tiny)
        self.exact = self.vocab == 0 or self.vocab <= EXACT_FREQ_VOCAB_LIMIT
        self.counts: Dict[int, int] = {} if self.exact else {}
        self.mg: Optional[MisraGries] = None if self.exact else MisraGries()
        self.total = 0          # total non-padding values seen
        self.obs_min = None     # observed min (excluding 0)
        self.obs_max = None     # observed max
        self.non_pad_total = 0  # alias for total, kept for clarity
        self.row_count = 0      # total rows visited
        self.row_with_value_count = 0  # rows where at least one non-padding value

    def update_row_batch(self, arr: np.ndarray) -> None:
        """Update from a 1D (single-value) or 2D (multi-value) numpy int array.

        Padding (value=0 or negative) is excluded from frequency stats but
        contributes to row_count.
        """
        if arr.size == 0:
            return
        if arr.ndim == 1:
            self.row_count += arr.shape[0]
            non_pad = arr[arr > 0]
            self.row_with_value_count += int((arr > 0).sum())
        else:
            B = arr.shape[0]
            self.row_count += B
            row_has = (arr > 0).any(axis=1)
            self.row_with_value_count += int(row_has.sum())
            non_pad = arr[arr > 0]
        if non_pad.size == 0:
            return
        # Bound updates
        mn = int(non_pad.min())
        mx = int(non_pad.max())
        self.obs_min = mn if self.obs_min is None else min(self.obs_min, mn)
        self.obs_max = mx if self.obs_max is None else max(self.obs_max, mx)
        self.total += int(non_pad.size)
        self.non_pad_total = self.total
        if self.exact:
            vals, freqs = np.unique(non_pad, return_counts=True)
            for v, f in zip(vals.tolist(), freqs.tolist()):
                self.counts[v] = self.counts.get(v, 0) + int(f)
        else:
            assert self.mg is not None
            self.mg.update_array(non_pad)

    def topk_coverage(self, k_list: Iterable[int]) -> List[Tuple[int, float]]:
        """Cumulative coverage at given K's (sorted descending by freq)."""
        items = self._sorted_items()
        if not items:
            return [(k, 0.0) for k in k_list]
        cum = np.cumsum([c for _, c in items])
        total = self.total if self.total else 1
        out = []
        for k in k_list:
            idx = min(k, len(cum)) - 1
            cov = float(cum[idx]) / total if idx >= 0 else 0.0
            out.append((k, cov))
        return out

    def k_for_coverage(self, target: float) -> Optional[int]:
        """Smallest K whose top-K covers ``target`` fraction of values."""
        items = self._sorted_items()
        if not items:
            return None
        cum = np.cumsum([c for _, c in items])
        total = self.total if self.total else 1
        for i, c in enumerate(cum):
            if c / total >= target:
                return i + 1
        return len(items)

    def gini(self) -> float:
        """Gini coefficient of the freq distribution (excluding padding)."""
        if not self._sorted_items():
            return 0.0
        items = self._sorted_items()
        freqs = np.asarray([c for _, c in items], dtype=np.float64)
        # Sort ascending for standard Gini formula.
        freqs = np.sort(freqs)
        n = freqs.size
        if n == 0 or freqs.sum() == 0:
            return 0.0
        idx = np.arange(1, n + 1, dtype=np.float64)
        return float((2 * (idx * freqs).sum() - (n + 1) * freqs.sum())
                     / (n * freqs.sum()))

    def entropy(self) -> float:
        """Shannon entropy in bits of the freq distribution."""
        if self.total == 0:
            return 0.0
        items = self._sorted_items()
        ps = np.asarray([c for _, c in items], dtype=np.float64) / self.total
        ps = ps[ps > 0]
        return float(-(ps * np.log2(ps)).sum())

    def hhi(self) -> float:
        """Herfindahl index (sum of squared shares; 1 = single value monopoly)."""
        if self.total == 0:
            return 0.0
        items = self._sorted_items()
        ps = np.asarray([c for _, c in items], dtype=np.float64) / self.total
        return float((ps * ps).sum())

    def num_unique(self) -> Optional[int]:
        """Exact unique count (None if Misra-Gries can't give an exact value)."""
        if self.exact:
            return len(self.counts)
        return None  # MG sketch doesn't preserve cardinality

    def long_tail_ratio(self, threshold: int = 5) -> Optional[float]:
        """Fraction of unique values with count < threshold (None if not exact)."""
        if not self.exact:
            return None
        if not self.counts:
            return 0.0
        rare = sum(1 for c in self.counts.values() if c < threshold)
        return rare / max(len(self.counts), 1)

    def _sorted_items(self) -> List[Tuple[int, int]]:
        if self.exact:
            return sorted(self.counts.items(), key=lambda x: -x[1])
        assert self.mg is not None
        return self.mg.topk()


# ============================================================================
# OOV tracker (val pass) + Length / Histogram helpers
# ============================================================================


class OOVTracker:
    """Track values exceeding train_max (= schema vocab) per fid."""

    __slots__ = ("vocab", "value_total", "value_oob", "row_total", "row_oob",
                 "max_oob", "min_oob")

    def __init__(self, vocab: int) -> None:
        self.vocab = int(vocab)
        self.value_total = 0
        self.value_oob = 0
        self.row_total = 0
        self.row_oob = 0
        self.max_oob = None
        self.min_oob = None

    def update(self, arr: np.ndarray) -> None:
        if arr.size == 0 or self.vocab <= 0:
            return
        # arr 1D = (B,) single-value, 2D = (B, dim) multi-value
        if arr.ndim == 1:
            self.row_total += arr.shape[0]
            mask = arr >= self.vocab
            row_oob = mask  # (B,)
            self.row_oob += int(row_oob.sum())
            self.value_total += arr.shape[0]
            self.value_oob += int(mask.sum())
            if mask.any():
                vals = arr[mask]
                mn = int(vals.min())
                mx = int(vals.max())
                self.max_oob = mx if self.max_oob is None else max(self.max_oob, mx)
                self.min_oob = mn if self.min_oob is None else min(self.min_oob, mn)
        else:
            B = arr.shape[0]
            self.row_total += B
            mask = arr >= self.vocab
            row_oob = mask.any(axis=1)
            self.row_oob += int(row_oob.sum())
            # multi-value: count each non-padding cell
            non_pad_mask = arr > 0
            self.value_total += int(non_pad_mask.sum())
            self.value_oob += int((mask & non_pad_mask).sum())
            if mask.any():
                vals = arr[mask]
                if vals.size:
                    mn = int(vals.min())
                    mx = int(vals.max())
                    self.max_oob = mx if self.max_oob is None else max(self.max_oob, mx)
                    self.min_oob = mn if self.min_oob is None else min(self.min_oob, mn)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab": self.vocab,
            "value_total": self.value_total,
            "value_oob": self.value_oob,
            "value_oob_rate": (self.value_oob / self.value_total) if self.value_total else 0.0,
            "row_total": self.row_total,
            "row_oob": self.row_oob,
            "row_oob_rate": (self.row_oob / self.row_total) if self.row_total else 0.0,
            "max_oob": self.max_oob,
            "min_oob": self.min_oob,
        }


class LengthDist:
    """Track length distribution + truncation rate."""

    __slots__ = ("max_len", "lens_reservoir", "trunc_count", "empty_count",
                 "total", "max_observed", "sum_observed", "trunc_token_loss")

    def __init__(self, max_len: int) -> None:
        self.max_len = max_len
        self.lens_reservoir = ReservoirSample(size=RESERVOIR_SIZE)
        self.trunc_count = 0
        self.empty_count = 0
        self.total = 0
        self.max_observed = 0
        self.sum_observed = 0
        self.trunc_token_loss = 0  # tokens lost to truncation

    def update(self, lens: np.ndarray, real_lens: Optional[np.ndarray] = None) -> None:
        """``lens`` is the post-truncation effective length (≤ max_len).

        ``real_lens`` is the pre-truncation length, if available.
        """
        if lens.size == 0:
            return
        self.total += lens.size
        self.empty_count += int((lens == 0).sum())
        self.max_observed = max(self.max_observed, int(lens.max()))
        self.sum_observed += int(lens.sum())
        if real_lens is not None and real_lens.size:
            self.lens_reservoir.update_array(real_lens.astype(np.float64))
            trunc_mask = real_lens > self.max_len
            self.trunc_count += int(trunc_mask.sum())
            if trunc_mask.any():
                self.trunc_token_loss += int(
                    (real_lens[trunc_mask] - self.max_len).sum()
                )
        else:
            self.lens_reservoir.update_array(lens.astype(np.float64))

    def to_dict(self) -> Dict[str, Any]:
        pcts = self.lens_reservoir.percentiles(
            (0.5, 0.9, 0.95, 0.99, 0.999)
        )
        avg = (self.sum_observed / self.total) if self.total else 0.0
        return {
            "max_len_cap": self.max_len,
            "n": self.total,
            "max": self.max_observed,
            "mean": avg,
            **pcts,
            "trunc_count": self.trunc_count,
            "trunc_rate": (self.trunc_count / self.total) if self.total else 0.0,
            "trunc_token_loss": self.trunc_token_loss,
            "empty_count": self.empty_count,
            "empty_rate": (self.empty_count / self.total) if self.total else 0.0,
        }


# ============================================================================
# Schema / column wiring (ported from dataset.py without dataset bloat)
# ============================================================================


class FidPlan:
    """Plan for a single int feature: which parquet column, vocab, dim."""

    __slots__ = ("group", "fid", "col_idx", "vocab", "dim", "col_name")

    def __init__(self, group: str, fid: int, col_idx: int, vocab: int,
                 dim: int, col_name: str) -> None:
        self.group = group
        self.fid = fid
        self.col_idx = col_idx
        self.vocab = vocab
        self.dim = dim
        self.col_name = col_name


class SchemaPlan:
    """Mirror of dataset.py's schema parsing, but minimal and explicit."""

    def __init__(self, raw: Dict[str, Any]) -> None:
        self.user_int: List[FidPlan] = []
        self.item_int: List[FidPlan] = []
        # user_dense list of (fid, dim, col_name)
        self.user_dense: List[Tuple[int, int, str]] = []
        # seq[domain] = {prefix, ts_fid, sideinfo: List[(fid, vocab, col_name)]}
        self.seq: Dict[str, Dict[str, Any]] = {}

        for fid, vs, dim in raw.get("user_int", []):
            self.user_int.append(
                FidPlan("user_int", fid, -1, int(vs), int(dim),
                        f"user_int_feats_{fid}")
            )
        for fid, vs, dim in raw.get("item_int", []):
            self.item_int.append(
                FidPlan("item_int", fid, -1, int(vs), int(dim),
                        f"item_int_feats_{fid}")
            )
        for fid, dim in raw.get("user_dense", []):
            self.user_dense.append((int(fid), int(dim), f"user_dense_feats_{fid}"))

        for domain, cfg in raw.get("seq", {}).items():
            prefix = cfg["prefix"]
            ts_fid = cfg.get("ts_fid")
            features = cfg.get("features", [])
            sideinfo: List[Tuple[int, int, str]] = []
            for fid, vs in features:
                if ts_fid is not None and fid == ts_fid:
                    continue
                sideinfo.append((int(fid), int(vs), f"{prefix}_{fid}"))
            self.seq[domain] = {
                "prefix": prefix,
                "ts_fid": ts_fid,
                "ts_col_name": f"{prefix}_{ts_fid}" if ts_fid is not None else None,
                "sideinfo": sideinfo,
                "all_features": features,
            }

    def resolve_columns(self, col_idx: Dict[str, int]) -> List[str]:
        """Bind col_idx to each FidPlan; return list of missing column names."""
        missing: List[str] = []
        for plan in self.user_int + self.item_int:
            ci = col_idx.get(plan.col_name)
            if ci is None:
                missing.append(plan.col_name)
            else:
                plan.col_idx = ci
        return missing


# ============================================================================
# Per-fid stat aggregator (combines freq + OOV + multi-source updates)
# ============================================================================


class FidAgg:
    """Holds train freq stats and val OOV stats for a single fid."""

    __slots__ = ("plan", "freq", "oov_val", "is_multi_value")

    def __init__(self, plan: FidPlan) -> None:
        self.plan = plan
        self.freq = FreqStat(plan.vocab)
        self.oov_val = OOVTracker(plan.vocab)
        self.is_multi_value = plan.dim > 1


class SeqFidAgg:
    """Holds value-level freq + OOV for a (domain, fid) seq sideinfo."""

    __slots__ = ("domain", "fid", "vocab", "col_name", "freq", "oov_val")

    def __init__(self, domain: str, fid: int, vocab: int, col_name: str) -> None:
        self.domain = domain
        self.fid = fid
        self.vocab = vocab
        self.col_name = col_name
        self.freq = FreqStat(vocab)
        self.oov_val = OOVTracker(vocab)


# ============================================================================
# Parquet iteration
# ============================================================================


def _list_parquet_files(data_dir: str) -> List[str]:
    """Sorted *.parquet under ``data_dir`` (or itself if it's a file)."""
    if os.path.isfile(data_dir):
        return [data_dir]
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No .parquet files in {data_dir}")
    return files


def _gather_row_groups(
    files: List[str],
) -> Tuple[List[Tuple[str, int, int]], int]:
    """Return [(file, rg_idx, num_rows), ...] and total row count."""
    rg_list: List[Tuple[str, int, int]] = []
    total = 0
    for f in files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            n = pf.metadata.row_group(i).num_rows
            rg_list.append((f, i, n))
            total += n
    return rg_list, total


def _iter_record_batches(
    rg_list: List[Tuple[str, int, int]],
    batch_size: int = 4096,
) -> Iterator[Tuple[int, "pa.RecordBatch"]]:
    """Iterate ``(rg_idx, RecordBatch)`` over given row group slice."""
    last_file = None
    pf = None
    for rg_global_idx, (f, rg_idx, _) in enumerate(rg_list):
        if f != last_file:
            pf = pq.ParquetFile(f)
            last_file = f
        assert pf is not None
        for batch in pf.iter_batches(batch_size=batch_size, row_groups=[rg_idx]):
            yield rg_global_idx, batch


# ============================================================================
# Pyarrow column accessors (handle scalar vs list, fill_null, dtype)
# ============================================================================


def _to_int64_array(col: "pa.Array") -> np.ndarray:
    """Read a scalar int column as 1D int64 numpy array (null/-1 → 0)."""
    arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
    arr[arr <= 0] = 0
    return arr


def _list_int_to_padded(col: "pa.ListArray", max_len: int, B: int
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Pad an Arrow ListArray<int> to (B, max_len). Values <=0 → 0.

    Returns (padded, raw_lengths). raw_lengths is the pre-truncation length.
    """
    offsets = col.offsets.to_numpy()
    values = col.values.to_numpy()
    padded = np.zeros((B, max_len), dtype=np.int64)
    raw_lens = np.zeros(B, dtype=np.int64)
    for i in range(B):
        s, e = int(offsets[i]), int(offsets[i + 1])
        rl = e - s
        raw_lens[i] = rl
        if rl <= 0:
            continue
        ul = min(rl, max_len)
        padded[i, :ul] = values[s:s + ul]
    padded[padded < 0] = 0
    return padded, raw_lens


def _list_float_to_padded(col: "pa.ListArray", max_dim: int, B: int) -> np.ndarray:
    """Pad an Arrow ListArray<float> to (B, max_dim)."""
    offsets = col.offsets.to_numpy()
    values = col.values.to_numpy()
    padded = np.zeros((B, max_dim), dtype=np.float32)
    for i in range(B):
        s, e = int(offsets[i]), int(offsets[i + 1])
        rl = e - s
        if rl <= 0:
            continue
        ul = min(rl, max_dim)
        padded[i, :ul] = values[s:s + ul]
    return padded


# ============================================================================
# Main collector container
# ============================================================================


class ProfileCollectors:
    """Owns all per-section state. Updated by the main loop, then rendered."""

    def __init__(self, schema: SchemaPlan, seq_max_lens: Dict[str, int]) -> None:
        self.schema = schema
        self.seq_max_lens = seq_max_lens

        # ---- Section 1: dataset overview ----
        self.train_total_rows = 0
        self.val_total_rows = 0
        self.train_rg_rows: List[int] = []
        self.val_rg_rows: List[int] = []
        self.train_ts_min: Optional[int] = None
        self.train_ts_max: Optional[int] = None
        self.val_ts_min: Optional[int] = None
        self.val_ts_max: Optional[int] = None
        self.column_names: List[str] = []
        self.parquet_files: List[str] = []

        # ---- Section 2: label ----
        self.label_type_counts: Counter = Counter()
        self.train_pos_count = 0
        self.val_pos_count = 0
        self.train_pos_per_rg: List[Tuple[int, int]] = []  # (positives, total)
        self.val_pos_per_rg: List[Tuple[int, int]] = []
        # For time-bucketed pos rate: bucket by day
        self.train_pos_by_day: Dict[int, List[int]] = {}  # day -> [pos, total]
        self.val_pos_by_day: Dict[int, List[int]] = {}

        # ---- Section 3: user/item identity ----
        self.train_users: set = set()
        self.val_users: set = set()
        self.user_row_counts: Counter = Counter()  # user_id -> rows seen (train)
        self.user_id_size_cap = 5_000_000  # safety: stop tracking if exceeds

        # ---- Section 4: user_int / item_int per fid ----
        self.user_int_aggs: List[FidAgg] = [FidAgg(p) for p in schema.user_int]
        self.item_int_aggs: List[FidAgg] = [FidAgg(p) for p in schema.item_int]

        # ---- Section 5: user_dense per fid ----
        # per-fid dim-level Welford + reservoir + nan/inf counters
        self.dense_aggs: Dict[int, Dict[str, Any]] = {}
        for fid, dim, col_name in schema.user_dense:
            self.dense_aggs[fid] = {
                "fid": fid,
                "dim": dim,
                "col_name": col_name,
                "welford": [Welford() for _ in range(dim)],
                "reservoir": [ReservoirSample(size=RESERVOIR_SIZE) for _ in range(dim)],
                "nan_count": 0,
                "inf_count": 0,
                "all_zero_count": 0,
                "row_count": 0,
            }

        # ---- Section 6: seq sideinfo per (domain, fid) ----
        self.seq_aggs: Dict[str, List[SeqFidAgg]] = {}
        for domain, cfg in schema.seq.items():
            self.seq_aggs[domain] = [
                SeqFidAgg(domain, fid, vs, col_name)
                for fid, vs, col_name in cfg["sideinfo"]
            ]

        # ---- Section 7: seq length per domain ----
        self.train_lens: Dict[str, LengthDist] = {
            d: LengthDist(seq_max_lens.get(d, 256)) for d in schema.seq
        }
        self.val_lens: Dict[str, LengthDist] = {
            d: LengthDist(seq_max_lens.get(d, 256)) for d in schema.seq
        }

        # ---- Section 8: seq timestamp per domain ----
        # For domains with ts_fid: collect time-diff distribution + bucket use
        self.seq_ts_stats: Dict[str, Dict[str, Any]] = {}
        for domain, cfg in schema.seq.items():
            self.seq_ts_stats[domain] = {
                "has_ts": cfg["ts_fid"] is not None,
                "diff_reservoir": ReservoirSample(size=RESERVOIR_SIZE),
                "negative_diff_count": 0,
                "positive_diff_count": 0,
                "zero_diff_count": 0,
                "bucket_counts": np.zeros(NUM_TIME_BUCKETS, dtype=np.int64),
                "ts_min": None,
                "ts_max": None,
                "pos0_diff_reservoir": ReservoirSample(size=RESERVOIR_SIZE),
                "poslast_diff_reservoir": ReservoirSample(size=RESERVOIR_SIZE),
            }

        # ---- Section 13: schema health ----
        self.missing_columns: List[str] = []
        self.unused_columns: List[str] = []
        self.column_dtypes: Dict[str, str] = {}

    # ----- update entry points (called per batch) -----

    def feed_train_batch(self, batch: "pa.RecordBatch", rg_idx: int,
                         col_idx: Dict[str, int]) -> None:
        self._feed_batch(batch, rg_idx, col_idx, split="train")

    def feed_val_batch(self, batch: "pa.RecordBatch", rg_idx: int,
                       col_idx: Dict[str, int]) -> None:
        self._feed_batch(batch, rg_idx, col_idx, split="val")

    def _feed_batch(self, batch: "pa.RecordBatch", rg_idx: int,
                    col_idx: Dict[str, int], split: str) -> None:
        B = batch.num_rows
        if B == 0:
            return

        # Common columns
        ts_ci = col_idx.get("timestamp")
        lt_ci = col_idx.get("label_type")
        uid_ci = col_idx.get("user_id")

        if ts_ci is not None:
            ts = batch.column(ts_ci).to_numpy(zero_copy_only=False).astype(np.int64)
        else:
            ts = np.zeros(B, dtype=np.int64)
        if lt_ci is not None:
            lt_arr = (batch.column(lt_ci).fill_null(-1)
                      .to_numpy(zero_copy_only=False).astype(np.int64))
        else:
            lt_arr = np.full(B, -1, dtype=np.int64)
        labels = (lt_arr == 2).astype(np.int64)
        if uid_ci is not None:
            try:
                uids = batch.column(uid_ci).to_pylist()
            except Exception:
                uids = [None] * B
        else:
            uids = [None] * B

        # Section 1 + 2 stats
        if split == "train":
            self.train_total_rows += B
            for v, c in zip(*np.unique(lt_arr, return_counts=True)):
                self.label_type_counts[int(v)] += int(c)
            self.train_pos_count += int(labels.sum())
            self.train_pos_per_rg.append((int(labels.sum()), B))
            if ts.size:
                tmin, tmax = int(ts.min()), int(ts.max())
                self.train_ts_min = tmin if self.train_ts_min is None else min(self.train_ts_min, tmin)
                self.train_ts_max = tmax if self.train_ts_max is None else max(self.train_ts_max, tmax)
                # day bucket
                days = ts // 86400
                for d in np.unique(days):
                    mask = days == d
                    bin_pos = int(labels[mask].sum())
                    bin_n = int(mask.sum())
                    bucket = self.train_pos_by_day.setdefault(int(d), [0, 0])
                    bucket[0] += bin_pos
                    bucket[1] += bin_n
        else:
            self.val_total_rows += B
            self.val_pos_count += int(labels.sum())
            self.val_pos_per_rg.append((int(labels.sum()), B))
            if ts.size:
                tmin, tmax = int(ts.min()), int(ts.max())
                self.val_ts_min = tmin if self.val_ts_min is None else min(self.val_ts_min, tmin)
                self.val_ts_max = tmax if self.val_ts_max is None else max(self.val_ts_max, tmax)
                days = ts // 86400
                for d in np.unique(days):
                    mask = days == d
                    bucket = self.val_pos_by_day.setdefault(int(d), [0, 0])
                    bucket[0] += int(labels[mask].sum())
                    bucket[1] += int(mask.sum())

        # Section 3: user identity (train only — val overlap done at finalize)
        if split == "train":
            for u in uids:
                if u is None:
                    continue
                if len(self.train_users) < self.user_id_size_cap:
                    self.train_users.add(u)
                self.user_row_counts[u] += 1
        else:
            for u in uids:
                if u is None:
                    continue
                if len(self.val_users) < self.user_id_size_cap:
                    self.val_users.add(u)

        # Section 4: user_int / item_int
        for agg in self.user_int_aggs + self.item_int_aggs:
            if agg.plan.col_idx < 0:
                continue
            col = batch.column(agg.plan.col_idx)
            if agg.plan.dim == 1:
                arr = _to_int64_array(col)
                if split == "train":
                    agg.freq.update_row_batch(arr)
                else:
                    agg.oov_val.update(arr)
            else:
                padded, _ = _list_int_to_padded(col, agg.plan.dim, B)
                if split == "train":
                    agg.freq.update_row_batch(padded)
                else:
                    agg.oov_val.update(padded)

        # Section 5: user_dense (train only — dense val stats not in scope)
        if split == "train":
            for fid, dim, col_name in self.schema.user_dense:
                ci = col_idx.get(col_name)
                if ci is None:
                    continue
                col = batch.column(ci)
                padded = _list_float_to_padded(col, dim, B)
                bucket = self.dense_aggs[fid]
                bucket["row_count"] += B
                # NaN / Inf
                nan_mask = np.isnan(padded)
                inf_mask = np.isinf(padded)
                bucket["nan_count"] += int(nan_mask.sum())
                bucket["inf_count"] += int(inf_mask.sum())
                # All-zero rows (across all dims)
                bucket["all_zero_count"] += int((padded == 0).all(axis=1).sum())
                # Per-dim Welford + reservoir
                for d in range(dim):
                    col_arr = padded[:, d]
                    bucket["welford"][d].update_array(col_arr)
                    bucket["reservoir"][d].update_array(col_arr)

        # Section 6: seq sideinfo
        for domain, aggs in self.seq_aggs.items():
            max_len = self.seq_max_lens.get(domain, 256)
            for agg in aggs:
                ci = col_idx.get(agg.col_name)
                if ci is None:
                    continue
                col = batch.column(ci)
                padded, raw_lens = _list_int_to_padded(col, max_len, B)
                if split == "train":
                    agg.freq.update_row_batch(padded)
                else:
                    agg.oov_val.update(padded)

        # Section 7: seq length (use first sideinfo's raw_lens as canonical)
        for domain, aggs in self.seq_aggs.items():
            if not aggs:
                continue
            max_len = self.seq_max_lens.get(domain, 256)
            ci = col_idx.get(aggs[0].col_name)
            if ci is None:
                continue
            col = batch.column(ci)
            _, raw_lens = _list_int_to_padded(col, max_len, B)
            eff_lens = np.minimum(raw_lens, max_len)
            ld = self.train_lens[domain] if split == "train" else self.val_lens[domain]
            ld.update(eff_lens, raw_lens)

        # Section 8: seq timestamp (train only)
        if split == "train":
            for domain, cfg in self.schema.seq.items():
                stats = self.seq_ts_stats[domain]
                if not stats["has_ts"]:
                    continue
                ts_col_name = cfg["ts_col_name"]
                ts_ci_local = col_idx.get(ts_col_name)
                if ts_ci_local is None:
                    continue
                max_len = self.seq_max_lens.get(domain, 256)
                ts_col = batch.column(ts_ci_local)
                ts_padded, ts_raw_lens = _list_int_to_padded(ts_col, max_len, B)
                # ts_padded shape (B, max_len). Compute (row_ts - seq_ts).
                row_ts_2d = ts.reshape(-1, 1)
                diff = row_ts_2d - ts_padded
                # Mask: only count valid (non-padding) seq ts positions.
                valid = ts_padded > 0
                if not valid.any():
                    continue
                diff_valid = diff[valid]
                if stats["ts_min"] is None or ts_padded[valid].min() < stats["ts_min"]:
                    stats["ts_min"] = int(ts_padded[valid].min())
                if stats["ts_max"] is None or ts_padded[valid].max() > stats["ts_max"]:
                    stats["ts_max"] = int(ts_padded[valid].max())
                stats["positive_diff_count"] += int((diff_valid > 0).sum())
                stats["negative_diff_count"] += int((diff_valid < 0).sum())
                stats["zero_diff_count"] += int((diff_valid == 0).sum())
                stats["diff_reservoir"].update_array(diff_valid.astype(np.float64))
                # Bucket use (matches dataset.py logic)
                clipped = np.clip(
                    np.searchsorted(BUCKET_BOUNDARIES, np.maximum(diff_valid, 0)),
                    0, len(BUCKET_BOUNDARIES) - 1,
                )
                buckets = clipped + 1
                for b, c in zip(*np.unique(buckets, return_counts=True)):
                    if 0 <= int(b) < NUM_TIME_BUCKETS:
                        stats["bucket_counts"][int(b)] += int(c)
                # pos 0 vs pos last diff (verify "pos 0 = recent" assumption)
                pos0_valid = (ts_padded[:, 0] > 0)
                if pos0_valid.any():
                    stats["pos0_diff_reservoir"].update_array(
                        diff[:, 0][pos0_valid].astype(np.float64)
                    )
                # last position per row (use raw_lens to find last non-padding)
                last_idx = np.clip(ts_raw_lens - 1, 0, max_len - 1)
                last_ts = ts_padded[np.arange(B), last_idx]
                last_valid = last_ts > 0
                if last_valid.any():
                    last_diff = (ts[last_valid] - last_ts[last_valid]).astype(np.float64)
                    stats["poslast_diff_reservoir"].update_array(last_diff)


# ============================================================================
# Report rendering (markdown to list of strings, capped to ≤1000 lines)
# ============================================================================


def _fmt_pct(x: float) -> str:
    return f"{x*100:.3f}%"


def _fmt_int(x: int) -> str:
    return f"{x:,}"


def _fmt_float(x: float, prec: int = 4) -> str:
    if x is None:
        return "—"
    if abs(x) >= 1e6 or (x != 0 and abs(x) < 1e-3):
        return f"{x:.{prec}e}"
    return f"{x:.{prec}f}"


def _topk_curve(freq: FreqStat, ks: Iterable[int]) -> List[str]:
    out = []
    for k, cov in freq.topk_coverage(list(ks)):
        if k >= 1_000_000:
            label = f"{k // 1_000_000}M"
        elif k >= 1_000:
            label = f"{k // 1_000}k"
        else:
            label = str(k)
        out.append(f"{label}={_fmt_pct(cov)}")
    return out


def render_report(c: ProfileCollectors, args: argparse.Namespace,
                  raw_dump: Dict[str, Any]) -> List[str]:
    """Build markdown lines list (~675 budget). One section at a time."""
    lines: List[str] = []
    L = lines.append
    SEP = "---"

    # ===== Header =====
    L(f"# TAAC Profile Report")
    L(f"")
    L(f"- generated_at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    L(f"- data_dir: `{args.data_dir}`")
    L(f"- schema: `{args.schema_path}`")
    L(f"- valid_ratio: {args.valid_ratio}")
    L(f"- seq_max_lens: {args.seq_max_lens}")
    L("")
    L(SEP)

    # ===== Decision summary (top of report for skim) =====
    L("")
    L("## Decision Summary")
    L("")
    _render_decision_summary(c, L)
    L("")
    L(SEP)

    # ===== Section 1 =====
    L("")
    L("## Section 1: Dataset Overview")
    L("")
    L(f"- parquet_files: {len(c.parquet_files)}")
    L(f"- column_count: {len(c.column_names)}")
    L(f"- train_rows: {_fmt_int(c.train_total_rows)} | val_rows: {_fmt_int(c.val_total_rows)}")
    L(f"- train_row_groups: {len(c.train_rg_rows)} | val_row_groups: {len(c.val_rg_rows)}")
    if c.train_rg_rows:
        rg = np.asarray(c.train_rg_rows)
        L(f"- train_rg_rows: min={rg.min()}, p50={int(np.percentile(rg, 50))}, "
          f"max={rg.max()}, mean={rg.mean():.0f}")
    if c.train_ts_min is not None:
        L(f"- train_ts_range: [{c.train_ts_min}, {c.train_ts_max}] "
          f"(span ≈ {(c.train_ts_max - c.train_ts_min) / 86400:.1f} days)")
    if c.val_ts_min is not None:
        L(f"- val_ts_range: [{c.val_ts_min}, {c.val_ts_max}] "
          f"(span ≈ {(c.val_ts_max - c.val_ts_min) / 86400:.1f} days)")
        if c.train_ts_max is not None:
            gap = c.val_ts_min - c.train_ts_max
            L(f"- train_tail vs val_head gap: {gap} sec ({gap/86400:.2f} days) "
              f"({'forward in time ✓' if gap > 0 else 'overlapping (RG split ≈ random) ⚠'})")
    L("")
    L(SEP)

    # ===== Section 2 =====
    L("")
    L("## Section 2: Label Distribution")
    L("")
    L("### label_type counts (all splits)")
    L("| label_type | count | share |")
    L("|---|---|---|")
    total_lt = sum(c.label_type_counts.values())
    for v in sorted(c.label_type_counts.keys()):
        cnt = c.label_type_counts[v]
        L(f"| {v} | {_fmt_int(cnt)} | {_fmt_pct(cnt / max(total_lt, 1))} |")
    L("")
    train_pr = c.train_pos_count / max(c.train_total_rows, 1)
    val_pr = c.val_pos_count / max(c.val_total_rows, 1)
    L(f"- train pos rate (label_type==2): {_fmt_pct(train_pr)} "
      f"({_fmt_int(c.train_pos_count)}/{_fmt_int(c.train_total_rows)})")
    L(f"- val pos rate: {_fmt_pct(val_pr)} "
      f"({_fmt_int(c.val_pos_count)}/{_fmt_int(c.val_total_rows)})")
    if c.train_total_rows and c.val_total_rows:
        delta = val_pr - train_pr
        L(f"- val - train pos rate Δ: {delta*100:+.3f}%")

    # Pos rate per row group (compact)
    if c.train_pos_per_rg:
        rates = [p / max(n, 1) for p, n in c.train_pos_per_rg if n > 0]
        if rates:
            arr = np.asarray(rates)
            L(f"- train per-RG pos rate: min={_fmt_pct(arr.min())}, "
              f"p50={_fmt_pct(np.percentile(arr,50))}, "
              f"max={_fmt_pct(arr.max())}, std={_fmt_pct(arr.std())}")
    if c.val_pos_per_rg:
        rates = [p / max(n, 1) for p, n in c.val_pos_per_rg if n > 0]
        if rates:
            arr = np.asarray(rates)
            L(f"- val per-RG pos rate: min={_fmt_pct(arr.min())}, "
              f"p50={_fmt_pct(np.percentile(arr,50))}, "
              f"max={_fmt_pct(arr.max())}, std={_fmt_pct(arr.std())}")

    # Pos rate by day (only show days with ≥0.1% of total rows)
    if c.train_pos_by_day:
        L("")
        L("### Pos rate by day (train, days ≥0.1% rows)")
        L("| day_id | n | pos_rate |")
        L("|---|---|---|")
        for d in sorted(c.train_pos_by_day.keys()):
            pos, n = c.train_pos_by_day[d]
            if n / max(c.train_total_rows, 1) < 0.001:
                continue
            L(f"| {d} | {_fmt_int(n)} | {_fmt_pct(pos / max(n, 1))} |")
    L("")
    L(SEP)

    # ===== Section 3 =====
    L("")
    L("## Section 3: User & Item Identity")
    L("")
    L(f"- train unique user_ids (sampled, cap={c.user_id_size_cap}): "
      f"{_fmt_int(len(c.train_users))}")
    L(f"- val unique user_ids: {_fmt_int(len(c.val_users))}")
    if c.train_users and c.val_users:
        overlap = c.train_users & c.val_users
        L(f"- user overlap (train ∩ val): {_fmt_int(len(overlap))}")
        L(f"- val users in train: {_fmt_pct(len(overlap) / max(len(c.val_users), 1))}")
    if c.user_row_counts:
        counts = np.asarray(list(c.user_row_counts.values()))
        L(f"- rows-per-user (train): n={len(counts)}, "
          f"min={counts.min()}, p50={int(np.percentile(counts, 50))}, "
          f"p90={int(np.percentile(counts, 90))}, "
          f"p99={int(np.percentile(counts, 99))}, max={counts.max()}")
    if c.train_ts_min is not None:
        # Heuristic timestamp precision check
        L(f"- timestamp precision: probably **second-level** "
          f"(values ~{c.train_ts_min:.2e})")
    L("")
    L(SEP)

    # ===== Section 4 =====
    L("")
    L("## Section 4: User_int / Item_int per fid")
    L("")
    L("Columns: `vocab_obs/vocab_schema | unique | non_zero_rate | "
      "top1k_cov | top10k_cov | top100k_cov | gini | val_oov_rate`")
    L("")
    L("### user_int")
    L("| fid | dim | vocab(s/o) | unique | nz_rate | top1k | top10k | top100k | gini | val_oov |")
    L("|---|---|---|---|---|---|---|---|---|---|")
    for agg in c.user_int_aggs:
        L(_fid_row(agg))
    L("")
    L("### item_int")
    L("| fid | dim | vocab(s/o) | unique | nz_rate | top1k | top10k | top100k | gini | val_oov |")
    L("|---|---|---|---|---|---|---|---|---|---|")
    for agg in c.item_int_aggs:
        L(_fid_row(agg))
    L("")
    L(SEP)

    # ===== Section 5 =====
    L("")
    L("## Section 5: User_dense per fid")
    L("")
    L("Per-fid summary (per-dim aggregated to fid level for compactness):")
    L("| fid | dim | n | mean(avg) | std(avg) | min | max | p50(avg) | p99(avg) | nan_rate | inf_rate | all_zero_rate |")
    L("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for fid, dim, col_name in c.schema.user_dense:
        bucket = c.dense_aggs[fid]
        means, stds, mins, maxs, p50s, p99s = [], [], [], [], [], []
        ns = []
        for d in range(dim):
            wd = bucket["welford"][d].to_dict()
            ns.append(wd["n"])
            means.append(wd["mean"])
            stds.append(wd["std"])
            mins.append(wd["min"])
            maxs.append(wd["max"])
            ps = bucket["reservoir"][d].percentiles((0.5, 0.99))
            p50s.append(ps.get("p50.0") or 0.0)
            p99s.append(ps.get("p99.0") or 0.0)
        if not ns:
            continue
        n_total = sum(ns)
        cells_total = bucket["row_count"] * dim
        nan_rate = bucket["nan_count"] / max(cells_total, 1)
        inf_rate = bucket["inf_count"] / max(cells_total, 1)
        az_rate = bucket["all_zero_count"] / max(bucket["row_count"], 1)
        L(f"| {fid} | {dim} | {_fmt_int(n_total)} | "
          f"{_fmt_float(np.mean(means))} | {_fmt_float(np.mean(stds))} | "
          f"{_fmt_float(min(mins))} | {_fmt_float(max(maxs))} | "
          f"{_fmt_float(np.mean(p50s))} | {_fmt_float(np.mean(p99s))} | "
          f"{_fmt_pct(nan_rate)} | {_fmt_pct(inf_rate)} | {_fmt_pct(az_rate)} |")
    L("")
    L(SEP)

    # ===== Section 6 =====
    L("")
    L("## Section 6: Seq sideinfo per (domain, fid)")
    L("")
    EMB_SKIP_FIDS = {29, 34, 47, 69}
    for domain in sorted(c.seq_aggs.keys()):
        L(f"### {domain}")
        L("| fid | vocab(s/o) | unique | top1k | top10k | top100k | gini | val_oov |")
        L("|---|---|---|---|---|---|---|---|")
        for sa in c.seq_aggs[domain]:
            row = _seq_fid_row(sa)
            L(row)
        L("")

    # Detailed treatment of emb_skip-skipped fids
    L("### emb_skip-skipped fids (detailed top-K curves)")
    L("")
    L("| fid | domain | vocab | total_obs | top100 | top1k | top10k | top100k | top1M | top10M |")
    L("|---|---|---|---|---|---|---|---|---|---|")
    for domain, aggs in c.seq_aggs.items():
        for sa in aggs:
            if sa.fid not in EMB_SKIP_FIDS:
                continue
            cur = sa.freq.topk_coverage(TOPK_REPORT)
            cells = []
            for k, cov in cur:
                if k >= 100_000_000:
                    break
                cells.append(_fmt_pct(cov))
            # Pad to 6 cells (100, 1k, 10k, 100k, 1M, 10M)
            while len(cells) < 6:
                cells.append("—")
            L(f"| {sa.fid} | {domain} | {_fmt_int(sa.vocab)} | "
              f"{_fmt_int(sa.freq.total)} | {' | '.join(cells[:6])} |")
    L("")
    L(SEP)

    # ===== Section 7 =====
    L("")
    L("## Section 7: Sequence length per domain")
    L("")
    L("| domain | split | n | max_cap | p50 | p90 | p99 | p99.9 | max | trunc_rate | trunc_tokens | empty_rate |")
    L("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for domain in sorted(c.train_lens.keys()):
        for split, ld in (("train", c.train_lens[domain]),
                          ("val", c.val_lens[domain])):
            d = ld.to_dict()
            L(f"| {domain} | {split} | {_fmt_int(d['n'])} | {d['max_len_cap']} | "
              f"{_fmt_float(d.get('p50.0'))} | {_fmt_float(d.get('p90.0'))} | "
              f"{_fmt_float(d.get('p99.0'))} | {_fmt_float(d.get('p99.9'))} | "
              f"{d['max']} | {_fmt_pct(d['trunc_rate'])} | "
              f"{_fmt_int(d['trunc_token_loss'])} | {_fmt_pct(d['empty_rate'])} |")
    L("")
    L(SEP)

    # ===== Section 8 =====
    L("")
    L("## Section 8: Sequence timestamp per domain")
    L("")
    for domain in sorted(c.seq_ts_stats.keys()):
        s = c.seq_ts_stats[domain]
        if not s["has_ts"]:
            L(f"### {domain}")
            L(f"- ts_fid is None (no timestamp column for this domain)")
            L("")
            continue
        L(f"### {domain}")
        if s["ts_min"] is not None:
            L(f"- ts_range: [{s['ts_min']}, {s['ts_max']}] "
              f"(span ≈ {(s['ts_max']-s['ts_min'])/86400:.1f} days)")
        tot = s["positive_diff_count"] + s["negative_diff_count"] + s["zero_diff_count"]
        if tot:
            L(f"- diff (row_ts - seq_ts): pos={_fmt_pct(s['positive_diff_count']/tot)}, "
              f"neg={_fmt_pct(s['negative_diff_count']/tot)}, "
              f"zero={_fmt_pct(s['zero_diff_count']/tot)}")
        ps = s["diff_reservoir"].percentiles((0.1, 0.5, 0.9, 0.99))
        L(f"- diff percentiles: p10={_fmt_float(ps.get('p10.0'))}, "
          f"p50={_fmt_float(ps.get('p50.0'))}, p90={_fmt_float(ps.get('p90.0'))}, "
          f"p99={_fmt_float(ps.get('p99.0'))}")
        # pos 0 vs pos last
        p0 = s["pos0_diff_reservoir"].percentiles((0.5,))
        pl = s["poslast_diff_reservoir"].percentiles((0.5,))
        L(f"- pos[0] median diff: {_fmt_float(p0.get('p50.0'))}, "
          f"pos[last] median diff: {_fmt_float(pl.get('p50.0'))} "
          f"({'pos 0 = recent ✓' if (p0.get('p50.0') or 0) < (pl.get('p50.0') or 0) else 'pos 0 = OLD ⚠'})")
        # Bucket histogram
        bcounts = s["bucket_counts"]
        btot = bcounts.sum()
        if btot > 0:
            top_buckets = np.argsort(-bcounts)[:5]
            top_str = ", ".join(
                f"b{int(b)}={_fmt_pct(bcounts[b]/btot)}" for b in top_buckets
            )
            empty_buckets = int((bcounts == 0).sum())
            L(f"- bucket use top-5: {top_str} | empty_buckets: {empty_buckets}/{NUM_TIME_BUCKETS}")
        L("")
    L(SEP)

    # ===== Section 9 =====
    L("")
    L("## Section 9: Direction-1 (UNK / OOV) decision table")
    L("")
    L("| group | fid | vocab | unique | val_value_oov | val_row_oob | k@99% cov | recommendation |")
    L("|---|---|---|---|---|---|---|---|")
    rows: List[Tuple[float, str]] = []
    for agg in c.user_int_aggs + c.item_int_aggs:
        rows.append(_oov_decision_row(agg.plan.group, agg.plan.fid, agg.freq, agg.oov_val))
    for domain, aggs in c.seq_aggs.items():
        for sa in aggs:
            rows.append(_oov_decision_row(f"seq_{domain}", sa.fid, sa.freq, sa.oov_val))
    rows.sort(key=lambda r: -r[0])
    for _, r in rows[:80]:  # cap at top-80 to fit budget
        L(r)
    if len(rows) > 80:
        L(f"")
        L(f"_({len(rows)-80} more fids omitted; OOV rate < top 80)_")
    L("")
    L(SEP)

    # ===== Section 10 =====
    L("")
    L("## Section 10: Direction-2 (emb_skip revival) decision table")
    L("")
    EMB_SKIP_FIDS = {29, 34, 47, 69}
    target_aggs: List[SeqFidAgg] = []
    for domain, aggs in c.seq_aggs.items():
        for sa in aggs:
            if sa.fid in EMB_SKIP_FIDS:
                target_aggs.append(sa)
    if not target_aggs:
        L("_(no emb_skip fids found in this schema)_")
    else:
        L("| fid | domain | schema_vocab | total_obs | k@95% | k@99% | k@99.9% | long_tail<5 | rec_bucket | added_params(@d=64) |")
        L("|---|---|---|---|---|---|---|---|---|---|")
        for sa in target_aggs:
            k95 = sa.freq.k_for_coverage(0.95)
            k99 = sa.freq.k_for_coverage(0.99)
            k999 = sa.freq.k_for_coverage(0.999)
            lt = sa.freq.long_tail_ratio(5)
            rec_bucket = k99 or 0
            added_params = rec_bucket * EMB_DIM_DEFAULT * 4 / (1024 * 1024)  # MB
            L(f"| {sa.fid} | {sa.domain} | {_fmt_int(sa.vocab)} | "
              f"{_fmt_int(sa.freq.total)} | {k95 or '—'} | {k99 or '—'} | "
              f"{k999 or '—'} | {_fmt_pct(lt) if lt is not None else '—'} | "
              f"{_fmt_int(rec_bucket)} | {added_params:.1f}MB |")
        L("")
        L("**Recommendations**:")
        for sa in target_aggs:
            k99 = sa.freq.k_for_coverage(0.99)
            if k99 is None or sa.freq.total == 0:
                L(f"- fid {sa.fid} ({sa.domain}): no signal observed; **skip** revival")
                continue
            ratio = k99 / max(sa.vocab, 1)
            if ratio < 0.01:
                L(f"- fid {sa.fid}: top-{_fmt_int(k99)} covers 99% (only {ratio*100:.2f}% of vocab); "
                  f"**hash trick** with bucket={_fmt_int(k99)}")
            elif ratio < 0.1:
                L(f"- fid {sa.fid}: top-{_fmt_int(k99)} covers 99% ({ratio*100:.1f}% of vocab); "
                  f"**freq truncate** + UNK pooling")
            else:
                L(f"- fid {sa.fid}: top-K covers 99% needs {ratio*100:.1f}% of vocab; "
                  f"**raise emb_skip threshold** to {_fmt_int(sa.vocab + 1)} or skip")
    L("")
    L(SEP)

    # ===== Section 11 =====
    L("")
    L("## Section 11: Direction-3 (long sequence) decision table")
    L("")
    L("| domain | max_cap | p99 | p99.9 | trunc_rate | tokens_lost | recommendation |")
    L("|---|---|---|---|---|---|---|")
    for domain in sorted(c.train_lens.keys()):
        ld = c.train_lens[domain].to_dict()
        p99 = ld.get("p99.0") or 0
        p999 = ld.get("p99.9") or 0
        rec = "no-op (cap above p99)"
        if ld["trunc_rate"] > 0.01:
            target = int(math.ceil(p99 / 64) * 64)
            rec = f"raise cap to {target} (depends on W1.0.3 fix)"
        L(f"| {domain} | {ld['max_len_cap']} | {_fmt_float(p99)} | {_fmt_float(p999)} | "
          f"{_fmt_pct(ld['trunc_rate'])} | {_fmt_int(ld['trunc_token_loss'])} | {rec} |")
    L("")
    L("_Note: revival of long-tail tokens requires fixing LongerEncoder direction (W1.0.3) "
      "before W1.7 yields valid signal._")
    L("")
    L(SEP)

    # ===== Section 12 =====
    L("")
    L("## Section 12: Direction-4 (per-feature emb_dim) decision table")
    L("")
    # Vocab histogram
    all_aggs: List[Tuple[str, int, int, int]] = []  # (group, fid, vocab, unique)
    for agg in c.user_int_aggs + c.item_int_aggs:
        u = agg.freq.num_unique() or 0
        all_aggs.append((agg.plan.group, agg.plan.fid, agg.plan.vocab, u))
    for domain, aggs in c.seq_aggs.items():
        for sa in aggs:
            u = sa.freq.num_unique() or 0
            all_aggs.append((f"seq_{domain}", sa.fid, sa.vocab, u))

    buckets = {"≤100": 0, "101-1k": 0, "1k-10k": 0, "10k-100k": 0,
               "100k-1M": 0, ">1M": 0}
    for _, _, v, _ in all_aggs:
        if v <= 100: buckets["≤100"] += 1
        elif v <= 1000: buckets["101-1k"] += 1
        elif v <= 10_000: buckets["1k-10k"] += 1
        elif v <= 100_000: buckets["10k-100k"] += 1
        elif v <= 1_000_000: buckets["100k-1M"] += 1
        else: buckets[">1M"] += 1
    L("### vocab distribution across all fids")
    L("| range | count | recommended emb_dim | current emb_dim |")
    L("|---|---|---|---|")
    rec_dims = {"≤100": 8, "101-1k": 16, "1k-10k": 32, "10k-100k": 64,
                "100k-1M": 64, ">1M": 64}
    for r, n in buckets.items():
        L(f"| {r} | {n} | {rec_dims[r]} | {EMB_DIM_DEFAULT} |")
    # Param savings estimate
    cur_params = sum(v * EMB_DIM_DEFAULT for _, _, v, _ in all_aggs if v > 0)
    rec_total = 0
    for _, _, v, _ in all_aggs:
        if v <= 0: continue
        if v <= 100: rec_total += v * 8
        elif v <= 1000: rec_total += v * 16
        elif v <= 10_000: rec_total += v * 32
        else: rec_total += v * 64
    saving = (cur_params - rec_total) / max(cur_params, 1)
    L("")
    L(f"- current total embedding params: {_fmt_int(cur_params)}")
    L(f"- recommended total: {_fmt_int(rec_total)}")
    L(f"- saving: {_fmt_pct(saving)}")
    L("")
    L(SEP)

    # ===== Section 13 =====
    L("")
    L("## Section 13: Schema Health Check")
    L("")
    L(f"- missing columns (in schema but absent in parquet): {len(c.missing_columns)}")
    if c.missing_columns[:10]:
        L(f"  - first 10: {c.missing_columns[:10]}")
    L(f"- unused columns (in parquet but absent in schema): {len(c.unused_columns)}")
    if c.unused_columns[:10]:
        L(f"  - first 10: {c.unused_columns[:10]}")
    # Vocab sanity: schema vs observed
    drift_rows = []
    for agg in c.user_int_aggs + c.item_int_aggs:
        if agg.freq.obs_max is None:
            continue
        if agg.freq.obs_max >= agg.plan.vocab:
            drift_rows.append((agg.plan.group, agg.plan.fid, agg.plan.vocab, agg.freq.obs_max))
    if drift_rows:
        L("")
        L("### schema vocab vs observed_max drift (train pass)")
        L("| group | fid | schema_vocab | observed_max |")
        L("|---|---|---|---|")
        for g, f, v, om in drift_rows[:20]:
            L(f"| {g} | {f} | {v} | {om} |")
    L("")
    L(SEP)
    L("")
    L(f"_End of report. Total lines: tracked at flush time._")

    return lines


def _fid_row(agg: FidAgg) -> str:
    f = agg.freq
    cov = dict(f.topk_coverage([1000, 10_000, 100_000]))
    unique = f.num_unique()
    nz_rate = f.row_with_value_count / max(f.row_count, 1)
    oov = agg.oov_val.to_dict()
    return (f"| {agg.plan.fid} | {agg.plan.dim} | "
            f"{_fmt_int(agg.plan.vocab)}/{f.obs_max or 0} | "
            f"{_fmt_int(unique) if unique is not None else '~MG'} | "
            f"{_fmt_pct(nz_rate)} | {_fmt_pct(cov.get(1000, 0))} | "
            f"{_fmt_pct(cov.get(10_000, 0))} | {_fmt_pct(cov.get(100_000, 0))} | "
            f"{_fmt_float(f.gini())} | {_fmt_pct(oov['value_oob_rate'])} |")


def _seq_fid_row(sa: SeqFidAgg) -> str:
    f = sa.freq
    cov = dict(f.topk_coverage([1000, 10_000, 100_000]))
    unique = f.num_unique()
    oov = sa.oov_val.to_dict()
    return (f"| {sa.fid} | {_fmt_int(sa.vocab)}/{f.obs_max or 0} | "
            f"{_fmt_int(unique) if unique is not None else '~MG'} | "
            f"{_fmt_pct(cov.get(1000, 0))} | {_fmt_pct(cov.get(10_000, 0))} | "
            f"{_fmt_pct(cov.get(100_000, 0))} | {_fmt_float(f.gini())} | "
            f"{_fmt_pct(oov['value_oob_rate'])} |")


def _oov_decision_row(
    group: str, fid: int, freq: FreqStat, oov: OOVTracker,
) -> Tuple[float, str]:
    od = oov.to_dict()
    rate = od["value_oob_rate"]
    k99 = freq.k_for_coverage(0.99)
    if rate < OOV_NOOP_RATE:
        rec = "no-op"
    elif rate < OOV_LOW_RATE:
        rec = "+1 UNK row + freq-truncate"
    else:
        rec = "multi-bucket hash UNK"
    line = (f"| {group} | {fid} | {_fmt_int(od['vocab'])} | "
            f"{_fmt_int(freq.num_unique() or 0)} | "
            f"{_fmt_pct(rate)} | {_fmt_pct(od['row_oob_rate'])} | "
            f"{_fmt_int(k99) if k99 else '—'} | {rec} |")
    return rate, line


def _render_decision_summary(c: ProfileCollectors, L) -> None:
    """Top-level skim of recommendations across 4 directions."""
    # Direction 1: UNK
    high_oov = [(a.plan.fid, a.oov_val.to_dict()["value_oob_rate"])
                for a in c.user_int_aggs + c.item_int_aggs
                if a.oov_val.to_dict()["value_oob_rate"] >= OOV_LOW_RATE]
    L(f"**Direction 1 (UNK / OOV)**: "
      f"{len(high_oov)} fid(s) have val OOV ≥ 1% (multi-bucket hash UNK warranted).")

    # Direction 2: emb_skip revival
    EMB_SKIP_FIDS = {29, 34, 47, 69}
    skip_signals = []
    for domain, aggs in c.seq_aggs.items():
        for sa in aggs:
            if sa.fid in EMB_SKIP_FIDS:
                k99 = sa.freq.k_for_coverage(0.99)
                if k99:
                    skip_signals.append((sa.fid, sa.domain, k99,
                                         sa.vocab))
    if skip_signals:
        L(f"**Direction 2 (emb_skip revival)**: "
          f"{len(skip_signals)}/4 emb_skip fids have measurable signal. "
          f"Smallest k@99%: {min(k for _, _, k, _ in skip_signals)}, "
          f"largest: {max(k for _, _, k, _ in skip_signals)}.")
    else:
        L("**Direction 2 (emb_skip revival)**: no emb_skip fids in schema "
          "(or none observed in train) — skip this direction.")

    # Direction 3: long sequence
    needs_lift = []
    for domain, ld in c.train_lens.items():
        d = ld.to_dict()
        if d["trunc_rate"] > 0.01:
            needs_lift.append((domain, d["trunc_rate"], d["trunc_token_loss"]))
    if needs_lift:
        L(f"**Direction 3 (long sequence)**: "
          f"{len(needs_lift)} domain(s) have >1% truncation rate. "
          f"Worst: {max(needs_lift, key=lambda x: x[1])[0]}.")
    else:
        L("**Direction 3 (long sequence)**: all domains have <1% truncation. "
          "Low priority unless W1.0.3 fix unlocks something.")

    # Direction 4: emb_dim
    L("**Direction 4 (per-fid emb_dim)**: see Section 12 for vocab histogram + "
      "param savings estimate. Recommend deferring unless after H1/H2 wins.")


# ============================================================================
# Main
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TAAC data profile (read-only)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="parquet dir (env: TRAIN_DATA_PATH)")
    parser.add_argument("--schema_path", type=str, default=None,
                        help="schema.json path (default: <data_dir>/schema.json)")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="optional log dir (env: TRAIN_LOG_PATH); "
                             "if writable, dumps profile_data.json there too")
    parser.add_argument("--valid_ratio", type=float, default=0.1,
                        help="tail fraction of row groups used as val")
    parser.add_argument("--train_ratio", type=float, default=1.0,
                        help="head fraction of training row groups to use "
                             "(useful for quick sanity test on partial data)")
    parser.add_argument("--seq_max_lens", type=str,
                        default="seq_a:256,seq_b:256,seq_c:512,seq_d:512",
                        help="per-domain seq cap, e.g. 'seq_a:256,seq_b:256'")
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="parquet record batch size for reading")
    parser.add_argument("--max_train_rows", type=int, default=0,
                        help="if >0, stop train pass after this many rows "
                             "(debugging only)")
    args = parser.parse_args()

    args.data_dir = os.environ.get("TRAIN_DATA_PATH", args.data_dir)
    args.log_dir = os.environ.get("TRAIN_LOG_PATH", args.log_dir)
    if not args.data_dir:
        raise SystemExit("TRAIN_DATA_PATH or --data_dir is required")
    if not args.schema_path:
        args.schema_path = os.path.join(args.data_dir, "schema.json")
    return args


def setup_logging(log_dir: Optional[str]) -> None:
    """Send INFO+ to stderr, WARNING+ to console; never to stdout."""
    root = logging.getLogger()
    root.handlers = []
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s",
                            datefmt="%H:%M:%S")
    # stderr at INFO so we see progress without polluting stdout.
    sh = logging.StreamHandler(stream=sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(log_dir, "profile.log"), "w")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except OSError as e:
            sh.handleError(logging.LogRecord(
                "profile", logging.WARNING, "", 0,
                f"can't write log_dir={log_dir}: {e}", None, None))


def parse_seq_max_lens(spec: str) -> Dict[str, int]:
    out = {}
    for pair in spec.split(","):
        if not pair.strip():
            continue
        k, v = pair.split(":")
        out[k.strip()] = int(v.strip())
    return out


def build_raw_dump(c: ProfileCollectors) -> Dict[str, Any]:
    """Compact JSON-serializable dict of all raw stats."""
    out: Dict[str, Any] = {}
    out["meta"] = {
        "train_rows": c.train_total_rows,
        "val_rows": c.val_total_rows,
        "train_pos_count": c.train_pos_count,
        "val_pos_count": c.val_pos_count,
    }

    def _fid_pack(group: str, agg: FidAgg) -> Dict[str, Any]:
        return {
            "group": group,
            "fid": agg.plan.fid,
            "dim": agg.plan.dim,
            "vocab_schema": agg.plan.vocab,
            "vocab_observed": agg.freq.obs_max,
            "unique": agg.freq.num_unique(),
            "non_zero_rate": (agg.freq.row_with_value_count / max(agg.freq.row_count, 1)),
            "topk_coverage": dict(agg.freq.topk_coverage(TOPK_REPORT)),
            "k_at_coverage": {
                str(t): agg.freq.k_for_coverage(t) for t in COVERAGE_TARGETS
            },
            "gini": agg.freq.gini(),
            "entropy_bits": agg.freq.entropy(),
            "hhi": agg.freq.hhi(),
            "long_tail_lt5": agg.freq.long_tail_ratio(5),
            "val_oov": agg.oov_val.to_dict(),
        }

    out["user_int"] = [_fid_pack("user_int", a) for a in c.user_int_aggs]
    out["item_int"] = [_fid_pack("item_int", a) for a in c.item_int_aggs]

    out["user_dense"] = []
    for fid, dim, col_name in c.schema.user_dense:
        b = c.dense_aggs[fid]
        out["user_dense"].append({
            "fid": fid,
            "dim": dim,
            "row_count": b["row_count"],
            "nan_count": b["nan_count"],
            "inf_count": b["inf_count"],
            "all_zero_count": b["all_zero_count"],
            "per_dim": [
                {**w.to_dict(),
                 **{f"p{int(round(p*1000))/10}": (
                    b["reservoir"][i].percentile(p))
                    for p in (0.1, 0.5, 0.9, 0.99)}}
                for i, w in enumerate(b["welford"])
            ],
        })

    out["seq_sideinfo"] = {}
    for domain, aggs in c.seq_aggs.items():
        out["seq_sideinfo"][domain] = []
        for sa in aggs:
            out["seq_sideinfo"][domain].append({
                "fid": sa.fid,
                "vocab_schema": sa.vocab,
                "vocab_observed": sa.freq.obs_max,
                "unique": sa.freq.num_unique(),
                "topk_coverage": dict(sa.freq.topk_coverage(TOPK_REPORT)),
                "k_at_coverage": {
                    str(t): sa.freq.k_for_coverage(t) for t in COVERAGE_TARGETS
                },
                "gini": sa.freq.gini(),
                "entropy_bits": sa.freq.entropy(),
                "long_tail_lt5": sa.freq.long_tail_ratio(5),
                "val_oov": sa.oov_val.to_dict(),
            })

    out["seq_length"] = {
        domain: {
            "train": c.train_lens[domain].to_dict(),
            "val": c.val_lens[domain].to_dict(),
        }
        for domain in c.train_lens
    }

    out["seq_timestamp"] = {}
    for domain, s in c.seq_ts_stats.items():
        out["seq_timestamp"][domain] = {
            "has_ts": s["has_ts"],
            "ts_min": s["ts_min"],
            "ts_max": s["ts_max"],
            "positive_diff_count": s["positive_diff_count"],
            "negative_diff_count": s["negative_diff_count"],
            "zero_diff_count": s["zero_diff_count"],
            "diff_p50": s["diff_reservoir"].percentile(0.5),
            "diff_p99": s["diff_reservoir"].percentile(0.99),
            "pos0_diff_p50": s["pos0_diff_reservoir"].percentile(0.5),
            "poslast_diff_p50": s["poslast_diff_reservoir"].percentile(0.5),
            "bucket_counts": s["bucket_counts"].tolist(),
        }

    out["label"] = {
        "label_type_counts": dict(c.label_type_counts),
        "train_pos_rate": c.train_pos_count / max(c.train_total_rows, 1),
        "val_pos_rate": c.val_pos_count / max(c.val_total_rows, 1),
    }

    out["schema_health"] = {
        "missing_columns": c.missing_columns,
        "unused_columns": c.unused_columns,
    }

    return out


def main() -> None:
    args = parse_args()
    setup_logging(args.log_dir)
    log = logging.getLogger("profile")
    log.info("Starting TAAC profile pass; data_dir=%s", args.data_dir)

    seq_max_lens = parse_seq_max_lens(args.seq_max_lens)

    # ---- Load schema ----
    with open(args.schema_path, "r") as f:
        raw_schema = json.load(f)
    schema = SchemaPlan(raw_schema)
    log.info("Schema parsed: user_int=%d, item_int=%d, user_dense=%d, seq=%s",
             len(schema.user_int), len(schema.item_int),
             len(schema.user_dense), list(schema.seq.keys()))

    # ---- Discover row groups ----
    pq_files = _list_parquet_files(args.data_dir)
    rg_list, total_rows = _gather_row_groups(pq_files)
    n_total_rgs = len(rg_list)
    n_val_rgs = max(1, int(n_total_rgs * args.valid_ratio))
    n_train_rgs = n_total_rgs - n_val_rgs
    if args.train_ratio < 1.0:
        n_train_rgs = max(1, int(n_train_rgs * args.train_ratio))
    train_rgs = rg_list[:n_train_rgs]
    val_rgs = rg_list[n_total_rgs - n_val_rgs:]
    log.info("RG split: %d train + %d val (of %d total)",
             len(train_rgs), len(val_rgs), n_total_rgs)

    # ---- Resolve column indices from first parquet's arrow schema ----
    pf0 = pq.ParquetFile(pq_files[0])
    arrow_schema = pf0.schema_arrow
    col_idx = {name: i for i, name in enumerate(arrow_schema.names)}
    log.info("Parquet column count: %d", len(col_idx))

    # ---- Bind plans & detect missing/unused ----
    missing = schema.resolve_columns(col_idx)
    referenced: set = set()
    for plan in schema.user_int + schema.item_int:
        referenced.add(plan.col_name)
    for fid, dim, name in schema.user_dense:
        referenced.add(name)
    for domain, cfg in schema.seq.items():
        if cfg["ts_col_name"]:
            referenced.add(cfg["ts_col_name"])
        for fid, vs, name in cfg["sideinfo"]:
            referenced.add(name)
    referenced |= {"timestamp", "label_type", "user_id"}
    unused = [n for n in arrow_schema.names if n not in referenced]

    # ---- Build collectors ----
    c = ProfileCollectors(schema, seq_max_lens)
    c.parquet_files = pq_files
    c.column_names = list(arrow_schema.names)
    c.missing_columns = missing
    c.unused_columns = unused
    c.train_rg_rows = [n for _, _, n in train_rgs]
    c.val_rg_rows = [n for _, _, n in val_rgs]

    # ---- Train pass ----
    log.info("Train pass: %d row_groups, %d rows", len(train_rgs),
             sum(c.train_rg_rows))
    t0 = time.time()
    n = 0
    for rg_idx, batch in _iter_record_batches(train_rgs, batch_size=args.batch_size):
        c.feed_train_batch(batch, rg_idx, col_idx)
        n += batch.num_rows
        if args.max_train_rows and n >= args.max_train_rows:
            log.info("max_train_rows hit at %d rows", n)
            break
        if n % (args.batch_size * 50) == 0:
            log.info("  train: %d / %d rows  (%.1f s)",
                     n, sum(c.train_rg_rows), time.time() - t0)
    log.info("Train pass done: %d rows in %.1f s", n, time.time() - t0)

    # ---- Val pass ----
    log.info("Val pass: %d row_groups, %d rows", len(val_rgs),
             sum(c.val_rg_rows))
    t1 = time.time()
    nv = 0
    for rg_idx, batch in _iter_record_batches(val_rgs, batch_size=args.batch_size):
        c.feed_val_batch(batch, rg_idx, col_idx)
        nv += batch.num_rows
    log.info("Val pass done: %d rows in %.1f s", nv, time.time() - t1)

    # ---- Build raw dump + render report ----
    raw = build_raw_dump(c)
    lines = render_report(c, args, raw)
    log.info("Report rendered: %d lines (budget 1000)", len(lines))

    # ---- Write to disk if log_dir available ----
    if args.log_dir:
        try:
            json_path = os.path.join(args.log_dir, "profile_data.json")
            with open(json_path, "w") as fp:
                json.dump(raw, fp, indent=2, default=str)
            log.info("Wrote raw dump to %s", json_path)
            md_path = os.path.join(args.log_dir, "profile_report.md")
            with open(md_path, "w") as fp:
                fp.write("\n".join(lines))
            log.info("Wrote markdown to %s", md_path)
        except OSError as e:
            log.warning("Failed to write disk artifacts: %s", e)

    # ---- Print to stdout (the platform's only retrieval channel) ----
    print("\n".join(lines), flush=True)
    print(f"PROFILE_JSON_DUMP={json.dumps(raw, separators=(',', ':'), default=str)}",
          flush=True)


if __name__ == "__main__":
    main()
