"""Missing-value EDA for TAAC parquet data.

Platform output contract:
  - stdout starts with a compact markdown report.
  - stdout ends with two copy-friendly one-line payloads:
      MISSING_EDA_REPORT_ONE_LINE=<json string containing the markdown>
      MISSING_EDA_JSON_DUMP=<compact json>
  - stderr is used for progress only.

The script mirrors the row-group split used by ``dataset.get_pcvr_data`` and is
read-only. It focuses on the ambiguity currently hidden by dataset conversion:
raw null / -1 / 0 all become id 0, and id 0 is later treated as padding.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


JSON_PREFIX = "MISSING_EDA_JSON_DUMP="
REPORT_PREFIX = "MISSING_EDA_REPORT_ONE_LINE="


@dataclass
class IntPlan:
    group: str
    fid: int
    vocab: int
    dim: int
    col_name: str


@dataclass
class DensePlan:
    fid: int
    dim: int
    col_name: str


@dataclass
class SeqPlan:
    domain: str
    prefix: str
    features: List[Tuple[int, int, str]]


def _pct(x: float) -> str:
    return f"{100.0 * x:.3f}%"


def _rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def _safe_ratio(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return num / den


class RowLift:
    """Track row count and positive labels for a boolean condition."""

    __slots__ = ("rows", "pos")

    def __init__(self) -> None:
        self.rows = 0
        self.pos = 0

    def update(self, flag: np.ndarray, label: np.ndarray) -> None:
        if flag.size == 0:
            return
        self.rows += int(flag.sum())
        if flag.any():
            self.pos += int(label[flag].sum())

    def to_dict(self, total_rows: int, base_pos_rate: float) -> Dict[str, Any]:
        pr = _rate(self.pos, self.rows)
        lift = _safe_ratio(pr, base_pos_rate)
        return {
            "rows": self.rows,
            "row_rate": _rate(self.rows, total_rows),
            "pos": self.pos,
            "pos_rate": pr,
            "lift_vs_base": lift,
        }


class IntMissingAgg:
    """Raw-missing counters for scalar or list categorical fids."""

    def __init__(self, plan: IntPlan) -> None:
        self.plan = plan
        self.rows = 0
        self.value_total = 0
        self.null_rows = RowLift()
        self.empty_rows = RowLift()
        self.no_positive_rows = RowLift()
        self.any_negative_rows = RowLift()
        self.any_neg1_rows = RowLift()
        self.any_zero_rows = RowLift()
        self.present_rows = RowLift()
        self.value_neg = 0
        self.value_neg1 = 0
        self.value_zero = 0
        self.value_positive = 0

    def update(self, col: pa.Array, label: np.ndarray) -> None:
        B = len(col)
        self.rows += B
        null_flag = _null_mask(col, B)
        self.null_rows.update(null_flag, label)

        if pa.types.is_list(col.type) or pa.types.is_large_list(col.type):
            offsets = col.offsets.to_numpy(zero_copy_only=False)
            values = col.values.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            lengths = offsets[1:] - offsets[:-1]
            empty_flag = (~null_flag) & (lengths == 0)
            any_neg = np.zeros(B, dtype=bool)
            any_neg1 = np.zeros(B, dtype=bool)
            any_zero = np.zeros(B, dtype=bool)
            any_pos = np.zeros(B, dtype=bool)
            for i in range(B):
                if null_flag[i]:
                    continue
                s = int(offsets[i])
                e = int(offsets[i + 1])
                if e <= s:
                    continue
                vals = values[s:e]
                self.value_total += int(vals.size)
                self.value_neg += int((vals < 0).sum())
                self.value_neg1 += int((vals == -1).sum())
                self.value_zero += int((vals == 0).sum())
                self.value_positive += int((vals > 0).sum())
                any_neg[i] = bool((vals < 0).any())
                any_neg1[i] = bool((vals == -1).any())
                any_zero[i] = bool((vals == 0).any())
                any_pos[i] = bool((vals > 0).any())
            self.empty_rows.update(empty_flag, label)
            self.any_negative_rows.update(any_neg, label)
            self.any_neg1_rows.update(any_neg1, label)
            self.any_zero_rows.update(any_zero, label)
            self.present_rows.update(any_pos, label)
            self.no_positive_rows.update(~any_pos, label)
        else:
            arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            non_null = ~null_flag
            neg = non_null & (arr < 0)
            neg1 = non_null & (arr == -1)
            zero = non_null & (arr == 0)
            pos = non_null & (arr > 0)
            self.value_total += int(non_null.sum())
            self.value_neg += int(neg.sum())
            self.value_neg1 += int(neg1.sum())
            self.value_zero += int(zero.sum())
            self.value_positive += int(pos.sum())
            self.any_negative_rows.update(neg, label)
            self.any_neg1_rows.update(neg1, label)
            self.any_zero_rows.update(zero, label)
            self.present_rows.update(pos, label)
            self.no_positive_rows.update(~pos, label)

    def to_dict(self, total_rows: int, base_pos_rate: float) -> Dict[str, Any]:
        return {
            "group": self.plan.group,
            "fid": self.plan.fid,
            "dim": self.plan.dim,
            "vocab": self.plan.vocab,
            "rows": self.rows,
            "value_total": self.value_total,
            "value_rates": {
                "negative": _rate(self.value_neg, self.value_total),
                "neg1": _rate(self.value_neg1, self.value_total),
                "zero": _rate(self.value_zero, self.value_total),
                "positive": _rate(self.value_positive, self.value_total),
            },
            "row_lift": {
                "null": self.null_rows.to_dict(total_rows, base_pos_rate),
                "empty_list": self.empty_rows.to_dict(total_rows, base_pos_rate),
                "no_positive": self.no_positive_rows.to_dict(total_rows, base_pos_rate),
                "any_negative": self.any_negative_rows.to_dict(total_rows, base_pos_rate),
                "any_neg1": self.any_neg1_rows.to_dict(total_rows, base_pos_rate),
                "any_zero": self.any_zero_rows.to_dict(total_rows, base_pos_rate),
                "present": self.present_rows.to_dict(total_rows, base_pos_rate),
            },
        }


class DenseMissingAgg:
    def __init__(self, plan: DensePlan) -> None:
        self.plan = plan
        self.rows = 0
        self.value_total = 0
        self.value_nan = 0
        self.value_inf = 0
        self.null_rows = RowLift()
        self.empty_rows = RowLift()
        self.short_rows = RowLift()
        self.all_zero_rows = RowLift()
        self.any_nan_rows = RowLift()
        self.any_inf_rows = RowLift()
        self.present_nonzero_rows = RowLift()

    def update(self, col: pa.Array, label: np.ndarray) -> None:
        B = len(col)
        self.rows += B
        null_flag = _null_mask(col, B)
        self.null_rows.update(null_flag, label)
        offsets = col.offsets.to_numpy(zero_copy_only=False)
        values = col.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        lengths = offsets[1:] - offsets[:-1]
        empty = (~null_flag) & (lengths == 0)
        short = (~null_flag) & (lengths < self.plan.dim)
        all_zero = np.zeros(B, dtype=bool)
        any_nan = np.zeros(B, dtype=bool)
        any_inf = np.zeros(B, dtype=bool)
        present_nonzero = np.zeros(B, dtype=bool)
        for i in range(B):
            if null_flag[i]:
                all_zero[i] = True
                continue
            s = int(offsets[i])
            e = int(offsets[i + 1])
            vals = values[s:e]
            use = vals[: self.plan.dim]
            self.value_total += int(use.size)
            if use.size:
                nan = np.isnan(use)
                inf = np.isinf(use)
                self.value_nan += int(nan.sum())
                self.value_inf += int(inf.sum())
                any_nan[i] = bool(nan.any())
                any_inf[i] = bool(inf.any())
                present_nonzero[i] = bool((np.nan_to_num(use, nan=0.0, posinf=0.0, neginf=0.0) != 0).any())
            all_zero[i] = not present_nonzero[i]
        self.empty_rows.update(empty, label)
        self.short_rows.update(short, label)
        self.all_zero_rows.update(all_zero, label)
        self.any_nan_rows.update(any_nan, label)
        self.any_inf_rows.update(any_inf, label)
        self.present_nonzero_rows.update(present_nonzero, label)

    def to_dict(self, total_rows: int, base_pos_rate: float) -> Dict[str, Any]:
        return {
            "fid": self.plan.fid,
            "dim": self.plan.dim,
            "rows": self.rows,
            "value_total": self.value_total,
            "value_nan_rate": _rate(self.value_nan, self.value_total),
            "value_inf_rate": _rate(self.value_inf, self.value_total),
            "row_lift": {
                "null": self.null_rows.to_dict(total_rows, base_pos_rate),
                "empty_list": self.empty_rows.to_dict(total_rows, base_pos_rate),
                "short_list": self.short_rows.to_dict(total_rows, base_pos_rate),
                "all_zero": self.all_zero_rows.to_dict(total_rows, base_pos_rate),
                "any_nan": self.any_nan_rows.to_dict(total_rows, base_pos_rate),
                "any_inf": self.any_inf_rows.to_dict(total_rows, base_pos_rate),
                "present_nonzero": self.present_nonzero_rows.to_dict(total_rows, base_pos_rate),
            },
        }


class SeqDomainAgg:
    def __init__(self, domain: str) -> None:
        self.domain = domain
        self.rows = 0
        self.empty_raw_rows = RowLift()
        self.no_positive_rows = RowLift()
        self.any_nonpositive_token_rows = RowLift()
        self.max_raw_len_sum = 0

    def update(self, feature_cols: List[pa.Array], label: np.ndarray) -> None:
        B = len(label)
        self.rows += B
        max_len = np.zeros(B, dtype=np.int64)
        any_pos = np.zeros(B, dtype=bool)
        any_nonpositive = np.zeros(B, dtype=bool)
        for col in feature_cols:
            null_flag = _null_mask(col, B)
            offsets = col.offsets.to_numpy(zero_copy_only=False)
            values = col.values.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            lengths = offsets[1:] - offsets[:-1]
            max_len = np.maximum(max_len, np.where(null_flag, 0, lengths))
            for i in range(B):
                if null_flag[i]:
                    continue
                s = int(offsets[i])
                e = int(offsets[i + 1])
                if e <= s:
                    continue
                vals = values[s:e]
                any_pos[i] |= bool((vals > 0).any())
                any_nonpositive[i] |= bool((vals <= 0).any())
        self.max_raw_len_sum += int(max_len.sum())
        self.empty_raw_rows.update(max_len == 0, label)
        self.no_positive_rows.update(~any_pos, label)
        self.any_nonpositive_token_rows.update(any_nonpositive, label)

    def to_dict(self, total_rows: int, base_pos_rate: float) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "rows": self.rows,
            "mean_raw_len": _rate(self.max_raw_len_sum, self.rows),
            "row_lift": {
                "empty_raw": self.empty_raw_rows.to_dict(total_rows, base_pos_rate),
                "no_positive_sideinfo": self.no_positive_rows.to_dict(total_rows, base_pos_rate),
                "any_nonpositive_token": self.any_nonpositive_token_rows.to_dict(total_rows, base_pos_rate),
            },
        }


def _null_mask(col: pa.Array, B: int) -> np.ndarray:
    if col.null_count == 0:
        return np.zeros(B, dtype=bool)
    return pc.is_null(col).to_numpy(zero_copy_only=False)


def _list_parquet_files(data_dir: str) -> List[str]:
    p = Path(data_dir)
    if p.is_file():
        return [str(p)]
    files = sorted(str(x) for x in p.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {data_dir}")
    return files


def _row_groups(files: Iterable[str]) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    for f in files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            out.append((f, i, pf.metadata.row_group(i).num_rows))
    return out


def _iter_batches(rgs: List[Tuple[str, int, int]], batch_size: int) -> Iterable[pa.RecordBatch]:
    for f, rg, _ in rgs:
        pf = pq.ParquetFile(f)
        table = pf.read_row_group(rg)
        for batch in table.to_batches(max_chunksize=batch_size):
            yield batch


def _parse_seq_max_lens(spec: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for part in spec.split(","):
        if not part.strip():
            continue
        k, v = part.split(":")
        out[k.strip()] = int(v.strip())
    return out


def _build_plans(schema: Dict[str, Any]) -> Tuple[List[IntPlan], List[IntPlan], List[DensePlan], List[SeqPlan]]:
    user_int = [
        IntPlan("user_int", int(fid), int(vs), int(dim), f"user_int_feats_{fid}")
        for fid, vs, dim in schema.get("user_int", [])
    ]
    item_int = [
        IntPlan("item_int", int(fid), int(vs), int(dim), f"item_int_feats_{fid}")
        for fid, vs, dim in schema.get("item_int", [])
    ]
    dense = [
        DensePlan(int(fid), int(dim), f"user_dense_feats_{fid}")
        for fid, dim in schema.get("user_dense", [])
    ]
    seq: List[SeqPlan] = []
    for domain, cfg in sorted(schema.get("seq", {}).items()):
        prefix = cfg["prefix"]
        features = [(int(fid), int(vs), f"{prefix}_{fid}") for fid, vs in cfg.get("features", [])]
        seq.append(SeqPlan(domain, prefix, features))
    return user_int, item_int, dense, seq


def _label(batch: pa.RecordBatch, col_idx: Dict[str, int]) -> np.ndarray:
    if "label_type" not in col_idx:
        return np.zeros(batch.num_rows, dtype=np.int64)
    return (
        batch.column(col_idx["label_type"])
        .fill_null(0)
        .to_numpy(zero_copy_only=False)
        .astype(np.int64, copy=False)
        == 2
    ).astype(np.int64)


def _feed_split(
    batches: Iterable[pa.RecordBatch],
    col_idx: Dict[str, int],
    user_aggs: List[IntMissingAgg],
    item_aggs: List[IntMissingAgg],
    dense_aggs: List[DenseMissingAgg],
    seq_aggs: List[SeqDomainAgg],
    seq_plans: List[SeqPlan],
    max_rows: int,
    log_every: int,
    split_name: str,
) -> Dict[str, int]:
    rows = 0
    pos = 0
    for batch in batches:
        labels = _label(batch, col_idx)
        rows += batch.num_rows
        pos += int(labels.sum())
        for agg in user_aggs + item_aggs:
            ci = col_idx.get(agg.plan.col_name)
            if ci is not None:
                agg.update(batch.column(ci), labels)
        for agg in dense_aggs:
            ci = col_idx.get(agg.plan.col_name)
            if ci is not None:
                agg.update(batch.column(ci), labels)
        for seq_agg, plan in zip(seq_aggs, seq_plans):
            cols = [batch.column(col_idx[name]) for _, _, name in plan.features if name in col_idx]
            if cols:
                seq_agg.update(cols, labels)
        if log_every > 0 and rows % log_every < batch.num_rows:
            logging.info("%s rows=%s", split_name, rows)
        if max_rows > 0 and rows >= max_rows:
            break
    return {"rows": rows, "pos": pos}


def _pack_int(aggs: List[IntMissingAgg], total_rows: int, base_pos_rate: float) -> List[Dict[str, Any]]:
    return [a.to_dict(total_rows, base_pos_rate) for a in aggs]


def _pack_dense(aggs: List[DenseMissingAgg], total_rows: int, base_pos_rate: float) -> List[Dict[str, Any]]:
    return [a.to_dict(total_rows, base_pos_rate) for a in aggs]


def _pack_seq(aggs: List[SeqDomainAgg], total_rows: int, base_pos_rate: float) -> List[Dict[str, Any]]:
    return [a.to_dict(total_rows, base_pos_rate) for a in aggs]


def _new_aggs(
    user_plans: List[IntPlan],
    item_plans: List[IntPlan],
    dense_plans: List[DensePlan],
    seq_plans: List[SeqPlan],
    col_idx: Dict[str, int],
) -> Tuple[List[IntMissingAgg], List[IntMissingAgg], List[DenseMissingAgg], List[SeqDomainAgg], List[SeqPlan]]:
    user_aggs = [IntMissingAgg(p) for p in user_plans if p.col_name in col_idx]
    item_aggs = [IntMissingAgg(p) for p in item_plans if p.col_name in col_idx]
    dense_aggs = [DenseMissingAgg(p) for p in dense_plans if p.col_name in col_idx]
    seq_present = [
        SeqPlan(s.domain, s.prefix, [f for f in s.features if f[2] in col_idx])
        for s in seq_plans
    ]
    seq_present = [s for s in seq_present if s.features]
    seq_aggs = [SeqDomainAgg(s.domain) for s in seq_present]
    return user_aggs, item_aggs, dense_aggs, seq_aggs, seq_present


def _pack_split(
    meta: Dict[str, int],
    user_aggs: List[IntMissingAgg],
    item_aggs: List[IntMissingAgg],
    dense_aggs: List[DenseMissingAgg],
    seq_aggs: List[SeqDomainAgg],
) -> Dict[str, Any]:
    base = _rate(meta["pos"], meta["rows"])
    return {
        "meta": {
            "rows": meta["rows"],
            "pos": meta["pos"],
            "pos_rate": base,
        },
        "user_int": _pack_int(user_aggs, meta["rows"], base),
        "item_int": _pack_int(item_aggs, meta["rows"], base),
        "user_dense": _pack_dense(dense_aggs, meta["rows"], base),
        "seq_domain": _pack_seq(seq_aggs, meta["rows"], base),
    }


def _lift(d: Dict[str, Any], key: str) -> Optional[float]:
    return d["row_lift"][key]["lift_vs_base"]


def _top_by(data: List[Dict[str, Any]], key: str, n: int = 12) -> List[Dict[str, Any]]:
    return sorted(
        data,
        key=lambda x: (
            x["row_lift"][key]["row_rate"],
            abs((x["row_lift"][key]["lift_vs_base"] or 1.0) - 1.0),
        ),
        reverse=True,
    )[:n]


def render_report(raw: Dict[str, Any]) -> str:
    meta = raw["meta"]
    base = meta["pos_rate"]
    val_by_key = {
        (r["group"], r["fid"]): r
        for r in raw.get("val", {}).get("user_int", []) + raw.get("val", {}).get("item_int", [])
    }
    lines: List[str] = []
    L = lines.append
    L("# TAAC Missing-Value EDA")
    L("")
    L("## Summary")
    L("")
    L(f"- rows: {_fmt_int(meta['rows'])}; pos_rate: {_pct(base)}")
    L(f"- schema missing columns: {len(raw['schema_health']['missing_columns'])}")
    L("- Goal: split raw `null / -1 / 0 / empty / all-zero` signals that current dataset maps to id 0.")
    L("")
    L("## High-Rate Categorical No-Positive Rows")
    L("")
    L("| group | fid | dim | no_positive_rate | no_positive_pos_rate | lift | present_pos_rate |")
    L("|---|---:|---:|---:|---:|---:|---:|")
    cats = raw["user_int"] + raw["item_int"]
    for r in _top_by(cats, "no_positive", 16):
        miss = r["row_lift"]["no_positive"]
        pres = r["row_lift"]["present"]
        L(
            f"| {r['group']} | {r['fid']} | {r['dim']} | "
            f"{_pct(miss['row_rate'])} | {_pct(miss['pos_rate'])} | "
            f"{_fmt_lift(miss['lift_vs_base'])} | {_pct(pres['pos_rate'])} |"
        )
    L("")
    L("## Train/Val Missing Stability")
    L("")
    L("| group | fid | train_no_positive | val_no_positive | Δpp | train_lift | val_lift |")
    L("|---|---:|---:|---:|---:|---:|---:|")
    for r in _top_by(cats, "no_positive", 12):
        v = val_by_key.get((r["group"], r["fid"]))
        tr = r["row_lift"]["no_positive"]
        if v is None:
            L(
                f"| {r['group']} | {r['fid']} | {_pct(tr['row_rate'])} | n/a | n/a | "
                f"{_fmt_lift(tr['lift_vs_base'])} | n/a |"
            )
            continue
        va = v["row_lift"]["no_positive"]
        L(
            f"| {r['group']} | {r['fid']} | {_pct(tr['row_rate'])} | "
            f"{_pct(va['row_rate'])} | {(va['row_rate'] - tr['row_rate']) * 100:+.3f} | "
            f"{_fmt_lift(tr['lift_vs_base'])} | {_fmt_lift(va['lift_vs_base'])} |"
        )
    L("")
    L("## Raw 0 / -1 / null Signals")
    L("")
    L("| group | fid | null_rate | any_neg1_rate | any_zero_rate | value_zero_rate | value_neg1_rate |")
    L("|---|---:|---:|---:|---:|---:|---:|")
    ranked_zero = sorted(cats, key=lambda x: x["row_lift"]["any_zero"]["row_rate"], reverse=True)[:16]
    for r in ranked_zero:
        L(
            f"| {r['group']} | {r['fid']} | "
            f"{_pct(r['row_lift']['null']['row_rate'])} | "
            f"{_pct(r['row_lift']['any_neg1']['row_rate'])} | "
            f"{_pct(r['row_lift']['any_zero']['row_rate'])} | "
            f"{_pct(r['value_rates']['zero'])} | {_pct(r['value_rates']['neg1'])} |"
        )
    L("")
    L("## Dense All-Zero Rows")
    L("")
    L("| fid | dim | all_zero_rate | all_zero_pos_rate | lift | present_nonzero_pos_rate | nan_rate | inf_rate |")
    L("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in _top_by(raw["user_dense"], "all_zero", 12):
        z = r["row_lift"]["all_zero"]
        p = r["row_lift"]["present_nonzero"]
        L(
            f"| {r['fid']} | {r['dim']} | {_pct(z['row_rate'])} | "
            f"{_pct(z['pos_rate'])} | {_fmt_lift(z['lift_vs_base'])} | "
            f"{_pct(p['pos_rate'])} | {_pct(r['value_nan_rate'])} | {_pct(r['value_inf_rate'])} |"
        )
    L("")
    L("## Sequence Missing-Like Rows")
    L("")
    L("| domain | mean_raw_len | empty_raw_rate | no_positive_sideinfo_rate | any_nonpositive_token_rate | no_positive_lift |")
    L("|---|---:|---:|---:|---:|---:|")
    for r in raw["seq_domain"]:
        np_row = r["row_lift"]["no_positive_sideinfo"]
        L(
            f"| {r['domain']} | {r['mean_raw_len']:.2f} | "
            f"{_pct(r['row_lift']['empty_raw']['row_rate'])} | "
            f"{_pct(np_row['row_rate'])} | "
            f"{_pct(r['row_lift']['any_nonpositive_token']['row_rate'])} | "
            f"{_fmt_lift(np_row['lift_vs_base'])} |"
        )
    L("")
    L("## Decision Hints")
    L("")
    L("- If high no-positive categorical fids show lift != 1, use a learnable missing bucket for scalar categorical fids instead of frozen padding 0.")
    L("- Keep multi-value and sequence padding as 0; only add missing buckets where raw missing/0 is feature value semantics, not structural padding.")
    L("- If dense all-zero rows have lift, add per-fid all-zero/missing indicators or a dense-missing token.")
    L("- NaN/Inf should remain a guardrail, not the main missing-value strategy, unless rates are non-zero above.")
    return "\n".join(lines) + "\n"


def _fmt_int(x: int) -> str:
    return f"{x:,}"


def _fmt_lift(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.3f}x"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TAAC missing-value EDA")
    p.add_argument("--data_dir", type=str, default=None, help="env fallback: TRAIN_DATA_PATH")
    p.add_argument("--schema_path", type=str, default=None, help="default: <data_dir>/schema.json")
    p.add_argument("--log_dir", type=str, default=None, help="env fallback: TRAIN_LOG_PATH")
    p.add_argument("--valid_ratio", type=float, default=0.1)
    p.add_argument("--train_ratio", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--max_rows", type=int, default=0, help="debug cap across train split only")
    p.add_argument("--log_every", type=int, default=250_000)
    p.add_argument("--seq_max_lens", type=str, default="seq_a:256,seq_b:256,seq_c:512,seq_d:512")
    args = p.parse_args()
    args.data_dir = os.environ.get("TRAIN_DATA_PATH", args.data_dir)
    args.log_dir = os.environ.get("TRAIN_LOG_PATH", args.log_dir)
    if not args.data_dir:
        raise SystemExit("TRAIN_DATA_PATH or --data_dir is required")
    if args.schema_path is None:
        args.schema_path = os.path.join(args.data_dir, "schema.json")
    return args


def setup_logging(log_dir: Optional[str]) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_dir)
    _ = _parse_seq_max_lens(args.seq_max_lens)  # kept for CLI symmetry with profile_data.py

    with open(args.schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    user_plans, item_plans, dense_plans, seq_plans = _build_plans(schema)

    pq_files = _list_parquet_files(args.data_dir)
    rgs = _row_groups(pq_files)
    if not rgs:
        raise SystemExit("no row groups found")
    n_val = max(1, int(len(rgs) * args.valid_ratio))
    n_train = len(rgs) - n_val
    if args.train_ratio < 1.0:
        n_train = max(1, int(n_train * args.train_ratio))
    train_rgs = rgs[:n_train]
    val_rgs = rgs[len(rgs) - n_val :]
    logging.info("row groups: train=%d val=%d total=%d", len(train_rgs), len(val_rgs), len(rgs))

    schema_names = pq.ParquetFile(pq_files[0]).schema_arrow.names
    col_idx = {name: i for i, name in enumerate(schema_names)}
    expected_cols = (
        [p.col_name for p in user_plans]
        + [p.col_name for p in item_plans]
        + [p.col_name for p in dense_plans]
        + [name for s in seq_plans for _, _, name in s.features]
        + ["label_type"]
    )
    missing_cols = sorted([c for c in expected_cols if c not in col_idx])
    unused_cols = sorted([c for c in schema_names if c not in set(expected_cols + ["timestamp", "user_id", "item_id", "label_time"])])

    user_aggs, item_aggs, dense_aggs, seq_aggs, seq_plans_present = _new_aggs(
        user_plans, item_plans, dense_plans, seq_plans, col_idx
    )
    val_user_aggs, val_item_aggs, val_dense_aggs, val_seq_aggs, val_seq_plans_present = _new_aggs(
        user_plans, item_plans, dense_plans, seq_plans, col_idx
    )

    train_meta = _feed_split(
        _iter_batches(train_rgs, args.batch_size),
        col_idx,
        user_aggs,
        item_aggs,
        dense_aggs,
        seq_aggs,
        seq_plans_present,
        args.max_rows,
        args.log_every,
        "train",
    )
    val_meta = _feed_split(
        _iter_batches(val_rgs, args.batch_size),
        col_idx,
        val_user_aggs,
        val_item_aggs,
        val_dense_aggs,
        val_seq_aggs,
        val_seq_plans_present,
        0,
        args.log_every,
        "val",
    )

    base = _rate(train_meta["pos"], train_meta["rows"])
    train_pack = _pack_split(train_meta, user_aggs, item_aggs, dense_aggs, seq_aggs)
    val_pack = _pack_split(val_meta, val_user_aggs, val_item_aggs, val_dense_aggs, val_seq_aggs)
    raw = {
        "meta": {
            "rows": train_meta["rows"],
            "pos": train_meta["pos"],
            "pos_rate": base,
            "val_rows": val_meta["rows"],
            "val_pos": val_meta["pos"],
            "val_pos_rate": _rate(val_meta["pos"], val_meta["rows"]),
            "valid_ratio": args.valid_ratio,
            "train_ratio": args.train_ratio,
            "data_dir": args.data_dir,
            "schema_path": args.schema_path,
        },
        "user_int": train_pack["user_int"],
        "item_int": train_pack["item_int"],
        "user_dense": train_pack["user_dense"],
        "seq_domain": train_pack["seq_domain"],
        "train": train_pack,
        "val": val_pack,
        "schema_health": {
            "missing_columns": missing_cols,
            "unused_columns": unused_cols,
        },
    }
    report = render_report(raw)

    if args.log_dir:
        out_dir = Path(args.log_dir)
        (out_dir / "missing_eda_report.md").write_text(report, encoding="utf-8")
        (out_dir / "missing_eda.json").write_text(
            json.dumps(raw, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(report, end="", flush=True)
    print(f"{REPORT_PREFIX}{json.dumps(report, ensure_ascii=False, separators=(',', ':'))}", flush=True)
    print(f"{JSON_PREFIX}{json.dumps(raw, ensure_ascii=False, separators=(',', ':'))}", flush=True)


if __name__ == "__main__":
    main()
