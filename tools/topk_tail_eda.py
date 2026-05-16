#!/usr/bin/env python3
"""Streaming EDA for TopK embedding + tail-default rescue candidates.

This script is designed for high-cardinality TAAC sequence features where a
plain exact id Counter can consume too much memory. It runs two streaming
passes over Parquet row groups:

1. Candidate pass: keep a bounded per-feature heavy-hitter Counter.
2. Exact pass: exactly count only those candidate ids, and aggregate all other
   ids into tail statistics.

Example:
    python tools/topk_tail_eda.py \
        --data-dir "$TRAIN_DATA_PATH" \
        --schema-path "$TRAIN_DATA_PATH/schema.json" \
        --workers 8 \
        --candidate-capacity 300000 \
        --out-json output/topk_tail_eda.json \
        --out-md output/topk_tail_eda.md
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as futures
import heapq
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_TARGETS = (
    "seq_a:38",
    "seq_b:69",
    "seq_b:74",
    "seq_b:76",
    "seq_b:88",
    "seq_c:29",
    "seq_c:34",
    "seq_c:36",
    "seq_c:47",
    "seq_d:22",
    "seq_d:23",
)

DEFAULT_TOPKS = (1000, 10000, 50000, 100000, 200000)
POS_BUCKETS = (
    ("pos0_10", 0, 10),
    ("pos10_50", 10, 50),
    ("pos50_100", 50, 100),
    ("pos100_256", 100, 256),
    ("pos256_plus", 256, None),
)


@dataclass(frozen=True)
class Target:
    domain: str
    fid: int
    vocab_size: int
    feature_col: str
    ts_col: str | None

    @property
    def key(self) -> str:
        return f"{self.domain}:{self.fid}"


@dataclass
class BasicStats:
    rows: int = 0
    pos_rows: int = 0
    rows_present: int = 0
    pos_rows_present: int = 0
    total_tokens: int = 0
    nonzero_tokens: int = 0
    pos_tokens: int = 0
    nonzero_pos_tokens: int = 0
    raw_len_hist: collections.Counter[int] = field(default_factory=collections.Counter)
    nonzero_len_hist: collections.Counter[int] = field(default_factory=collections.Counter)
    pos_bucket_total: collections.Counter[str] = field(default_factory=collections.Counter)
    pos_bucket_pos: collections.Counter[str] = field(default_factory=collections.Counter)

    def update_lengths(self, raw_len: int, nonzero_len: int, is_pos: bool) -> None:
        self.raw_len_hist[int(raw_len)] += 1
        self.nonzero_len_hist[int(nonzero_len)] += 1
        if raw_len > 0:
            self.rows_present += 1
            if is_pos:
                self.pos_rows_present += 1

    def merge(self, other: "BasicStats") -> None:
        self.rows += other.rows
        self.pos_rows += other.pos_rows
        self.rows_present += other.rows_present
        self.pos_rows_present += other.pos_rows_present
        self.total_tokens += other.total_tokens
        self.nonzero_tokens += other.nonzero_tokens
        self.pos_tokens += other.pos_tokens
        self.nonzero_pos_tokens += other.nonzero_pos_tokens
        self.raw_len_hist.update(other.raw_len_hist)
        self.nonzero_len_hist.update(other.nonzero_len_hist)
        self.pos_bucket_total.update(other.pos_bucket_total)
        self.pos_bucket_pos.update(other.pos_bucket_pos)


@dataclass
class ExactStats:
    head_counts: collections.Counter[int] = field(default_factory=collections.Counter)
    head_pos_counts: collections.Counter[int] = field(default_factory=collections.Counter)
    head_pos_bucket_counts: dict[str, collections.Counter[int]] = field(
        default_factory=lambda: {name: collections.Counter() for name, _, _ in POS_BUCKETS}
    )
    tail_count: int = 0
    tail_pos_count: int = 0
    tail_pos_bucket_total: collections.Counter[str] = field(default_factory=collections.Counter)
    tail_pos_bucket_pos: collections.Counter[str] = field(default_factory=collections.Counter)

    def merge(self, other: "ExactStats") -> None:
        self.head_counts.update(other.head_counts)
        self.head_pos_counts.update(other.head_pos_counts)
        for name in self.head_pos_bucket_counts:
            self.head_pos_bucket_counts[name].update(other.head_pos_bucket_counts[name])
        self.tail_count += other.tail_count
        self.tail_pos_count += other.tail_pos_count
        self.tail_pos_bucket_total.update(other.tail_pos_bucket_total)
        self.tail_pos_bucket_pos.update(other.tail_pos_bucket_pos)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TopK + tail-default EDA for high-cardinality TAAC sequence fids"
    )
    parser.add_argument("--data-dir", required=True, help="Parquet directory or a single parquet file")
    parser.add_argument("--schema-path", required=True, help="TAAC schema.json path")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS),
                        help="Comma-separated targets, e.g. seq_b:69,seq_c:29")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers. Default is 8-thread mode.")
    parser.add_argument("--executor", choices=("process", "thread"), default="process",
                        help="process gives real parallelism for Python-heavy token counting")
    parser.add_argument("--row-group-batch-size", type=int, default=10,
                        help="Number of row groups per submitted task. Smaller values print progress more often.")
    parser.add_argument("--candidate-capacity", type=int, default=300000,
                        help="Bounded candidate ids kept per target in pass 1")
    parser.add_argument("--local-prune-factor", type=float, default=1.5,
                        help="Prune local counters above capacity * factor after each row group")
    parser.add_argument("--max-row-groups", type=int, default=0,
                        help="Debug cap. 0 means scan all row groups.")
    parser.add_argument("--topks", default=",".join(str(k) for k in DEFAULT_TOPKS),
                        help="Comma-separated coverage cutoffs")
    parser.add_argument("--out-json", default="output/topk_tail_eda.json")
    parser.add_argument("--out-md", default="output/topk_tail_eda.md")
    parser.add_argument("--export-topk-map", default="",
                        help="Optional path for a TopK id map JSON used by TopK+default rescue.")
    parser.add_argument("--export-map-targets", default="",
                        help="Comma-separated export specs, e.g. seq_c:34:10000,seq_a:38:100000")
    parser.add_argument("--print-topk-map-one-line", action="store_true",
                        help="Print exported TopK map JSON as one JSON-escaped line.")
    parser.add_argument("--print-md", action="store_true",
                        help="Print the full Markdown report to stdout for platforms where output files are inaccessible.")
    parser.add_argument("--print-json", action="store_true",
                        help="Print the full JSON result to stdout after the Markdown/summary output.")
    parser.add_argument("--print-md-one-line", action="store_true", default=True,
                        help="Print the full Markdown report as one JSON-escaped line. Enabled by default.")
    parser.add_argument("--no-print-md-one-line", dest="print_md_one_line", action="store_false",
                        help="Disable the one-line Markdown report output.")
    return parser.parse_args()


def list_parquet_files(data_dir: Path) -> list[Path]:
    if data_dir.is_file():
        return [data_dir]
    return sorted(data_dir.glob("*.parquet"))


def load_schema(schema_path: Path) -> dict[str, Any]:
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_targets(schema: dict[str, Any], target_specs: Iterable[str]) -> dict[str, Target]:
    seq_cfg = schema["seq"]
    targets: dict[str, Target] = {}
    for spec in target_specs:
        spec = spec.strip()
        if not spec:
            continue
        domain, fid_s = spec.split(":")
        fid = int(fid_s)
        cfg = seq_cfg[domain]
        vocab_by_fid = {int(f): int(vs) for f, vs in cfg["features"]}
        if fid not in vocab_by_fid:
            raise ValueError(f"target {spec} not found in schema")
        prefix = cfg["prefix"]
        ts_fid = cfg.get("ts_fid")
        targets[spec] = Target(
            domain=domain,
            fid=fid,
            vocab_size=vocab_by_fid[fid],
            feature_col=f"{prefix}_{fid}",
            ts_col=f"{prefix}_{ts_fid}" if ts_fid is not None else None,
        )
    return targets


def build_row_group_tasks(parquet_files: list[Path], max_row_groups: int) -> list[tuple[str, int]]:
    tasks: list[tuple[str, int]] = []
    for path in parquet_files:
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.num_row_groups):
            tasks.append((str(path), rg_idx))
            if max_row_groups and len(tasks) >= max_row_groups:
                return tasks
    return tasks


def top_level_columns(parquet_path: Path) -> set[str]:
    """Return top-level Parquet column names.

    ``ParquetFile.schema.names`` exposes nested leaf names for list columns
    (often repeated as ``element``). ``schema_arrow.names`` is the top-level
    schema used by ``read_row_group(columns=[...])``.
    """
    return set(pq.ParquetFile(parquet_path).schema_arrow.names)


def validate_target_columns(parquet_files: list[Path], targets: dict[str, Target]) -> None:
    if not parquet_files:
        return
    columns = top_level_columns(parquet_files[0])
    required = {"label_type", *(target.feature_col for target in targets.values())}
    missing = sorted(required - columns)
    if not missing:
        return

    seq_like = sorted(
        name for name in columns
        if name.startswith("domain_") or name.startswith("seq_")
    )
    preview = ", ".join(seq_like[:40])
    raise SystemExit(
        "missing required parquet columns: "
        + ", ".join(missing)
        + "\nTop-level sequence-like columns preview: "
        + (preview or "<none>")
    )


def split_tasks(tasks: list[tuple[str, int]], batch_size: int) -> list[list[tuple[str, int]]]:
    batch_size = max(1, batch_size)
    return [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]


def list_array_offsets_values(array: pa.ChunkedArray | pa.Array) -> tuple[np.ndarray, np.ndarray]:
    arr = array.combine_chunks() if isinstance(array, pa.ChunkedArray) else array
    # Parquet list columns are ListArray/LargeListArray. The explicit combine
    # keeps offsets contiguous within this row group.
    offsets = arr.offsets.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    values = arr.values.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    return offsets, values


def position_bucket_name(pos: int) -> str:
    for name, start, end in POS_BUCKETS:
        if pos >= start and (end is None or pos < end):
            return name
    return POS_BUCKETS[-1][0]


def prune_counter(counter: collections.Counter[int], capacity: int) -> None:
    if capacity <= 0 or len(counter) <= capacity:
        return
    keep = heapq.nlargest(capacity, counter.items(), key=lambda kv: kv[1])
    counter.clear()
    counter.update(dict(keep))


def merge_counters_bounded(
    dst: collections.Counter[int],
    src: collections.Counter[int],
    capacity: int,
) -> None:
    dst.update(src)
    prune_counter(dst, capacity)


def hist_quantile(hist: collections.Counter[int], q: float) -> int:
    if not hist:
        return 0
    total = sum(hist.values())
    threshold = max(1, math.ceil(total * q))
    running = 0
    for value in sorted(hist):
        running += hist[value]
        if running >= threshold:
            return int(value)
    return int(max(hist))


def pass1_worker(
    tasks: list[tuple[str, int]],
    target_payload: dict[str, dict[str, Any]],
    candidate_capacity: int,
    local_prune_factor: float,
) -> dict[str, Any]:
    targets = {
        key: Target(**payload)
        for key, payload in target_payload.items()
    }
    columns = sorted({
        "label_type",
        *(target.feature_col for target in targets.values()),
    })
    stats = {key: BasicStats() for key in targets}
    counters = {key: collections.Counter() for key in targets}
    prune_at = max(candidate_capacity, int(candidate_capacity * local_prune_factor))

    for parquet_path, rg_idx in tasks:
        pf = pq.ParquetFile(parquet_path)
        available = set(pf.schema_arrow.names)
        read_cols = [c for c in columns if c in available]
        table = pf.read_row_group(rg_idx, columns=read_cols)
        labels = table["label_type"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        pos_mask = labels == 2

        for key, target in targets.items():
            if target.feature_col not in table.column_names:
                continue
            st = stats[key]
            st.rows += len(labels)
            st.pos_rows += int(pos_mask.sum())
            offsets, values = list_array_offsets_values(table[target.feature_col])

            local_counter = counters[key]
            for row_idx in range(len(labels)):
                start = int(offsets[row_idx])
                end = int(offsets[row_idx + 1])
                raw_len = max(0, end - start)
                is_pos = bool(pos_mask[row_idx])
                if raw_len <= 0:
                    st.update_lengths(0, 0, is_pos)
                    continue

                row_values = values[start:end]
                nonzero = row_values[row_values > 0]
                nz_len = int(nonzero.size)
                st.update_lengths(raw_len, nz_len, is_pos)
                st.total_tokens += raw_len
                st.nonzero_tokens += nz_len
                if is_pos:
                    st.pos_tokens += raw_len
                    st.nonzero_pos_tokens += nz_len

                if nz_len:
                    local_counter.update(int(v) for v in nonzero)
                    if is_pos:
                        for pos, value in enumerate(row_values):
                            if value <= 0:
                                continue
                            bucket = position_bucket_name(pos)
                            st.pos_bucket_total[bucket] += 1
                            st.pos_bucket_pos[bucket] += 1
                    else:
                        for pos, value in enumerate(row_values):
                            if value <= 0:
                                continue
                            st.pos_bucket_total[position_bucket_name(pos)] += 1

            if len(local_counter) > prune_at:
                prune_counter(local_counter, candidate_capacity)

    for counter in counters.values():
        prune_counter(counter, candidate_capacity)

    return {
        "stats": stats,
        "counters": counters,
    }


def pass2_worker(
    tasks: list[tuple[str, int]],
    target_payload: dict[str, dict[str, Any]],
    candidate_payload: dict[str, list[int]],
) -> dict[str, ExactStats]:
    targets = {
        key: Target(**payload)
        for key, payload in target_payload.items()
    }
    candidate_sets = {
        key: set(int(v) for v in values)
        for key, values in candidate_payload.items()
    }
    columns = sorted({
        "label_type",
        *(target.feature_col for target in targets.values()),
    })
    stats = {key: ExactStats() for key in targets}

    for parquet_path, rg_idx in tasks:
        pf = pq.ParquetFile(parquet_path)
        available = set(pf.schema_arrow.names)
        read_cols = [c for c in columns if c in available]
        table = pf.read_row_group(rg_idx, columns=read_cols)
        labels = table["label_type"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        pos_mask = labels == 2

        for key, target in targets.items():
            if target.feature_col not in table.column_names:
                continue
            candidate_ids = candidate_sets[key]
            exact = stats[key]
            offsets, values = list_array_offsets_values(table[target.feature_col])

            for row_idx in range(len(labels)):
                start = int(offsets[row_idx])
                end = int(offsets[row_idx + 1])
                if end <= start:
                    continue
                is_pos = bool(pos_mask[row_idx])
                for pos, raw_value in enumerate(values[start:end]):
                    value = int(raw_value)
                    if value <= 0:
                        continue
                    bucket = position_bucket_name(pos)
                    if value in candidate_ids:
                        exact.head_counts[value] += 1
                        exact.head_pos_bucket_counts[bucket][value] += 1
                        if is_pos:
                            exact.head_pos_counts[value] += 1
                    else:
                        exact.tail_count += 1
                        exact.tail_pos_bucket_total[bucket] += 1
                        if is_pos:
                            exact.tail_pos_count += 1
                            exact.tail_pos_bucket_pos[bucket] += 1

    return stats


def pct(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def lift(rate: float, base: float) -> float:
    return float(rate / base) if base else 0.0


def summarize_target(
    target: Target,
    basic: BasicStats,
    exact: ExactStats,
    topks: list[int],
) -> dict[str, Any]:
    sorted_counts = exact.head_counts.most_common()
    total = basic.nonzero_tokens
    base_pos_rate = pct(basic.nonzero_pos_tokens, basic.nonzero_tokens)

    topk_summary: dict[str, Any] = {}
    cumulative = 0
    cumulative_pos = 0
    cumulative_pos0_50 = 0
    for rank, (value, count) in enumerate(sorted_counts, start=1):
        cumulative += count
        cumulative_pos += exact.head_pos_counts.get(value, 0)
        for bucket_name in ("pos0_10", "pos10_50"):
            cumulative_pos0_50 += exact.head_pos_bucket_counts[bucket_name].get(value, 0)
        if rank in topks:
            topk_summary[f"top{rank}"] = {
                "count": int(cumulative),
                "coverage": pct(cumulative, total),
                "pos_rate": pct(cumulative_pos, cumulative),
                "pos0_50_coverage": pct(cumulative_pos0_50, total),
            }

    # Fill requested TopK values that exceed candidate size with final cumulative.
    pos0_50_final = 0
    for bucket_name in ("pos0_10", "pos10_50"):
        pos0_50_final += sum(exact.head_pos_bucket_counts[bucket_name].values())
    for k in topks:
        name = f"top{k}"
        if name not in topk_summary:
            topk_summary[name] = {
                "count": int(cumulative),
                "coverage": pct(cumulative, total),
                "pos_rate": pct(cumulative_pos, cumulative),
                "pos0_50_coverage": pct(pos0_50_final, total),
            }

    recommended_k = 0
    for k in topks:
        cov = topk_summary[f"top{k}"]["coverage"]
        if cov >= 0.90:
            recommended_k = k
            break
    if recommended_k == 0:
        for k in topks:
            cov = topk_summary[f"top{k}"]["coverage"]
            if cov >= 0.70:
                recommended_k = k
                break
    if recommended_k == 0:
        recommended_k = topks[-1]

    rec = topk_summary[f"top{recommended_k}"]
    tail_rate = pct(exact.tail_pos_count, exact.tail_count)
    head_tail_lift = lift(rec["pos_rate"], tail_rate)
    signal = abs(head_tail_lift - 1.0)
    rescue_score = (
        rec["coverage"]
        * (1.0 + min(1.0, rec["pos0_50_coverage"] * 2.0))
        * (0.25 + signal)
        * math.log1p(max(0, total))
    )

    return {
        "domain": target.domain,
        "fid": target.fid,
        "vocab_size": target.vocab_size,
        "rows": basic.rows,
        "pos_rate_rows": pct(basic.pos_rows, basic.rows),
        "rows_present_rate": pct(basic.rows_present, basic.rows),
        "rows_present_pos_rate": pct(basic.pos_rows_present, basic.rows_present),
        "raw_len_p50": hist_quantile(basic.raw_len_hist, 0.50),
        "raw_len_p90": hist_quantile(basic.raw_len_hist, 0.90),
        "raw_len_p99": hist_quantile(basic.raw_len_hist, 0.99),
        "nonzero_len_p50": hist_quantile(basic.nonzero_len_hist, 0.50),
        "nonzero_len_p90": hist_quantile(basic.nonzero_len_hist, 0.90),
        "nonzero_len_p99": hist_quantile(basic.nonzero_len_hist, 0.99),
        "total_tokens": basic.total_tokens,
        "nonzero_tokens": basic.nonzero_tokens,
        "nonzero_token_rate": pct(basic.nonzero_tokens, basic.total_tokens),
        "token_pos_rate": base_pos_rate,
        "unique_candidates": len(exact.head_counts),
        "tail_count": exact.tail_count,
        "tail_share": pct(exact.tail_count, total),
        "tail_pos_rate": tail_rate,
        "topk": topk_summary,
        "recommended_k": recommended_k,
        "recommended_coverage": rec["coverage"],
        "recommended_head_pos_rate": rec["pos_rate"],
        "recommended_head_tail_lift": head_tail_lift,
        "recommended_pos0_50_coverage": rec["pos0_50_coverage"],
        "position_bucket_total": dict(basic.pos_bucket_total),
        "tail_position_bucket_total": dict(exact.tail_pos_bucket_total),
        "tail_position_bucket_pos": dict(exact.tail_pos_bucket_pos),
        "rescue_score": rescue_score,
        "top_ids_preview": [
            {
                "id": int(value),
                "count": int(count),
                "coverage": pct(count, total),
                "pos_rate": pct(exact.head_pos_counts.get(value, 0), count),
            }
            for value, count in sorted_counts[:20]
        ],
    }


def parse_export_map_targets(specs: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for raw_spec in specs.split(","):
        spec = raw_spec.strip()
        if not spec:
            continue
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"invalid --export-map-targets spec {spec!r}; "
                "expected domain:fid:k"
            )
        domain, fid_s, k_s = parts
        key = f"{domain}:{int(fid_s)}"
        k = int(k_s)
        if k <= 0:
            raise ValueError(f"invalid k for {spec!r}; k must be positive")
        result[key] = k
    return result


def build_topk_map(
    targets: dict[str, Target],
    exact_stats: dict[str, ExactStats],
    export_specs: dict[str, int],
    candidate_capacity: int,
) -> dict[str, Any]:
    exported: dict[str, Any] = {}
    warnings: list[str] = []
    for key, k in sorted(export_specs.items()):
        if key not in targets:
            raise ValueError(f"export target {key!r} is not in --targets")
        target = targets[key]
        if k > candidate_capacity:
            warnings.append(
                f"{key}: requested k={k} exceeds candidate_capacity={candidate_capacity}; "
                "increase --candidate-capacity for a reliable export"
            )
        top_items = exact_stats[key].head_counts.most_common(k)
        ids = [int(value) for value, _ in top_items]
        if len(ids) < k:
            warnings.append(
                f"{key}: requested k={k}, but only {len(ids)} candidate ids were available"
            )
        exported[key] = {
            "domain": target.domain,
            "fid": target.fid,
            "vocab_size": target.vocab_size,
            "k": int(k),
            "actual_k": len(ids),
            "padding_id": 0,
            "default_id": len(ids) + 1,
            "ids": ids,
        }
    return {
        "version": 1,
        "format": "taac_seq_topk_default_map",
        "description": (
            "For each target, remap raw ids in ids[] to 1..actual_k by rank; "
            "raw id 0 stays padding_id=0; positive ids not in ids[] map to default_id."
        ),
        "targets": exported,
        "warnings": warnings,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# TopK Tail EDA",
        "",
        f"- rows/row_groups scanned: {result['row_groups_scanned']} row groups",
        f"- workers: {result['workers']} ({result['executor']})",
        f"- candidate_capacity: {result['candidate_capacity']:,}",
        "",
        "## Rescue Ranking",
        "",
        "| rank | domain | fid | vocab | nonzero_tokens | top100k_cov | top200k_cov | tail_share | head_tail_lift | pos0_50_top100k | recommended_k | rescue_score |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summaries = sorted(
        result["targets"],
        key=lambda row: row["rescue_score"],
        reverse=True,
    )
    for idx, row in enumerate(summaries, start=1):
        top100 = row["topk"].get("top100000", {})
        top200 = row["topk"].get("top200000", {})
        lines.append(
            "| {rank} | {domain} | {fid} | {vocab:,} | {tokens:,} | {top100:.2%} | "
            "{top200:.2%} | {tail:.2%} | {lift:.3f}x | {pos50:.2%} | {rk:,} | {score:.3f} |".format(
                rank=idx,
                domain=row["domain"],
                fid=row["fid"],
                vocab=row["vocab_size"],
                tokens=row["nonzero_tokens"],
                top100=top100.get("coverage", 0.0),
                top200=top200.get("coverage", 0.0),
                tail=row["tail_share"],
                lift=row["recommended_head_tail_lift"],
                pos50=top100.get("pos0_50_coverage", 0.0),
                rk=row["recommended_k"],
                score=row["rescue_score"],
            )
        )

    lines.extend([
        "",
        "## Per-Feature Details",
        "",
    ])
    for row in summaries:
        lines.extend([
            f"### {row['domain']} fid {row['fid']}",
            "",
            f"- vocab_size: {row['vocab_size']:,}",
            f"- nonzero_tokens: {row['nonzero_tokens']:,} ({row['nonzero_token_rate']:.2%} of raw tokens)",
            f"- raw_len p50/p90/p99: {row['raw_len_p50']} / {row['raw_len_p90']} / {row['raw_len_p99']}",
            f"- recommended_k: {row['recommended_k']:,}",
            f"- recommended coverage: {row['recommended_coverage']:.2%}",
            f"- head/tail pos-rate lift: {row['recommended_head_tail_lift']:.3f}x",
            "",
            "| k | coverage | pos_rate | pos0_50_coverage |",
            "|---:|---:|---:|---:|",
        ])
        for name, top in row["topk"].items():
            lines.append(
                f"| {name[3:]} | {top['coverage']:.2%} | "
                f"{top['pos_rate']:.3%} | {top['pos0_50_coverage']:.2%} |"
            )
        lines.extend([
            "",
            "| id | count | coverage | pos_rate |",
            "|---:|---:|---:|---:|",
        ])
        for item in row["top_ids_preview"][:10]:
            lines.append(
                f"| {item['id']} | {item['count']:,} | "
                f"{item['coverage']:.3%} | {item['pos_rate']:.3%} |"
            )
        lines.append("")
    return "\n".join(lines)


def log_progress(message: str) -> None:
    """Emit one platform-friendly progress line."""
    print(f"[topk-tail-eda] {message}", flush=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    data_dir = Path(args.data_dir)
    schema = load_schema(Path(args.schema_path))
    targets = build_targets(schema, args.targets.split(","))
    parquet_files = list_parquet_files(data_dir)
    if not parquet_files:
        raise SystemExit(f"no parquet files found under {data_dir}")
    validate_target_columns(parquet_files, targets)

    tasks = build_row_group_tasks(parquet_files, args.max_row_groups)
    if not tasks:
        raise SystemExit("no row groups found")
    chunks = split_tasks(tasks, args.row_group_batch_size)
    worker_count = min(args.workers, len(chunks))
    log_progress(
        "start "
        f"files={len(parquet_files)} row_groups={len(tasks)} "
        f"tasks={len(chunks)} row_group_batch_size={args.row_group_batch_size} "
        f"targets={len(targets)} workers={worker_count} executor={args.executor} "
        f"candidate_capacity={args.candidate_capacity}"
    )
    target_payload = {
        key: {
            "domain": target.domain,
            "fid": target.fid,
            "vocab_size": target.vocab_size,
            "feature_col": target.feature_col,
            "ts_col": target.ts_col,
        }
        for key, target in targets.items()
    }
    executor_cls = futures.ProcessPoolExecutor if args.executor == "process" else futures.ThreadPoolExecutor

    merged_basic = {key: BasicStats() for key in targets}
    merged_candidates = {key: collections.Counter() for key in targets}
    pass1_start = time.perf_counter()
    log_progress("pass1_candidate_scan begin")
    with executor_cls(max_workers=worker_count) as pool:
        futs = [
            pool.submit(
                pass1_worker,
                chunk,
                target_payload,
                args.candidate_capacity,
                args.local_prune_factor,
            )
            for chunk in chunks
        ]
        done = 0
        for fut in futures.as_completed(futs):
            payload = fut.result()
            for key, st in payload["stats"].items():
                merged_basic[key].merge(st)
            for key, counter in payload["counters"].items():
                merge_counters_bounded(
                    merged_candidates[key],
                    counter,
                    args.candidate_capacity,
                )
            done += 1
            log_progress(
                f"pass1_candidate_scan progress tasks_done={done}/{len(futs)} "
                f"elapsed_sec={time.perf_counter() - pass1_start:.1f}"
            )

    candidate_payload = {
        key: [int(value) for value, _ in counter.most_common(args.candidate_capacity)]
        for key, counter in merged_candidates.items()
    }
    candidate_counts = ", ".join(
        f"{key}={len(values)}" for key, values in sorted(candidate_payload.items())
    )
    log_progress(
        f"pass1_candidate_scan done elapsed_sec={time.perf_counter() - pass1_start:.1f} "
        f"candidates[{candidate_counts}]"
    )

    merged_exact = {key: ExactStats() for key in targets}
    pass2_start = time.perf_counter()
    log_progress("pass2_exact_recount begin")
    with executor_cls(max_workers=worker_count) as pool:
        futs = [
            pool.submit(pass2_worker, chunk, target_payload, candidate_payload)
            for chunk in chunks
        ]
        done = 0
        for fut in futures.as_completed(futs):
            payload = fut.result()
            for key, st in payload.items():
                merged_exact[key].merge(st)
            done += 1
            log_progress(
                f"pass2_exact_recount progress tasks_done={done}/{len(futs)} "
                f"elapsed_sec={time.perf_counter() - pass2_start:.1f}"
            )
    log_progress(f"pass2_exact_recount done elapsed_sec={time.perf_counter() - pass2_start:.1f}")

    topks = [int(v) for v in args.topks.split(",") if v.strip()]
    topks = sorted(set(topks))
    summaries = [
        summarize_target(targets[key], merged_basic[key], merged_exact[key], topks)
        for key in sorted(targets)
    ]
    topk_map = None
    if args.export_map_targets:
        export_specs = parse_export_map_targets(args.export_map_targets)
        topk_map = build_topk_map(
            targets=targets,
            exact_stats=merged_exact,
            export_specs=export_specs,
            candidate_capacity=args.candidate_capacity,
        )
        log_progress(
            "topk_map built "
            f"targets={len(topk_map['targets'])} warnings={len(topk_map['warnings'])}"
        )
    log_progress(f"summarize done total_elapsed_sec={time.perf_counter() - started_at:.1f}")
    result = {
        "data_dir": str(data_dir),
        "schema_path": str(args.schema_path),
        "targets_requested": list(targets),
        "row_groups_scanned": len(tasks),
        "tasks": len(chunks),
        "row_group_batch_size": args.row_group_batch_size,
        "workers": args.workers,
        "executor": args.executor,
        "candidate_capacity": args.candidate_capacity,
        "topks": topks,
        "elapsed_sec": time.perf_counter() - started_at,
        "targets": summaries,
    }
    if topk_map is not None:
        result["topk_map"] = topk_map
    return result


def final_summary_line(result: dict[str, Any], out_json: Path, out_md: Path) -> str:
    ranked = sorted(
        result["targets"],
        key=lambda row: row["rescue_score"],
        reverse=True,
    )
    top = [
        {
            "target": f"{row['domain']}:{row['fid']}",
            "recommended_k": row["recommended_k"],
            "coverage": round(row["recommended_coverage"], 6),
            "head_tail_lift": round(row["recommended_head_tail_lift"], 6),
            "score": round(row["rescue_score"], 6),
        }
        for row in ranked[:5]
    ]
    payload = {
        "row_groups": result["row_groups_scanned"],
        "elapsed_sec": round(result.get("elapsed_sec", 0.0), 1),
        "top": top,
        "json": str(out_json),
        "md": str(out_md),
    }
    if "topk_map_path" in result:
        payload["topk_map"] = result["topk_map_path"]
    return "TOPK_TAIL_EDA_DONE " + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def print_block(marker: str, text: str) -> None:
    print(f"{marker}_BEGIN", flush=True)
    for line in text.splitlines():
        print(line, flush=True)
    print(f"{marker}_END", flush=True)


def print_one_line_file(marker: str, text: str) -> None:
    print(
        marker + " " + json.dumps(text, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )


def print_one_line_json(marker: str, payload: Any) -> None:
    print(
        marker + " " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    result = run(args)

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_topk_map = Path(args.export_topk_map) if args.export_topk_map else None
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    if out_topk_map is not None:
        out_topk_map.parent.mkdir(parents=True, exist_ok=True)
        if "topk_map" not in result:
            raise SystemExit("--export-topk-map requires --export-map-targets")
        with out_topk_map.open("w", encoding="utf-8") as f:
            json.dump(result["topk_map"], f, indent=2, ensure_ascii=False)
        result["topk_map_path"] = str(out_topk_map)
        log_progress(f"wrote_topk_map path={out_topk_map}")
        for warning in result["topk_map"].get("warnings", []):
            log_progress(f"topk_map_warning {warning}")
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    md_report = render_markdown(result)
    out_md.write_text(md_report, encoding="utf-8")
    log_progress(f"wrote_json path={out_json}")
    log_progress(f"wrote_md path={out_md}")
    print(final_summary_line(result, out_json, out_md), flush=True)
    if args.print_md_one_line:
        print_one_line_file("TOPK_TAIL_EDA_MARKDOWN_FILE", md_report)
    if args.print_md:
        print_block("TOPK_TAIL_EDA_MARKDOWN", md_report)
    if args.print_json:
        print_block(
            "TOPK_TAIL_EDA_JSON",
            json.dumps(result, indent=2, ensure_ascii=False),
        )
    if args.print_topk_map_one_line and "topk_map" in result:
        print_one_line_json(
            "TOPK_TAIL_EDA_TOPK_MAP_FILE",
            result["topk_map"],
        )


if __name__ == "__main__":
    main()
