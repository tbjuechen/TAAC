#!/usr/bin/env python3
"""EDA for per-domain sequence time buckets.

The current model uses one shared recency bucket embedding for all sequence
domains. This script samples parquet rows and compares each domain's
``row_timestamp - seq_timestamp`` and adjacent-token ``delta_t`` distributions.

Example:
    python tools/time_bucket_eda.py \
        --data-dir "$TRAIN_DATA_PATH" \
        --schema-path "$TRAIN_DATA_PATH/schema.json" \
        --max-rows 200000 \
        --out-md output/time_bucket_eda.md \
        --out-json output/time_bucket_eda.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


RECENCY_BOUNDARIES = np.array([
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

DELTA_BOUNDARIES = np.array([
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
    4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288,
    1048576, 2097152, 4194304, 8388608, 16777216, 33554432,
    67108864, 134217728, 268435456, 536870912, 1073741824,
], dtype=np.int64)

QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def _list_parquet_files(data_dir: Path) -> list[Path]:
    if data_dir.is_file():
        return [data_dir]
    return sorted(data_dir.glob("*.parquet"))


def _load_schema(schema_path: Path) -> dict[str, Any]:
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_int_array(value: Any) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=np.int64)
    if isinstance(value, np.ndarray):
        return value.astype(np.int64, copy=False)
    if isinstance(value, list):
        return np.asarray(value, dtype=np.int64)
    return np.asarray(list(value), dtype=np.int64)


def _bucket_counts(values: np.ndarray, boundaries: np.ndarray, clip_upper: bool) -> list[int]:
    if values.size == 0:
        return [0] * (len(boundaries) + (1 if clip_upper else 2))
    if clip_upper:
        raw = np.clip(np.searchsorted(boundaries, values), 0, len(boundaries) - 1)
        bucket_ids = raw + 1
        n_buckets = len(boundaries) + 1
    else:
        raw = np.clip(np.searchsorted(boundaries, values), 0, len(boundaries))
        bucket_ids = raw + 1
        n_buckets = len(boundaries) + 2
    return np.bincount(bucket_ids, minlength=n_buckets).astype(int).tolist()


def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def _js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    a_sum = a.sum()
    b_sum = b.sum()
    if a_sum <= 0 or b_sum <= 0:
        return 0.0
    p = a / a_sum
    q = b / b_sum
    m = 0.5 * (p + q)

    def kl(x: np.ndarray, y: np.ndarray) -> float:
        mask = x > 0
        return float((x[mask] * np.log2(x[mask] / y[mask])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _summarize_values(values: np.ndarray, boundaries: np.ndarray, clip_upper: bool) -> dict[str, Any]:
    if values.size == 0:
        return {
            "count": 0,
            "quantiles": {},
            "bucket_counts": [],
            "empty_buckets": 0,
            "top_bucket_share": 0.0,
            "entropy_bits": 0.0,
            "suggested_quantile_boundaries": [],
        }

    counts = np.asarray(_bucket_counts(values, boundaries, clip_upper), dtype=np.int64)
    non_padding = counts[1:]
    total = int(non_padding.sum())
    qs = np.quantile(values, QUANTILES)
    suggested = np.unique(np.maximum(1, np.round(
        np.quantile(values, np.linspace(0.02, 0.98, len(boundaries)))
    ).astype(np.int64)))
    return {
        "count": int(values.size),
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean()),
        "quantiles": {f"p{int(q * 100):02d}": int(v) for q, v in zip(QUANTILES, qs)},
        "bucket_counts": counts.astype(int).tolist(),
        "empty_buckets": int((non_padding == 0).sum()),
        "top_bucket_share": float(non_padding.max() / total) if total else 0.0,
        "entropy_bits": _entropy(non_padding),
        "suggested_quantile_boundaries": suggested.astype(int).tolist(),
    }


def _format_seconds(seconds: int | float) -> str:
    seconds = float(seconds)
    if seconds < 120:
        return f"{seconds:.0f}s"
    if seconds < 7200:
        return f"{seconds / 60:.1f}m"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    schema_path = Path(args.schema_path)
    schema = _load_schema(schema_path)
    parquet_files = _list_parquet_files(data_dir)
    if not parquet_files:
        raise SystemExit(f"no parquet files found under {data_dir}")

    seq_cfg = schema["seq"]
    domains = sorted(seq_cfg.keys())
    ts_cols = {
        d: f"{seq_cfg[d]['prefix']}_{seq_cfg[d]['ts_fid']}"
        for d in domains
        if seq_cfg[d].get("ts_fid") is not None
    }
    columns = ["timestamp", "label_type", *ts_cols.values()]

    recency_values: dict[str, list[np.ndarray]] = {d: [] for d in domains}
    delta_values: dict[str, list[np.ndarray]] = {d: [] for d in domains}
    pos0_recency: dict[str, list[int]] = {d: [] for d in domains}
    rows_seen = 0
    pos_rows = 0

    for parquet_path in parquet_files:
        pf = pq.ParquetFile(parquet_path)
        for rg_idx in range(pf.num_row_groups):
            if args.max_rows and rows_seen >= args.max_rows:
                break
            table = pf.read_row_group(rg_idx, columns=[c for c in columns if c in pf.schema.names])
            df = table.to_pandas()
            if args.max_rows:
                df = df.iloc[: max(0, args.max_rows - rows_seen)]
            if df.empty:
                continue
            timestamps = df["timestamp"].to_numpy(dtype=np.int64)
            labels = df["label_type"].to_numpy()
            pos_rows += int((labels == 2).sum())

            for domain, col in ts_cols.items():
                if col not in df:
                    continue
                rec_parts: list[np.ndarray] = []
                delta_parts: list[np.ndarray] = []
                for row_ts, seq_ts_raw in zip(timestamps, df[col].to_numpy()):
                    seq_ts = _as_int_array(seq_ts_raw)
                    seq_ts = seq_ts[seq_ts > 0]
                    if seq_ts.size == 0:
                        continue
                    rec = np.maximum(row_ts - seq_ts, 0)
                    rec_parts.append(rec)
                    pos0_recency[domain].append(int(rec[0]))
                    if seq_ts.size > 1:
                        delta = np.maximum(seq_ts[:-1] - seq_ts[1:], 0)
                        delta = delta[delta > 0]
                        if delta.size:
                            delta_parts.append(delta)
                if rec_parts:
                    recency_values[domain].append(np.concatenate(rec_parts))
                if delta_parts:
                    delta_values[domain].append(np.concatenate(delta_parts))

            rows_seen += len(df)
        if args.max_rows and rows_seen >= args.max_rows:
            break

    recency_arrays = {
        d: np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        for d, parts in recency_values.items()
    }
    delta_arrays = {
        d: np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        for d, parts in delta_values.items()
    }

    result: dict[str, Any] = {
        "data_dir": str(data_dir),
        "schema_path": str(schema_path),
        "rows_seen": rows_seen,
        "pos_rate": pos_rows / rows_seen if rows_seen else 0.0,
        "recency": {},
        "delta": {},
        "shared_bucket_js_divergence": {},
    }

    rec_bucket_arrays: dict[str, np.ndarray] = {}
    for domain in domains:
        rec_summary = _summarize_values(
            recency_arrays[domain], RECENCY_BOUNDARIES, clip_upper=True)
        delta_summary = _summarize_values(
            delta_arrays[domain], DELTA_BOUNDARIES, clip_upper=False)
        if pos0_recency[domain]:
            pos0 = np.asarray(pos0_recency[domain], dtype=np.int64)
            rec_summary["pos0_quantiles"] = {
                f"p{int(q * 100):02d}": int(v)
                for q, v in zip(QUANTILES, np.quantile(pos0, QUANTILES))
            }
        result["recency"][domain] = rec_summary
        result["delta"][domain] = delta_summary
        rec_bucket_arrays[domain] = np.asarray(rec_summary["bucket_counts"], dtype=np.float64)

    for i, left in enumerate(domains):
        for right in domains[i + 1:]:
            result["shared_bucket_js_divergence"][f"{left}__{right}"] = _js_divergence(
                rec_bucket_arrays[left][1:], rec_bucket_arrays[right][1:])

    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Time Bucket EDA",
        "",
        f"- data_dir: `{result['data_dir']}`",
        f"- rows_seen: {result['rows_seen']:,}",
        f"- pos_rate: {result['pos_rate']:.4%}",
        "",
        "## Recency By Domain",
        "",
        "| domain | tokens | p10 | p50 | p90 | p99 | pos0_p50 | empty shared buckets | top bucket | entropy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for domain, s in result["recency"].items():
        q = s.get("quantiles", {})
        p0 = s.get("pos0_quantiles", {})
        lines.append(
            f"| {domain} | {s['count']:,} | {_format_seconds(q.get('p10', 0))} | "
            f"{_format_seconds(q.get('p50', 0))} | {_format_seconds(q.get('p90', 0))} | "
            f"{_format_seconds(q.get('p99', 0))} | {_format_seconds(p0.get('p50', 0))} | "
            f"{s['empty_buckets']} | {s['top_bucket_share']:.2%} | {s['entropy_bits']:.2f} |"
        )

    lines.extend([
        "",
        "## Delta By Domain",
        "",
        "| domain | pairs | p10 | p50 | p90 | p99 | empty shared buckets | top bucket | entropy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for domain, s in result["delta"].items():
        q = s.get("quantiles", {})
        lines.append(
            f"| {domain} | {s['count']:,} | {_format_seconds(q.get('p10', 0))} | "
            f"{_format_seconds(q.get('p50', 0))} | {_format_seconds(q.get('p90', 0))} | "
            f"{_format_seconds(q.get('p99', 0))} | {s['empty_buckets']} | "
            f"{s['top_bucket_share']:.2%} | {s['entropy_bits']:.2f} |"
        )

    lines.extend([
        "",
        "## Shared Recency Bucket Divergence",
        "",
        "| pair | JS divergence bits |",
        "|---|---:|",
    ])
    for pair, jsd in sorted(result["shared_bucket_js_divergence"].items()):
        lines.append(f"| {pair} | {jsd:.4f} |")

    lines.extend([
        "",
        "## Suggested Per-Domain Recency Boundaries",
        "",
        "The lists below are quantile-derived candidates. Treat them as EDA output, not final constants.",
        "",
    ])
    for domain, s in result["recency"].items():
        boundaries = ", ".join(str(x) for x in s["suggested_quantile_boundaries"])
        lines.append(f"### {domain}")
        lines.append("")
        lines.append(f"`[{boundaries}]`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--schema-path", required=True)
    parser.add_argument("--max-rows", type=int, default=200_000,
                        help="maximum rows to sample; 0 means all rows")
    parser.add_argument("--out-md", default="output/time_bucket_eda.md")
    parser.add_argument("--out-json", default="output/time_bucket_eda.json")
    args = parser.parse_args()
    if args.max_rows == 0:
        args.max_rows = None

    result = run(args)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(result), encoding="utf-8")

    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
