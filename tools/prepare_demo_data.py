"""Prepare the TAAC2026 Hugging Face demo dataset for local training.

The training code expects a directory containing parquet files plus a
``schema.json`` that records feature dimensions and vocab sizes. The public
``TAAC2026/data_sample_1000`` dataset is enough for a smoke test, but it does
not ship that project-specific schema. This script downloads the sample,
writes it as a multi-row-group parquet file, and infers a conservative schema
from the observed values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


DATASET_ID = "TAAC2026/data_sample_1000"

SCHEMA_GROUPS = {
    "user_int_scalar": [1, 3, 4, *range(48, 60), 82, 86, *range(92, 110)],
    "user_int_array": [15, 60, *range(62, 67), 80, *range(89, 92)],
    "user_dense": [61, *range(62, 67), 87, *range(89, 92)],
    "item_int_scalar": [*range(5, 11), 12, 13, 16, 81, *range(83, 86)],
    "item_int_array": [11],
    "seq": {
        "seq_a": ("domain_a_seq", list(range(38, 47))),
        "seq_b": ("domain_b_seq", [*range(67, 80), 88]),
        "seq_c": ("domain_c_seq", [*range(27, 38), 47]),
        "seq_d": ("domain_d_seq", list(range(17, 27))),
    },
}


def _is_list_like(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray))


def _flatten(values: Iterable[Any]) -> list[Any]:
    flat: list[Any] = []
    for value in values:
        if _is_list_like(value):
            flat.extend(list(value))
        elif pd.notna(value):
            flat.append(value)
    return flat


def _max_list_len(series: pd.Series) -> int:
    lengths = [len(x) for x in series.dropna() if _is_list_like(x)]
    return max(lengths, default=1)


def _vocab_size(series: pd.Series) -> int:
    values = _flatten(series.dropna())
    if not values:
        return 1
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if arr.empty:
        return 1
    # Dataset code treats valid ids as 0 <= id < vocab_size, so max+1 keeps
    # the largest observed anonymous id in range. Embedding tables add one more
    # slot internally, preserving 0 as padding.
    return max(1, int(arr.max()) + 1)


def build_schema(df: pd.DataFrame) -> dict[str, Any]:
    user_int = []
    for fid in SCHEMA_GROUPS["user_int_scalar"]:
        col = f"user_int_feats_{fid}"
        user_int.append([fid, _vocab_size(df[col]), 1])
    for fid in SCHEMA_GROUPS["user_int_array"]:
        col = f"user_int_feats_{fid}"
        user_int.append([fid, _vocab_size(df[col]), _max_list_len(df[col])])

    item_int = []
    for fid in SCHEMA_GROUPS["item_int_scalar"]:
        col = f"item_int_feats_{fid}"
        item_int.append([fid, _vocab_size(df[col]), 1])
    for fid in SCHEMA_GROUPS["item_int_array"]:
        col = f"item_int_feats_{fid}"
        item_int.append([fid, _vocab_size(df[col]), _max_list_len(df[col])])

    user_dense = []
    for fid in SCHEMA_GROUPS["user_dense"]:
        col = f"user_dense_feats_{fid}"
        user_dense.append([fid, _max_list_len(df[col])])

    seq = {}
    for domain, (prefix, fids) in SCHEMA_GROUPS["seq"].items():
        features = []
        for fid in fids:
            col = f"{prefix}_{fid}"
            features.append([fid, _vocab_size(df[col])])
        seq[domain] = {
            "prefix": prefix,
            # The public schema says sequence side information may include
            # timestamps, but released column names are anonymized. Keep this
            # unknown for the demo so time-bucket ids stay zero.
            "ts_fid": None,
            "features": features,
        }

    return {
        "user_int": user_int,
        "item_int": item_int,
        "user_dense": user_dense,
        "seq": seq,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TAAC2026 demo data")
    parser.add_argument("--out_dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--dataset_id", type=str, default=DATASET_ID)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--row_group_size", type=int, default=200)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(args.dataset_id, split=args.split)
    df = ds.to_pandas()

    table = pa.Table.from_pandas(df, preserve_index=False)
    parquet_path = args.out_dir / "sample.parquet"
    pq.write_table(table, parquet_path, row_group_size=args.row_group_size)

    schema = build_schema(df)
    schema_path = args.out_dir / "schema.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    print(f"wrote {parquet_path} ({len(df)} rows)")
    print(f"wrote {schema_path}")


if __name__ == "__main__":
    main()
