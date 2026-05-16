#!/usr/bin/env python3
"""Build a compact seq topK rescue map from a saved EDA export."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def parse_targets(specs: str) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for raw_spec in specs.split(","):
        spec = raw_spec.strip()
        if not spec:
            continue
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"invalid target spec {spec!r}; expected domain:fid:k")
        domain, fid_s, k_s = parts
        k = int(k_s)
        if k <= 0:
            raise ValueError(f"invalid k in {spec!r}; k must be positive")
        result[f"{domain}:{int(fid_s)}"] = k
    return result


def extract_source_map(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload.get("topk_map"), dict):
        return payload["topk_map"]
    if isinstance(payload.get("targets"), dict):
        return payload
    if isinstance(payload.get("targets"), list):
        raise ValueError(
            "input is a plain EDA report without full topK ids. Rerun "
            "topk_tail_eda.py with --export-map-targets and --export-topk-map.")
    raise ValueError("input JSON does not contain a usable topk_map/targets object")


def build_map(source_map: Dict[str, Any], requested: Dict[str, int]) -> Dict[str, Any]:
    source_targets = source_map.get("targets")
    if not isinstance(source_targets, dict):
        raise ValueError("source map has no targets object")

    exported: Dict[str, Any] = {}
    warnings = list(source_map.get("warnings", []))
    for key, k in sorted(requested.items()):
        if key not in source_targets:
            raise KeyError(f"target {key!r} is missing from source map")
        cfg = dict(source_targets[key])
        ids = [int(v) for v in cfg.get("ids", [])]
        if len(ids) < k:
            raise ValueError(
                f"target {key!r} requested k={k}, but source map only has "
                f"{len(ids)} ids")
        ids = ids[:k]
        cfg["k"] = int(k)
        cfg["actual_k"] = len(ids)
        cfg["padding_id"] = 0
        cfg["default_id"] = len(ids) + 1
        cfg["ids"] = ids
        exported[key] = cfg

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build seq topK rescue map")
    parser.add_argument("--eda-json", required=True,
                        help="topk_tail_eda.json or source topk map JSON")
    parser.add_argument("--targets", required=True,
                        help="Comma-separated specs, e.g. seq_c:34:10000,seq_c:29:200000")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--print-one-line", action="store_true")
    args = parser.parse_args()

    with open(args.eda_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    result = build_map(extract_source_map(payload), parse_targets(args.targets))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    summary = {
        key: {
            "actual_k": cfg["actual_k"],
            "default_id": cfg["default_id"],
            "vocab_size_for_model": cfg["default_id"],
        }
        for key, cfg in result["targets"].items()
    }
    print(
        "TOPK_RESCUE_MAP_DONE "
        + json.dumps({"out_json": str(out_path), "targets": summary},
                     ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )
    if args.print_one_line:
        print(
            "TOPK_RESCUE_MAP_FILE "
            + json.dumps(result, ensure_ascii=False, separators=(",", ":")),
            flush=True,
        )


if __name__ == "__main__":
    main()
