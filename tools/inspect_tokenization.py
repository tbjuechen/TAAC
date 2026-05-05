"""Inspect TAAC baseline tokenization from schema metadata.

This is a static tokenizer audit: it does not read Parquet rows. It answers:

- Which user/item int fids land in each RankMixer chunk?
- Which large multi-hot fids are cut across chunk boundaries?
- Where are the v7 dense groups in the flattened user_dense vector?
- Which sequence fids map to the v7 item/action/stat roles?

Example:
    python tools/inspect_tokenization.py \
        --schema_path src/taac2026_schema.json \
        --out_md docs/eda/2026-05-05-tokenization-audit.md \
        --out_json docs/eda/2026-05-05-tokenization-audit.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


V7_DENSE_GROUPS = {
    "emb": [61, 87],
    "stat": [62, 63, 64, 65, 66],
    "quantile": [89, 90, 91],
}

V7_SEQ_ROLES = {
    "seq_a": {"item": [38], "action": [40], "stat": [42, 43, 44, 45]},
    "seq_b": {"item": [69], "action": [68], "stat": [70, 71, 72, 73, 74, 75, 76, 77, 78, 79]},
    "seq_c": {"item": [29], "action": [28], "stat": [30, 31, 32, 33, 34, 35, 36, 37]},
    "seq_d": {"item": [], "action": [17], "stat": [18, 19, 20, 21, 22, 23, 24, 25]},
}


@dataclass
class FeatureSpan:
    fid: int
    vocab_size: int
    length: int
    emb_start: int
    emb_end: int
    offset: int


@dataclass
class ChunkAssignment:
    chunk_id: int
    start: int
    end: int
    width: int
    fids: list[int]
    partial_fids: list[int]
    rows: list[dict[str, Any]]


def _build_int_spans(cols: list[list[int]], emb_dim: int) -> list[FeatureSpan]:
    spans: list[FeatureSpan] = []
    offset = 0
    emb_pos = 0
    for fid, vocab_size, length in cols:
        spans.append(
            FeatureSpan(
                fid=int(fid),
                vocab_size=int(vocab_size),
                length=int(length),
                emb_start=emb_pos,
                emb_end=emb_pos + emb_dim,
                offset=offset,
            )
        )
        offset += int(length)
        emb_pos += emb_dim
    return spans


def _rankmixer_chunks(
    spans: list[FeatureSpan],
    num_tokens: int,
) -> dict[str, Any]:
    total_emb_dim = spans[-1].emb_end if spans else 0
    chunk_dim = math.ceil(total_emb_dim / num_tokens) if num_tokens > 0 else 0
    padded_total_dim = chunk_dim * num_tokens
    pad_size = padded_total_dim - total_emb_dim
    chunks: list[ChunkAssignment] = []
    for chunk_id in range(num_tokens):
        start = chunk_id * chunk_dim
        end = start + chunk_dim
        rows = []
        fids = []
        partial_fids = []
        for span in spans:
            overlap_start = max(start, span.emb_start)
            overlap_end = min(end, span.emb_end)
            if overlap_start >= overlap_end:
                continue
            is_partial = overlap_start != span.emb_start or overlap_end != span.emb_end
            fids.append(span.fid)
            if is_partial:
                partial_fids.append(span.fid)
            rows.append(
                {
                    "fid": span.fid,
                    "vocab_size": span.vocab_size,
                    "length": span.length,
                    "feature_emb_span": [span.emb_start, span.emb_end],
                    "overlap": [overlap_start, overlap_end],
                    "partial": is_partial,
                    "raw_offset": span.offset,
                }
            )
        chunks.append(
            ChunkAssignment(
                chunk_id=chunk_id,
                start=start,
                end=end,
                width=chunk_dim,
                fids=fids,
                partial_fids=partial_fids,
                rows=rows,
            )
        )
    return {
        "total_emb_dim": total_emb_dim,
        "num_tokens": num_tokens,
        "chunk_dim": chunk_dim,
        "padded_total_dim": padded_total_dim,
        "pad_size": pad_size,
        "chunks": [asdict(c) for c in chunks],
    }


def _dense_offsets(cols: list[list[int]]) -> list[dict[str, int]]:
    rows = []
    offset = 0
    for fid, length in cols:
        rows.append(
            {
                "fid": int(fid),
                "length": int(length),
                "offset": offset,
                "end": offset + int(length),
            }
        )
        offset += int(length)
    return rows


def _dense_group_report(dense_rows: list[dict[str, int]]) -> dict[str, Any]:
    by_fid = {row["fid"]: row for row in dense_rows}
    groups: dict[str, Any] = {}
    missing: dict[str, list[int]] = {}
    for name, fids in V7_DENSE_GROUPS.items():
        rows = [by_fid[fid] for fid in fids if fid in by_fid]
        missing[name] = [fid for fid in fids if fid not in by_fid]
        groups[name] = {
            "fids": [row["fid"] for row in rows],
            "total_dim": sum(row["length"] for row in rows),
            "spans": rows,
        }
    return {"groups": groups, "missing": missing}


def _seq_role_report(raw_seq: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for domain, cfg in sorted(raw_seq.items()):
        features = {int(fid): int(vs) for fid, vs in cfg["features"]}
        ts_fid = cfg.get("ts_fid")
        sideinfo = [fid for fid in features if fid != ts_fid]
        roles = V7_SEQ_ROLES.get(domain, {})
        role_rows: dict[str, Any] = {}
        assigned: set[int] = set()
        for role, fids in roles.items():
            rows = []
            for fid in fids:
                rows.append(
                    {
                        "fid": fid,
                        "present": fid in features,
                        "vocab_size": features.get(fid),
                    }
                )
                assigned.add(fid)
            role_rows[role] = rows
        unassigned = [fid for fid in sideinfo if fid not in assigned]
        result[domain] = {
            "prefix": cfg.get("prefix"),
            "ts_fid": ts_fid,
            "num_sideinfo": len(sideinfo),
            "roles": role_rows,
            "unassigned_sideinfo": [
                {"fid": fid, "vocab_size": features[fid]} for fid in unassigned
            ],
        }
    return result


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# TAAC Tokenization Audit\n")
    lines.append(f"- schema: `{report['schema_path']}`")
    lines.append(f"- emb_dim: {report['emb_dim']}")
    lines.append(f"- d_model: {report['d_model']}")
    lines.append(f"- num_queries: {report['num_queries']}")
    lines.append(f"- user_ns_tokens: {report['user_ns_tokens']}")
    lines.append(f"- item_ns_tokens: {report['item_ns_tokens']}")
    lines.append("")

    lines.append("## T Constraint Check\n")
    t_rows = []
    for name, info in report["t_constraints"].items():
        t_rows.append([name, info["num_ns"], info["T"], report["d_model"], info["d_model_divisible"]])
    lines.append(_markdown_table(["config", "num_ns", "T", "d_model", "d_model % T == 0"], t_rows))
    lines.append("")

    lines.append("## RankMixer Summary\n")
    summary_rows = []
    for name in ["user_int", "item_int"]:
        rm = report["rankmixer"][name]
        summary_rows.append(
            [
                name,
                rm["num_tokens"],
                rm["total_emb_dim"],
                rm["chunk_dim"],
                rm["pad_size"],
            ]
        )
    lines.append(_markdown_table(["side", "tokens", "total_emb_dim", "chunk_dim", "pad"], summary_rows))
    lines.append("")

    for name in ["user_int", "item_int"]:
        lines.append(f"## {name} RankMixer Chunks\n")
        chunk_rows = []
        for chunk in report["rankmixer"][name]["chunks"]:
            partial = ",".join(map(str, chunk["partial_fids"])) or "-"
            fids = ",".join(map(str, chunk["fids"]))
            chunk_rows.append(
                [
                    chunk["chunk_id"],
                    f"{chunk['start']}:{chunk['end']}",
                    len(chunk["fids"]),
                    partial,
                    fids,
                ]
            )
        lines.append(_markdown_table(["chunk", "emb_span", "num_fids", "partial_fids", "fids"], chunk_rows))
        lines.append("")

    lines.append("## User Dense Offsets\n")
    dense_rows = [
        [row["fid"], row["length"], row["offset"], row["end"]]
        for row in report["user_dense_offsets"]
    ]
    lines.append(_markdown_table(["fid", "dim", "offset", "end"], dense_rows))
    lines.append("")

    lines.append("## v7 Dense Groups\n")
    group_rows = []
    for name, group in report["v7_dense_groups"]["groups"].items():
        spans = "; ".join(
            f"{row['fid']}[{row['offset']}:{row['end']}]" for row in group["spans"]
        )
        missing = ",".join(map(str, report["v7_dense_groups"]["missing"][name])) or "-"
        group_rows.append([name, group["total_dim"], ",".join(map(str, group["fids"])), missing, spans])
    lines.append(_markdown_table(["group", "total_dim", "fids", "missing", "spans"], group_rows))
    lines.append("")

    lines.append("## v7 Sequence Roles\n")
    for domain, info in report["v7_seq_roles"].items():
        lines.append(f"### {domain}\n")
        rows = []
        for role, role_rows in info["roles"].items():
            if role_rows:
                fids = ", ".join(
                    f"{r['fid']}({r['vocab_size'] if r['present'] else 'missing'})"
                    for r in role_rows
                )
            else:
                fids = "-"
            rows.append([role, fids])
        unassigned = ", ".join(
            f"{r['fid']}({r['vocab_size']})" for r in info["unassigned_sideinfo"]
        ) or "-"
        rows.append(["unassigned", unassigned])
        lines.append(_markdown_table(["role", "fids(vocab)"], rows))
        lines.append("")

    lines.append("## Immediate Readouts\n")
    lines.append("- Current RankMixer chunks are mechanical embedding-dimension slices, not semantic groups.")
    lines.append("- Any `partial_fids` means one fid embedding vector is split across two NS tokens.")
    lines.append("- v7 dense grouping can be implemented from schema offsets without Parquet reads.")
    lines.append("- v7 dense grouping changes `num_ns`, so `rank_mixer_mode=full` may require a different `d_model`.")
    lines.append("- SemanticSeqEmbedder depends on reviving high-cardinality sequence item roles instead of direct `emb_skip_threshold` raise.")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    schema_path = Path(args.schema_path)
    raw = json.loads(schema_path.read_text(encoding="utf-8"))

    user_spans = _build_int_spans(raw["user_int"], args.emb_dim)
    item_spans = _build_int_spans(raw["item_int"], args.emb_dim)
    user_dense = _dense_offsets(raw.get("user_dense", []))

    baseline_num_ns = args.user_ns_tokens + (1 if user_dense else 0) + args.item_ns_tokens
    v7_dense_num_ns = args.user_ns_tokens + 3 + args.item_ns_tokens
    num_sequences = len(raw["seq"])
    report = {
        "schema_path": str(schema_path),
        "emb_dim": args.emb_dim,
        "d_model": args.d_model,
        "num_queries": args.num_queries,
        "num_sequences": num_sequences,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        "t_constraints": {
            "baseline": {
                "num_ns": baseline_num_ns,
                "T": args.num_queries * num_sequences + baseline_num_ns,
                "d_model_divisible": args.d_model % (args.num_queries * num_sequences + baseline_num_ns) == 0,
            },
            "v7_dense_groups": {
                "num_ns": v7_dense_num_ns,
                "T": args.num_queries * num_sequences + v7_dense_num_ns,
                "d_model_divisible": args.d_model % (args.num_queries * num_sequences + v7_dense_num_ns) == 0,
            },
        },
        "rankmixer": {
            "user_int": _rankmixer_chunks(user_spans, args.user_ns_tokens),
            "item_int": _rankmixer_chunks(item_spans, args.item_ns_tokens),
        },
        "user_dense_offsets": user_dense,
        "v7_dense_groups": _dense_group_report(user_dense),
        "v7_seq_roles": _seq_role_report(raw["seq"]),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect TAAC tokenizer layout")
    parser.add_argument("--schema_path", default="src/taac2026_schema.json")
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_queries", type=int, default=2)
    parser.add_argument("--user_ns_tokens", type=int, default=5)
    parser.add_argument("--item_ns_tokens", type=int, default=2)
    parser.add_argument("--out_md", default="docs/eda/2026-05-05-tokenization-audit.md")
    parser.add_argument("--out_json", default="docs/eda/2026-05-05-tokenization-audit.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_markdown(report, Path(args.out_md))
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
