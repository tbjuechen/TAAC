"""Parse PROFILE_JSON_DUMP from a TAAC profile stdout capture.

Usage:
    python tools/parse_profile.py <stdout_file> [--out-dir DIR]

Extracts the trailing ``PROFILE_JSON_DUMP=<json>`` line, decodes it, and:
  - copies the markdown report to ``<DIR>/profile_report.md``
  - dumps the parsed JSON to ``<DIR>/profile_data.json`` (pretty-printed)

Defaults ``DIR`` to ``$(dirname stdout_file)/parsed``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


JSON_PREFIX = "PROFILE_JSON_DUMP="


def _split(text: str) -> tuple[str, str]:
    """Return (markdown_body, json_str). Errors if no JSON marker found."""
    lines = text.splitlines()
    json_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith(JSON_PREFIX):
            json_idx = i
            break
    if json_idx is None:
        raise SystemExit(
            f"error: no '{JSON_PREFIX}' line found in input. "
            f"Did the profile run finish? (input has {len(lines)} lines)"
        )
    md = "\n".join(lines[:json_idx]).rstrip() + "\n"
    js = lines[json_idx][len(JSON_PREFIX):]
    return md, js


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stdout_file", type=str,
                   help="path to a captured profile stdout file")
    p.add_argument("--out-dir", type=str, default=None,
                   help="output directory (default: <dir(stdout)>/parsed)")
    args = p.parse_args()

    src = Path(args.stdout_file)
    if not src.is_file():
        raise SystemExit(f"error: {src} is not a file")

    out_dir = Path(args.out_dir) if args.out_dir else src.parent / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)

    text = src.read_text(encoding="utf-8")
    md, js = _split(text)

    md_path = out_dir / "profile_report.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"wrote {md_path} ({len(md.splitlines())} lines)")

    try:
        data = json.loads(js)
    except json.JSONDecodeError as e:
        raise SystemExit(f"error: PROFILE_JSON_DUMP is not valid JSON: {e}")

    json_path = out_dir / "profile_data.json"
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"wrote {json_path} ({json_path.stat().st_size:,} bytes)")

    # Summary
    meta = data.get("meta", {})
    print()
    print("Summary:")
    print(f"  train_rows: {meta.get('train_rows', 'n/a'):>12,}")
    print(f"  val_rows  : {meta.get('val_rows',   'n/a'):>12,}")
    print(f"  train_pos : {meta.get('train_pos_count', 'n/a'):>12,}")
    print(f"  val_pos   : {meta.get('val_pos_count',   'n/a'):>12,}")


if __name__ == "__main__":
    main()
