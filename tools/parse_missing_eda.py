"""Parse missing-value EDA stdout.

Usage:
    python tools/parse_missing_eda.py <stdout_file> [--out-dir DIR]

The EDA script prints markdown plus two copy-friendly one-line payloads:
``MISSING_EDA_REPORT_ONE_LINE=<json-string>`` and
``MISSING_EDA_JSON_DUMP=<json>``. This parser writes:
  - missing_eda_report.md
  - missing_eda.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPORT_PREFIX = "MISSING_EDA_REPORT_ONE_LINE="
JSON_PREFIX = "MISSING_EDA_JSON_DUMP="


def _find_payload(lines: list[str], prefix: str) -> str:
    for line in reversed(lines):
        if line.startswith(prefix):
            return line[len(prefix):]
    raise SystemExit(f"error: no {prefix!r} payload found")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stdout_file", type=str)
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    src = Path(args.stdout_file)
    if not src.is_file():
        raise SystemExit(f"error: {src} is not a file")

    out_dir = Path(args.out_dir) if args.out_dir else src.parent / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = src.read_text(encoding="utf-8").splitlines()
    report = json.loads(_find_payload(lines, REPORT_PREFIX))
    data = json.loads(_find_payload(lines, JSON_PREFIX))

    md_path = out_dir / "missing_eda_report.md"
    md_path.write_text(report, encoding="utf-8")
    print(f"wrote {md_path} ({len(report.splitlines())} lines)")

    json_path = out_dir / "missing_eda.json"
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {json_path} ({json_path.stat().st_size:,} bytes)")

    meta = data.get("meta", {})
    print()
    print("Summary:")
    print(f"  train_rows: {meta.get('rows', 'n/a')}")
    print(f"  train_pos_rate: {meta.get('pos_rate', 'n/a')}")
    print(f"  val_rows: {meta.get('val_rows', 'n/a')}")
    print(f"  val_pos_rate: {meta.get('val_pos_rate', 'n/a')}")


if __name__ == "__main__":
    main()
