#!/bin/bash
# Local smoke test for src/profile_data.py over data/demo.
# Mirrors tools/run_demo.sh structure.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

OUT_DIR="${PROJECT_ROOT}/output/profile/demo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" -u "${PROJECT_ROOT}/src/profile_data.py" \
    --data_dir "${PROJECT_ROOT}/data/demo" \
    --schema_path "${PROJECT_ROOT}/data/demo/schema.json" \
    --log_dir "${OUT_DIR}" \
    --valid_ratio 0.2 \
    --seq_max_lens "seq_a:16,seq_b:16,seq_c:16,seq_d:16" \
    --batch_size 256 \
    "$@" \
    > "${OUT_DIR}/stdout.txt" \
    2> "${OUT_DIR}/stderr.txt"

LINE_COUNT=$(wc -l < "${OUT_DIR}/stdout.txt")
echo ""
echo "=== Demo profile run complete ==="
echo "  output dir : ${OUT_DIR}"
echo "  stdout lines: ${LINE_COUNT} (budget 1000)"
echo "  stderr tail :"
tail -20 "${OUT_DIR}/stderr.txt" | sed 's/^/    /'
if [ "${LINE_COUNT}" -gt 1000 ]; then
    echo "  WARNING: stdout exceeds 1000-line platform cap" >&2
    exit 1
fi
echo "  Run: python ${PROJECT_ROOT}/tools/parse_profile.py ${OUT_DIR}/stdout.txt"
