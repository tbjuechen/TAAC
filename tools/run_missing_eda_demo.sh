#!/bin/bash
# Local smoke test for tools/missing_eda.py over data/demo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

OUT_DIR="${PROJECT_ROOT}/output/missing_eda/demo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" -u "${PROJECT_ROOT}/tools/missing_eda.py" \
    --data_dir "${PROJECT_ROOT}/data/demo" \
    --schema_path "${PROJECT_ROOT}/data/demo/schema.json" \
    --log_dir "${OUT_DIR}" \
    --valid_ratio 0.2 \
    --batch_size 256 \
    "$@" \
    > "${OUT_DIR}/stdout.txt" \
    2> "${OUT_DIR}/stderr.txt"

LINE_COUNT=$(wc -l < "${OUT_DIR}/stdout.txt")
echo ""
echo "=== Demo missing EDA run complete ==="
echo "  output dir   : ${OUT_DIR}"
echo "  stdout lines : ${LINE_COUNT} (budget 1000)"
echo "  stderr tail  :"
tail -20 "${OUT_DIR}/stderr.txt" | sed 's/^/    /'
if [ "${LINE_COUNT}" -gt 1000 ]; then
    echo "  WARNING: stdout exceeds 1000-line platform cap" >&2
    exit 1
fi
echo "  Run: python ${PROJECT_ROOT}/tools/parse_missing_eda.py ${OUT_DIR}/stdout.txt"
