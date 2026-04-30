#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TRAIN_TF_EVENTS_PATH="${PROJECT_ROOT}/runs/tensorboard"

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_demo_data.py" \
    --out_dir "${PROJECT_ROOT}/data/demo"

"${PYTHON_BIN}" -u "${SCRIPT_DIR}/run_train.py" \
    --data_dir "${PROJECT_ROOT}/data/demo" \
    --schema_path "${PROJECT_ROOT}/data/demo/schema.json" \
    --ckpt_dir "${PROJECT_ROOT}/runs/demo_checkpoints" \
    --log_dir "${PROJECT_ROOT}/runs/demo_logs" \
    --batch_size 32 \
    --num_epochs 1 \
    --patience 1 \
    --device cpu \
    --num_workers 0 \
    --buffer_batches 1 \
    --valid_ratio 0.2 \
    --seq_max_lens seq_a:16,seq_b:16,seq_c:16,seq_d:16 \
    --d_model 16 \
    --emb_dim 8 \
    --num_queries 2 \
    --num_hyformer_blocks 1 \
    --num_heads 4 \
    --seq_encoder_type swiglu \
    --hidden_mult 2 \
    --dropout_rate 0.01 \
    --rank_mixer_mode full \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --reinit_sparse_after_epoch 999 \
    "$@"
