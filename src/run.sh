#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ============================================================
# 实验切换区 —— 改这里就够了 (W1.0.3 / W1.7)
# ============================================================
# Daily driver (transformer baseline): 不传任何 env var
# Longer head A/B (W1.7-L1):       SEQ_ENCODER_TYPE=longer GATHER_SIDE=head ./run.sh
# Longer tail A/B (W1.0.3 验证):   SEQ_ENCODER_TYPE=longer GATHER_SIDE=tail ./run.sh
# 长序列 (W1.7-L2):  SEQ_ENCODER_TYPE=longer GATHER_SIDE=head SEQ_MAX_LENS='seq_a:256,seq_b:256,seq_c:512,seq_d:2048' ./run.sh
# LR: warmup 默认开；cosine 默认关，需要时 USE_COSINE_DECAY=1 ./run.sh
# Missing: dense all-zero/null indicator 默认开；关闭用 DENSE_MISSING_INDICATORS=0 ./run.sh
SEQ_ENCODER_TYPE="${SEQ_ENCODER_TYPE:-transformer}"
GATHER_SIDE="${GATHER_SIDE:-head}"
SEQ_TOP_K="${SEQ_TOP_K:-50}"
SEQ_MAX_LENS="${SEQ_MAX_LENS:-seq_a:256,seq_b:256,seq_c:512,seq_d:512}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
USE_COSINE_DECAY="${USE_COSINE_DECAY:-0}"
DENSE_MISSING_INDICATORS="${DENSE_MISSING_INDICATORS:-1}"
# ============================================================

LR_ARGS=(--warmup_steps "${WARMUP_STEPS}")
if [[ "${USE_COSINE_DECAY}" == "1" || "${USE_COSINE_DECAY}" == "true" ]]; then
    LR_ARGS+=(--use_cosine_decay)
fi

DENSE_MISSING_ARGS=(--use_dense_missing_indicators)
if [[ "${DENSE_MISSING_INDICATORS}" == "0" || "${DENSE_MISSING_INDICATORS}" == "false" ]]; then
    DENSE_MISSING_ARGS=(--no_dense_missing_indicators)
fi

# ---- Active config: RankMixer NS tokenizer (no ns_groups.json required) ----
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --use_amp \
    --use_compile \
    --compile_mode reduce-overhead \
    --seq_encoder_type "${SEQ_ENCODER_TYPE}" \
    --seq_top_k "${SEQ_TOP_K}" \
    --longer_gather_side "${GATHER_SIDE}" \
    --seq_max_lens "${SEQ_MAX_LENS}" \
    "${LR_ARGS[@]}" \
    "${DENSE_MISSING_ARGS[@]}" \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     --use_amp \
#     --use_compile \
#     --compile_mode reduce-overhead \
#     "$@"
