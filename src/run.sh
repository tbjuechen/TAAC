#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ============================================================
# 实验切换区 —— 改这里就够了 (W1.0.3 / W1.7 / W2.7)
# ============================================================
# Daily driver (transformer baseline): 不传任何 env var
# Longer head A/B (W1.7-L1):       SEQ_ENCODER_TYPE=longer GATHER_SIDE=head ./run.sh
# Longer tail A/B (W1.0.3 验证):   SEQ_ENCODER_TYPE=longer GATHER_SIDE=tail ./run.sh
# 长序列 (W1.7-L2):  SEQ_ENCODER_TYPE=longer GATHER_SIDE=head SEQ_MAX_LENS='seq_a:256,seq_b:256,seq_c:512,seq_d:2048' ./run.sh
# W2.7 delta buckets:              USE_DELTA_BUCKETS=1 ./run.sh
# W2.7.1 per-domain time emb:      PER_DOMAIN_TIME_EMB=1 ./run.sh
# W2.7.2 time residual emb:        DOMAIN_TIME_RESIDUAL_EMB=1 ./run.sh
# W2.8 time summary NS token:      USE_TIME_SUMMARY=1 ./run.sh
# W2.9 seq periodic concat:        USE_SEQ_PERIODIC_TIME=1 ./run.sh
# W2.9 hour-only ablation:         USE_SEQ_HOUR=1 ./run.sh
# W2.9 dow-only ablation:          USE_SEQ_DOW=1 ./run.sh
# W2.9b per-domain periodic:       USE_PER_DOMAIN_PERIODIC_TIME=1 ./run.sh
# EMA validation/checkpoint:       EMA_DECAY=1 ./run.sh
# EMA decay sweep:                 EMA_DECAY=0.9995 ./run.sh
SEQ_ENCODER_TYPE="${SEQ_ENCODER_TYPE:-transformer}"
GATHER_SIDE="${GATHER_SIDE:-head}"
SEQ_TOP_K="${SEQ_TOP_K:-50}"
SEQ_MAX_LENS="${SEQ_MAX_LENS:-seq_a:256,seq_b:256,seq_c:512,seq_d:512}"
USE_DELTA_BUCKETS="${USE_DELTA_BUCKETS:-0}"
PER_DOMAIN_TIME_EMB="${PER_DOMAIN_TIME_EMB:-0}"
DOMAIN_TIME_RESIDUAL_EMB="${DOMAIN_TIME_RESIDUAL_EMB:-0}"
USE_TIME_SUMMARY="${USE_TIME_SUMMARY:-0}"
USE_SEQ_PERIODIC_TIME="${USE_SEQ_PERIODIC_TIME:-1}"
USE_SEQ_HOUR="${USE_SEQ_HOUR:-0}"
USE_SEQ_DOW="${USE_SEQ_DOW:-0}"
USE_PER_DOMAIN_PERIODIC_TIME="${USE_PER_DOMAIN_PERIODIC_TIME:-0}"
EMA_DECAY="${EMA_DECAY:-0}"
if [ "${EMA_DECAY}" = "1" ]; then
    EMA_DECAY=0.999
fi
if [ -z "${D_MODEL+x}" ]; then
    if [ "${USE_TIME_SUMMARY}" = "1" ]; then
        D_MODEL=68
    else
        D_MODEL=64
    fi
fi
# ============================================================

EXTRA_FLAGS=""
[ "${USE_DELTA_BUCKETS}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --use_delta_buckets"
[ "${PER_DOMAIN_TIME_EMB}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --per_domain_time_embeddings"
[ "${DOMAIN_TIME_RESIDUAL_EMB}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --domain_time_residual_embeddings"
[ "${USE_TIME_SUMMARY}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --use_time_summary_features"
[ "${USE_SEQ_PERIODIC_TIME}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --use_seq_periodic_time_features"
[ "${USE_SEQ_HOUR}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --use_seq_hour_of_day_feature"
[ "${USE_SEQ_DOW}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --use_seq_day_of_week_feature"
[ "${USE_PER_DOMAIN_PERIODIC_TIME}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --per_domain_seq_periodic_time_features"
[ "${EMA_DECAY}" != "0" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --ema_decay ${EMA_DECAY}"

# ---- Active config: RankMixer NS tokenizer (no ns_groups.json required) ----
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 4 \
    --item_ns_tokens 2 \
    --split_user_int_shared_fids \
    --use_dense_group_projector \
    --num_queries 2 \
    --d_model "${D_MODEL}" \
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
    ${EXTRA_FLAGS} \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=13 (7 user_int + 2 user_dense + 4 item_int),
# rank_mixer_mode=full has no valid num_queries because T=4*num_queries+13
# does not divide 64 for positive num_queries. Use ffn_only/none or adjust
# d_model before enabling this block.
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
