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
# W2.7.3 gated time diff emb:      GATED_TIME_DIFF_EMB=1 ./run.sh
# W2.8 time summary NS token:      USE_TIME_SUMMARY=1 ./run.sh
# W2.9 seq periodic concat:        USE_SEQ_PERIODIC_TIME=1 ./run.sh  # HOD + DOW + MOY
# W2.9 hour-only ablation:         USE_SEQ_HOUR=1 ./run.sh
# W2.9 dow-only ablation:          USE_SEQ_DOW=1 ./run.sh
# W2.9 moy-only ablation:          USE_SEQ_MONTH_OF_YEAR=1 ./run.sh
# W2.9b per-domain periodic:       USE_PER_DOMAIN_PERIODIC_TIME=1 ./run.sh
# TopK rescue (emb_skip=100w + compact topK/default for seq_c:34/29):
#   USE_TOPK_RESCUE=1 ./run.sh
# W2.9c reinit periodic time emb:  REINIT_SEQ_PERIODIC_TIME=1 ./run.sh
# W2.10 hybrid recency buckets:    TIME_BUCKET_BOUNDARIES=hybrid_v1 ./run.sh
# EMA validation/checkpoint:       EMA_DECAY=0.999 ./run.sh
# User/history token dropout:      USER_TOKEN_DROPOUT_RATE=0.03 SEQ_TOKEN_DROPOUT_RATE=0.03 ./run.sh
SEQ_ENCODER_TYPE="${SEQ_ENCODER_TYPE:-transformer}"
GATHER_SIDE="${GATHER_SIDE:-head}"
SEQ_TOP_K="${SEQ_TOP_K:-50}"
SEQ_MAX_LENS="${SEQ_MAX_LENS:-seq_a:256,seq_b:256,seq_c:512,seq_d:512}"
USE_DELTA_BUCKETS="${USE_DELTA_BUCKETS:-0}"
PER_DOMAIN_TIME_EMB="${PER_DOMAIN_TIME_EMB:-0}"
DOMAIN_TIME_RESIDUAL_EMB="${DOMAIN_TIME_RESIDUAL_EMB:-0}"
GATED_TIME_DIFF_EMB="${GATED_TIME_DIFF_EMB:-0}"
USE_TIME_SUMMARY="${USE_TIME_SUMMARY:-0}"
USE_SEQ_PERIODIC_TIME="${USE_SEQ_PERIODIC_TIME:-1}"
USE_SEQ_HOUR="${USE_SEQ_HOUR:-0}"
USE_SEQ_DOW="${USE_SEQ_DOW:-0}"
USE_SEQ_MOY="${USE_SEQ_MOY:-0}"
USE_SEQ_MONTH_OF_YEAR="${USE_SEQ_MONTH_OF_YEAR:-${USE_SEQ_MOY}}"
USE_PER_DOMAIN_PERIODIC_TIME="${USE_PER_DOMAIN_PERIODIC_TIME:-0}"
USE_TOPK_RESCUE="${USE_TOPK_RESCUE:-0}"
TOPK_RESCUE_TARGETS="${TOPK_RESCUE_TARGETS:-}"
TOPK_RESCUE_EDA_JSON="${TOPK_RESCUE_EDA_JSON:-}"
TOPK_RESCUE_MAP="${TOPK_RESCUE_MAP:-}"
TOPK_RESCUE_AUTO_EXPORT="${TOPK_RESCUE_AUTO_EXPORT:-1}"
TOPK_RESCUE_SOURCE_MAP="${TOPK_RESCUE_SOURCE_MAP:-}"
TOPK_RESCUE_FULL_EDA_JSON="${TOPK_RESCUE_FULL_EDA_JSON:-}"
TOPK_RESCUE_FULL_EDA_MD="${TOPK_RESCUE_FULL_EDA_MD:-}"
REINIT_SEQ_PERIODIC_TIME="${REINIT_SEQ_PERIODIC_TIME:-0}"
TIME_BUCKET_BOUNDARIES="${TIME_BUCKET_BOUNDARIES:-original}"
EMA_DECAY="${EMA_DECAY:-0}"
USER_TOKEN_DROPOUT_RATE="${USER_TOKEN_DROPOUT_RATE:-0}"
SEQ_TOKEN_DROPOUT_RATE="${SEQ_TOKEN_DROPOUT_RATE:-0}"
if [ -z "${D_MODEL+x}" ]; then
    if [ "${USE_TIME_SUMMARY}" = "1" ]; then
        D_MODEL=68
    else
        D_MODEL=64
    fi
fi
# ============================================================

if [ "${USE_TOPK_RESCUE}" = "1" ] && [ -z "${TOPK_RESCUE_TARGETS}" ]; then
    TOPK_RESCUE_TARGETS="seq_c:34:10000,seq_c:29:200000"
fi

EXTRA_FLAGS=""
[ "${USE_DELTA_BUCKETS}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --use_delta_buckets"
[ "${PER_DOMAIN_TIME_EMB}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --per_domain_time_embeddings"
[ "${DOMAIN_TIME_RESIDUAL_EMB}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --domain_time_residual_embeddings"
[ "${GATED_TIME_DIFF_EMB}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --gated_time_diff_embeddings"
[ "${USE_TIME_SUMMARY}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --use_time_summary_features"
[ "${USE_SEQ_PERIODIC_TIME}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --use_seq_periodic_time_features"
[ "${USE_SEQ_HOUR}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --use_seq_hour_of_day_feature"
[ "${USE_SEQ_DOW}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --use_seq_day_of_week_feature"
[ "${USE_SEQ_MONTH_OF_YEAR}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --use_seq_month_of_year_feature"
[ "${USE_PER_DOMAIN_PERIODIC_TIME}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --per_domain_seq_periodic_time_features"
[ "${REINIT_SEQ_PERIODIC_TIME}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --reinit_seq_periodic_time_features"
EXTRA_FLAGS="${EXTRA_FLAGS} --time_bucket_boundaries ${TIME_BUCKET_BOUNDARIES}"
[ "${EMA_DECAY}" != "0" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --ema_decay ${EMA_DECAY}"
[ "${USER_TOKEN_DROPOUT_RATE}" != "0" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --user_token_dropout_rate ${USER_TOKEN_DROPOUT_RATE}"
[ "${SEQ_TOKEN_DROPOUT_RATE}" != "0" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --seq_token_dropout_rate ${SEQ_TOKEN_DROPOUT_RATE}"

if [ -n "${TOPK_RESCUE_TARGETS}" ]; then
    if [ -z "${TOPK_RESCUE_EDA_JSON}" ] || [ -z "${TOPK_RESCUE_MAP}" ]; then
        if [ -z "${USER_CACHE_PATH:-}" ]; then
            echo "[run.sh] TOPK_RESCUE_TARGETS set but USER_CACHE_PATH is unavailable; set TOPK_RESCUE_EDA_JSON and TOPK_RESCUE_MAP explicitly"
            exit 1
        fi
        TOPK_RESCUE_EDA_JSON="${TOPK_RESCUE_EDA_JSON:-${USER_CACHE_PATH}/taac_topk_eda/topk_tail_eda.json}"
        TOPK_RESCUE_MAP="${TOPK_RESCUE_MAP:-${USER_CACHE_PATH}/taac_topk_eda/topk_rescue_map.json}"
    fi
    if [ ! -f "${TOPK_RESCUE_EDA_JSON}" ]; then
        echo "[run.sh] TOPK_RESCUE_EDA_JSON not found: ${TOPK_RESCUE_EDA_JSON}"
        exit 1
    fi
    mkdir -p "$(dirname "${TOPK_RESCUE_MAP}")"
    echo "[run.sh] building topk rescue map targets=${TOPK_RESCUE_TARGETS}"
    echo "[run.sh] topk rescue source=${TOPK_RESCUE_EDA_JSON}"
    if ! python3 -u "${SCRIPT_DIR}/build_topk_rescue_map.py" \
        --eda-json "${TOPK_RESCUE_EDA_JSON}" \
        --targets "${TOPK_RESCUE_TARGETS}" \
        --out-json "${TOPK_RESCUE_MAP}"; then
        if [ "${TOPK_RESCUE_AUTO_EXPORT}" != "1" ]; then
            echo "[run.sh] topk rescue source has no full ids and TOPK_RESCUE_AUTO_EXPORT!=1"
            exit 1
        fi
        if [ -z "${TRAIN_DATA_PATH:-}" ]; then
            echo "[run.sh] cannot auto-export topk rescue map because TRAIN_DATA_PATH is unset"
            exit 1
        fi

        if [ -f "${SCRIPT_DIR}/topk_tail_eda.py" ]; then
            TOPK_TAIL_EDA_SCRIPT="${SCRIPT_DIR}/topk_tail_eda.py"
        elif [ -f "${SCRIPT_DIR}/tools/topk_tail_eda.py" ]; then
            TOPK_TAIL_EDA_SCRIPT="${SCRIPT_DIR}/tools/topk_tail_eda.py"
        elif [ -f "${SCRIPT_DIR}/../tools/topk_tail_eda.py" ]; then
            TOPK_TAIL_EDA_SCRIPT="${SCRIPT_DIR}/../tools/topk_tail_eda.py"
        else
            echo "[run.sh] cannot auto-export: topk_tail_eda.py not found next to run.sh or under tools/"
            exit 1
        fi

        TOPK_RESCUE_SCAN_TARGETS="$(python3 -c 'import sys; print(",".join(":".join(x.split(":")[:2]) for x in sys.argv[1].split(",") if x.strip()))' "${TOPK_RESCUE_TARGETS}")"
        TOPK_RESCUE_SOURCE_MAP="${TOPK_RESCUE_SOURCE_MAP:-$(dirname "${TOPK_RESCUE_MAP}")/topk_source_map.json}"
        TOPK_RESCUE_FULL_EDA_JSON="${TOPK_RESCUE_FULL_EDA_JSON:-$(dirname "${TOPK_RESCUE_MAP}")/topk_tail_eda_with_map.json}"
        TOPK_RESCUE_FULL_EDA_MD="${TOPK_RESCUE_FULL_EDA_MD:-$(dirname "${TOPK_RESCUE_MAP}")/topk_tail_eda_with_map.md}"

        echo "[run.sh] auto-exporting full topk ids targets=${TOPK_RESCUE_SCAN_TARGETS}"
        python3 -u "${TOPK_TAIL_EDA_SCRIPT}" \
            --data-dir "${TRAIN_DATA_PATH}" \
            --schema-path "${TRAIN_DATA_PATH}/schema.json" \
            --targets "${TOPK_RESCUE_SCAN_TARGETS}" \
            --workers 8 \
            --executor process \
            --row-group-batch-size 10 \
            --candidate-capacity 300000 \
            --export-map-targets "${TOPK_RESCUE_TARGETS}" \
            --export-topk-map "${TOPK_RESCUE_SOURCE_MAP}" \
            --out-json "${TOPK_RESCUE_FULL_EDA_JSON}" \
            --out-md "${TOPK_RESCUE_FULL_EDA_MD}" \
            --no-print-md-one-line || exit 1

        echo "[run.sh] rebuilding topk rescue map from exported full ids: ${TOPK_RESCUE_SOURCE_MAP}"
        python3 -u "${SCRIPT_DIR}/build_topk_rescue_map.py" \
            --eda-json "${TOPK_RESCUE_SOURCE_MAP}" \
            --targets "${TOPK_RESCUE_TARGETS}" \
            --out-json "${TOPK_RESCUE_MAP}" || exit 1
    fi
    EXTRA_FLAGS="${EXTRA_FLAGS} --topk_rescue_map ${TOPK_RESCUE_MAP}"
fi

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
