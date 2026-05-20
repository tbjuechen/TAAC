#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ============================================================
# 实验切换区
# ============================================================
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
REINIT_SEQ_PERIODIC_TIME="${REINIT_SEQ_PERIODIC_TIME:-0}"
TIME_BUCKET_BOUNDARIES="${TIME_BUCKET_BOUNDARIES:-hybrid_v1}"
EMA_DECAY="${EMA_DECAY:-0.999}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.05}"
RANK_MIXER_SWIGLU_TYPE="${RANK_MIXER_SWIGLU_TYPE:-shared}"
RANK_MIXER_SWIGLU_GROUPS="${RANK_MIXER_SWIGLU_GROUPS:-}"

D_MODEL="${D_MODEL:-64}"
NUM_HYFORMER_BLOCKS="${NUM_HYFORMER_BLOCKS:-2}"
EMB_SKIP_THRESHOLD="${EMB_SKIP_THRESHOLD:-300000}"

SOURCE_MAP="${USER_CACHE_PATH}/taac_topk_eda/count_maps/topk_rescue_count100.json"
OUT_MAP="${USER_CACHE_PATH}/taac_topk_eda/count_maps/topk_rescue_count100_6targets.json"
# ============================================================

echo "[run.sh] entered"
echo "[run.sh] SCRIPT_DIR=${SCRIPT_DIR}"
echo "[run.sh] SOURCE_MAP=${SOURCE_MAP}"
echo "[run.sh] OUT_MAP=${OUT_MAP}"
echo "[run.sh] EMB_SKIP_THRESHOLD=${EMB_SKIP_THRESHOLD}"
echo "[run.sh] NUM_HYFORMER_BLOCKS=${NUM_HYFORMER_BLOCKS}"
echo "[run.sh] TIME_BUCKET_BOUNDARIES=${TIME_BUCKET_BOUNDARIES}"
echo "[run.sh] LABEL_SMOOTHING=${LABEL_SMOOTHING}"
echo "[run.sh] RANK_MIXER_SWIGLU_TYPE=${RANK_MIXER_SWIGLU_TYPE}"
echo "[run.sh] RANK_MIXER_SWIGLU_GROUPS=${RANK_MIXER_SWIGLU_GROUPS}"

python3 -u -c "import json; src='${SOURCE_MAP}'; out='${OUT_MAP}'; keep={'seq_a:38','seq_b:74','seq_b:76','seq_b:88','seq_c:34','seq_d:22'}; d=json.load(open(src)); d['targets']={k:v for k,v in d['targets'].items() if k in keep}; assert set(d['targets'])==keep, set(d['targets']); json.dump(d, open(out,'w'), ensure_ascii=False, separators=(',',':')); print('TOPK_RESCUE_COUNT100_6TARGETS_DONE '+json.dumps({'out_json':out,'num_targets':len(d['targets']),'targets':sorted(d['targets']),'embedding_rows_sum':sum(int(v.get('vocab_size_for_model',v.get('default_id',0))) for v in d['targets'].values())},ensure_ascii=False,separators=(',',':')), flush=True)"

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
[ "${LABEL_SMOOTHING}" != "0" ] && [ "${LABEL_SMOOTHING}" != "0.0" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --label_smoothing ${LABEL_SMOOTHING}"
EXTRA_FLAGS="${EXTRA_FLAGS} --rank_mixer_swiglu_type ${RANK_MIXER_SWIGLU_TYPE}"
[ -n "${RANK_MIXER_SWIGLU_GROUPS}" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --rank_mixer_swiglu_groups ${RANK_MIXER_SWIGLU_GROUPS}"

echo "[run.sh] start training"

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 4 \
    --item_ns_tokens 2 \
    --split_user_int_shared_fids \
    --use_dense_group_projector \
    --num_queries 2 \
    --d_model "${D_MODEL}" \
    --num_hyformer_blocks "${NUM_HYFORMER_BLOCKS}" \
    --ns_groups_json "" \
    --emb_skip_threshold "${EMB_SKIP_THRESHOLD}" \
    --topk_rescue_map "${OUT_MAP}" \
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

echo "[run.sh] done"
