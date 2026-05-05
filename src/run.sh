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
# v7 dense groups: DENSE_GROUP_PROJECTOR=v7 requires d_model/T compatibility
#   (baseline T=16; v7 dense groups T=18, so use e.g. --d_model 72 or --rank_mixer_mode ffn_only)
SEQ_ENCODER_TYPE="${SEQ_ENCODER_TYPE:-transformer}"
GATHER_SIDE="${GATHER_SIDE:-head}"
SEQ_TOP_K="${SEQ_TOP_K:-50}"
SEQ_MAX_LENS="${SEQ_MAX_LENS:-seq_a:256,seq_b:256,seq_c:512,seq_d:512}"
DENSE_GROUP_PROJECTOR="${DENSE_GROUP_PROJECTOR:-none}"
STAT_DENSE_TRANSFORM="${STAT_DENSE_TRANSFORM:-log1p_clip}"
# ============================================================

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
    --dense_group_projector "${DENSE_GROUP_PROJECTOR}" \
    --stat_dense_transform "${STAT_DENSE_TRANSFORM}" \
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
