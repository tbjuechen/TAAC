#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- W2.6 pair-weighted pool A/B (default = baseline) ----
#   PAIR_WEIGHTED_POOL=none        bash run.sh   # baseline mean-pool
#   PAIR_WEIGHTED_POOL=log1p       bash run.sh   # [DEAD F27] log1p on fid 62-66 only
#   PAIR_WEIGHTED_POOL=full        bash run.sh   # [DEAD F27] log1p on 62-66 + sigmoid on 89-91
#   PAIR_WEIGHTED_POOL=transformer bash run.sh   # W2.6 v2: PairSetEncoder (bucket+attn pool) on all 8 paired fids
PAIR_WEIGHTED_POOL="${PAIR_WEIGHTED_POOL:-none}"

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
    --pair_weighted_pool "$PAIR_WEIGHTED_POOL" \
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
#     --pair_weighted_pool "$PAIR_WEIGHTED_POOL" \
#     "$@"
