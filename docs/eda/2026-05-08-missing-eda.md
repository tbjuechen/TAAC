# TAAC Missing-Value EDA

## Summary

- rows: 907,381; pos_rate: 9.582%
- schema missing columns: 0
- Goal: split raw `null / -1 / 0 / empty / all-zero` signals that current dataset maps to id 0.

## High-Rate Categorical No-Positive Rows

| group | fid | dim | no_positive_rate | no_positive_pos_rate | lift | present_pos_rate |
|---|---:|---:|---:|---:|---:|---:|
| user_int | 101 | 1 | 92.048% | 9.533% | 0.995x | 10.146% |
| user_int | 102 | 1 | 89.558% | 9.483% | 0.990x | 10.429% |
| user_int | 103 | 1 | 88.269% | 9.528% | 0.994x | 9.990% |
| item_int | 83 | 1 | 87.940% | 9.168% | 0.957x | 12.600% |
| item_int | 84 | 1 | 87.940% | 9.168% | 0.957x | 12.600% |
| item_int | 85 | 1 | 87.940% | 9.168% | 0.957x | 12.600% |
| user_int | 109 | 1 | 86.788% | 9.493% | 0.991x | 10.164% |
| user_int | 100 | 1 | 85.461% | 9.537% | 0.995x | 9.847% |
| user_int | 99 | 1 | 81.733% | 9.371% | 0.978x | 10.523% |
| item_int | 81 | 1 | 72.205% | 9.162% | 0.956x | 10.672% |
| user_int | 86 | 1 | 71.459% | 9.533% | 0.995x | 9.704% |
| user_int | 96 | 1 | 70.758% | 9.795% | 1.022x | 9.066% |
| user_int | 60 | 2 | 67.672% | 9.588% | 1.001x | 9.570% |
| item_int | 8 | 1 | 61.213% | 9.386% | 0.980x | 9.892% |
| user_int | 94 | 1 | 53.101% | 9.437% | 0.985x | 9.746% |
| user_int | 108 | 1 | 52.857% | 9.427% | 0.984x | 9.755% |

## Train/Val Missing Stability

| group | fid | train_no_positive | val_no_positive | Δpp | train_lift | val_lift |
|---|---:|---:|---:|---:|---:|---:|
| user_int | 101 | 92.048% | 91.978% | -0.070 | 0.995x | 0.991x |
| user_int | 102 | 89.558% | 89.645% | +0.087 | 0.990x | 0.987x |
| user_int | 103 | 88.269% | 88.286% | +0.016 | 0.994x | 0.990x |
| item_int | 83 | 87.940% | 88.067% | +0.126 | 0.957x | 0.962x |
| item_int | 84 | 87.940% | 88.067% | +0.126 | 0.957x | 0.962x |
| item_int | 85 | 87.940% | 88.067% | +0.126 | 0.957x | 0.962x |
| user_int | 109 | 86.788% | 86.897% | +0.109 | 0.991x | 0.986x |
| user_int | 100 | 85.461% | 85.415% | -0.046 | 0.995x | 0.995x |
| user_int | 99 | 81.733% | 81.508% | -0.225 | 0.978x | 0.980x |
| item_int | 81 | 72.205% | 72.304% | +0.099 | 0.956x | 0.956x |
| user_int | 86 | 71.459% | 71.515% | +0.056 | 0.995x | 0.986x |
| user_int | 96 | 70.758% | 70.811% | +0.053 | 1.022x | 1.017x |

## Raw 0 / -1 / null Signals

| group | fid | null_rate | any_neg1_rate | any_zero_rate | value_zero_rate | value_neg1_rate |
|---|---:|---:|---:|---:|---:|---:|
| user_int | 89 | 4.127% | 0.000% | 95.873% | 10.000% | 0.000% |
| user_int | 90 | 7.644% | 0.000% | 92.356% | 10.000% | 0.000% |
| item_int | 81 | 0.003% | 0.000% | 72.202% | 72.204% | 0.000% |
| item_int | 8 | 0.003% | 0.863% | 60.346% | 60.348% | 0.864% |
| user_int | 91 | 48.023% | 0.000% | 51.977% | 10.000% | 0.000% |
| item_int | 7 | 0.003% | 0.000% | 14.937% | 14.938% | 0.000% |
| item_int | 12 | 0.003% | 0.000% | 14.937% | 14.938% | 0.000% |
| user_int | 63 | 6.267% | 0.000% | 9.878% | 3.936% | 0.000% |
| item_int | 6 | 0.003% | 0.000% | 4.703% | 4.703% | 0.000% |
| user_int | 57 | 2.050% | 0.000% | 0.227% | 0.232% | 0.000% |
| item_int | 16 | 0.001% | 0.000% | 0.185% | 0.185% | 0.000% |
| user_int | 50 | 0.121% | 0.000% | 0.125% | 0.125% | 0.000% |
| user_int | 62 | 6.266% | 0.000% | 0.003% | 0.001% | 0.000% |
| item_int | 5 | 0.003% | 0.000% | 0.001% | 0.001% | 0.000% |
| user_int | 4 | 2.707% | 0.000% | 0.001% | 0.001% | 0.000% |
| user_int | 1 | 0.015% | 0.000% | 0.000% | 0.000% | 0.000% |

## Dense All-Zero Rows

| fid | dim | all_zero_rate | all_zero_pos_rate | lift | present_nonzero_pos_rate | nan_rate | inf_rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 91 | 10 | 48.023% | 9.352% | 0.976x | 9.794% | 0.000% | 0.000% |
| 90 | 10 | 7.644% | 9.041% | 0.944x | 9.627% | 0.000% | 0.000% |
| 66 | 150 | 7.543% | 11.963% | 1.248x | 9.388% | 0.000% | 0.000% |
| 65 | 111 | 7.166% | 11.933% | 1.245x | 9.400% | 0.000% | 0.000% |
| 63 | 19 | 6.267% | 12.135% | 1.266x | 9.411% | 0.000% | 0.000% |
| 62 | 6 | 6.266% | 12.138% | 1.267x | 9.411% | 0.000% | 0.000% |
| 64 | 26 | 6.265% | 12.139% | 1.267x | 9.411% | 0.000% | 0.000% |
| 89 | 10 | 4.127% | 9.223% | 0.963x | 9.597% | 0.000% | 0.000% |
| 87 | 320 | 0.643% | 10.399% | 1.085x | 9.577% | 0.000% | 0.000% |
| 61 | 256 | 0.129% | 16.382% | 1.710x | 9.573% | 0.000% | 0.000% |

## Sequence Missing-Like Rows

| domain | mean_raw_len | empty_raw_rate | no_positive_sideinfo_rate | any_nonpositive_token_rate | no_positive_lift |
|---|---:|---:|---:|---:|---:|
| seq_a | 751.33 | 0.075% | 0.075% | 99.925% | 1.204x |
| seq_b | 722.65 | 0.114% | 0.114% | 99.886% | 1.283x |
| seq_c | 513.00 | 0.050% | 0.050% | 99.940% | 1.705x |
| seq_d | 2455.66 | 0.097% | 0.097% | 99.902% | 0.488x |

## Decision Hints

- If high no-positive categorical fids show lift != 1, use a learnable missing bucket for scalar categorical fids instead of frozen padding 0.
- Keep multi-value and sequence padding as 0; only add missing buckets where raw missing/0 is feature value semantics, not structural padding.
- If dense all-zero rows have lift, add per-fid all-zero/missing indicators or a dense-missing token.
- NaN/Inf should remain a guardrail, not the main missing-value strategy, unless rates are non-zero above.
