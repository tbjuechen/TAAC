# TAAC Tokenization Audit

- schema: `src/taac2026_schema.json`
- emb_dim: 64
- d_model: 64
- num_queries: 2
- user_ns_tokens: 5
- item_ns_tokens: 2

## T Constraint Check

| config | num_ns | T | d_model | d_model % T == 0 |
| --- | --- | --- | --- | --- |
| baseline | 8 | 16 | 64 | True |
| v7_dense_groups | 10 | 18 | 64 | False |

## RankMixer Summary

| side | tokens | total_emb_dim | chunk_dim | pad |
| --- | --- | --- | --- | --- |
| user_int | 5 | 2944 | 589 | 1 |
| item_int | 2 | 896 | 448 | 0 |

## user_int RankMixer Chunks

| chunk | emb_span | num_fids | partial_fids | fids |
| --- | --- | --- | --- | --- |
| 0 | 0:589 | 10 | 53 | 1,3,4,15,48,49,50,51,52,53 |
| 1 | 589:1178 | 10 | 53,63 | 53,54,55,56,57,58,59,60,62,63 |
| 2 | 1178:1767 | 10 | 63,91 | 63,64,65,66,80,82,86,89,90,91 |
| 3 | 1767:2356 | 10 | 91,100 | 91,92,93,94,95,96,97,98,99,100 |
| 4 | 2356:2945 | 10 | 100 | 100,101,102,103,104,105,106,107,108,109 |

## item_int RankMixer Chunks

| chunk | emb_span | num_fids | partial_fids | fids |
| --- | --- | --- | --- | --- |
| 0 | 0:448 | 7 | - | 5,6,7,8,9,10,11 |
| 1 | 448:896 | 7 | - | 12,13,16,81,83,84,85 |

## User Dense Offsets

| fid | dim | offset | end |
| --- | --- | --- | --- |
| 61 | 256 | 0 | 256 |
| 62 | 6 | 256 | 262 |
| 63 | 19 | 262 | 281 |
| 64 | 26 | 281 | 307 |
| 65 | 111 | 307 | 418 |
| 66 | 150 | 418 | 568 |
| 87 | 320 | 568 | 888 |
| 89 | 10 | 888 | 898 |
| 90 | 10 | 898 | 908 |
| 91 | 10 | 908 | 918 |

## v7 Dense Groups

| group | total_dim | fids | missing | spans |
| --- | --- | --- | --- | --- |
| emb | 576 | 61,87 | - | 61[0:256]; 87[568:888] |
| stat | 312 | 62,63,64,65,66 | - | 62[256:262]; 63[262:281]; 64[281:307]; 65[307:418]; 66[418:568] |
| quantile | 30 | 89,90,91 | - | 89[888:898]; 90[898:908]; 91[908:918] |

## v7 Sequence Roles

### seq_a

| role | fids(vocab) |
| --- | --- |
| item | 38(745286) |
| action | 40(19) |
| stat | 42(1005), 43(3342), 44(12735), 45(7612) |
| unassigned | 41(11), 46(18) |

### seq_b

| role | fids(vocab) |
| --- | --- |
| item | 69(64710562) |
| action | 68(28) |
| stat | 70(726), 71(2669), 72(10203), 73(6761), 74(476333), 75(31), 76(132080), 77(166), 78(4229), 79(11387) |
| unassigned | 88(199678) |

### seq_c

| role | fids(vocab) |
| --- | --- |
| item | 29(5764358) |
| action | 28(73) |
| stat | 30(846), 31(6805), 32(7), 33(5), 34(1031305), 35(2896), 36(977479), 37(9433) |
| unassigned | 47(86335515) |

### seq_d

| role | fids(vocab) |
| --- | --- |
| item | - |
| action | 17(5) |
| stat | 18(966), 19(3300), 20(10785), 21(4929), 22(404398), 23(606041), 24(531), 25(15) |
| unassigned | - |

## Immediate Readouts

- Current RankMixer chunks are mechanical embedding-dimension slices, not semantic groups.
- Any `partial_fids` means one fid embedding vector is split across two NS tokens.
- v7 dense grouping can be implemented from schema offsets without Parquet reads.
- v7 dense grouping changes `num_ns`, so `rank_mixer_mode=full` may require a different `d_model`.
- SemanticSeqEmbedder depends on reviving high-cardinality sequence item roles instead of direct `emb_skip_threshold` raise.
