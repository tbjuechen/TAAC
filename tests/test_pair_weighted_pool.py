"""Unit tests for pair-weighted pool in NSTokenizers (W2.6 重写).

See docs/superpowers/specs/2026-05-03-pair-feature-design.md for design.

API after refactor: tokenizer.forward(int_feats, paired_dense=None) where
paired_dense is {fid_idx: (B, length) precomputed weight tensor}. Caller is
responsible for applying any transform (log1p / sigmoid / abs / ...) before
calling. Helper does pure weighted-mean + fallback.
"""
import torch
import torch.nn.functional as F

from model import GroupNSTokenizer, RankMixerNSTokenizer


def _make_simple_specs():
    """单 fid，vocab=10，length=4: 模拟 fid 62 简化版。"""
    return [(10, 0, 4)], [[0]]


# ─────────────────────────────────────────────────────────────────
# Equivalence: paired_dense=None must match baseline mean-pool
# ─────────────────────────────────────────────────────────────────

def test_uniform_mode_matches_baseline_mean_pool():
    """paired_dense=None must be bit-identical to current mean-pool."""
    torch.manual_seed(0)
    feature_specs, groups = _make_simple_specs()
    tok = GroupNSTokenizer(
        feature_specs=feature_specs, groups=groups,
        emb_dim=8, d_model=16, emb_skip_threshold=0,
    )
    tok.eval()
    int_feats = torch.tensor([[1, 2, 3, 0], [4, 0, 0, 0]], dtype=torch.long)
    out_baseline = tok(int_feats)
    out_new = tok(int_feats, paired_dense=None)
    assert torch.allclose(out_baseline, out_new, atol=1e-7)


def test_rankmixer_uniform_mode_matches_baseline():
    torch.manual_seed(0)
    feature_specs, groups = _make_simple_specs()
    tok = RankMixerNSTokenizer(
        feature_specs=feature_specs, groups=groups,
        emb_dim=8, d_model=16, num_ns_tokens=2, emb_skip_threshold=0,
    )
    tok.eval()
    int_feats = torch.tensor([[1, 2, 3, 0], [4, 0, 0, 0]], dtype=torch.long)
    out_baseline = tok(int_feats)
    out_new = tok(int_feats, paired_dense=None)
    assert torch.allclose(out_baseline, out_new, atol=1e-7)


# ─────────────────────────────────────────────────────────────────
# log1p path (caller pre-computes log1p weights)
# ─────────────────────────────────────────────────────────────────

def test_log1p_weighted_pool_correctness():
    """Hand-compute log1p-weighted pool and assert match (GroupNSTokenizer)."""
    torch.manual_seed(0)
    feature_specs, groups = _make_simple_specs()
    tok = GroupNSTokenizer(
        feature_specs=feature_specs, groups=groups,
        emb_dim=8, d_model=16, emb_skip_threshold=0,
    )
    tok.eval()

    int_feats = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    vals = torch.tensor([[10.0, 20.0, 0.5, 999.0]])
    weights = torch.log1p(vals.clamp(min=0))  # caller pre-computes

    emb = tok.embs[0]
    e1, e2, e3 = emb(torch.tensor([1, 2, 3]))
    expected_w = torch.log1p(torch.tensor([10.0, 20.0, 0.5]))
    expected_pool = ((expected_w[0] * e1 + expected_w[1] * e2 + expected_w[2] * e3) /
                     expected_w.sum()).unsqueeze(0)
    expected_token = F.silu(tok.group_projs[0](expected_pool)).unsqueeze(1)

    out = tok(int_feats, paired_dense={0: weights})
    assert torch.allclose(out, expected_token, atol=1e-5), \
        f"log1p mismatch: got {out}, expected {expected_token}"


def test_fallback_to_mean_pool_when_weight_all_zero():
    """All-zero precomputed weight (e.g. log1p of all-zero dense) → fall back to mean-pool."""
    torch.manual_seed(0)
    feature_specs, groups = _make_simple_specs()
    tok = GroupNSTokenizer(
        feature_specs=feature_specs, groups=groups,
        emb_dim=8, d_model=16, emb_skip_threshold=0,
    )
    tok.eval()
    int_feats = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    weights_zero = torch.zeros(1, 4)

    out_uniform = tok(int_feats, paired_dense=None)
    out_zero = tok(int_feats, paired_dense={0: weights_zero})
    assert torch.allclose(out_uniform, out_zero, atol=1e-7), \
        "all-zero weights must fall back to mean-pool"


def test_unknown_user_returns_zero():
    """ids all padding: both paths give identical (uniform_pool=0) output."""
    torch.manual_seed(0)
    feature_specs, groups = _make_simple_specs()
    tok = GroupNSTokenizer(
        feature_specs=feature_specs, groups=groups,
        emb_dim=8, d_model=16, emb_skip_threshold=0,
    )
    tok.eval()
    int_feats = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
    weights_anything = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    out_uniform = tok(int_feats, paired_dense=None)
    out_weighted = tok(int_feats, paired_dense={0: weights_anything})
    assert torch.allclose(out_uniform, out_weighted, atol=1e-7)


def test_partial_padding_no_pollution():
    """Padded positions must not contribute, even with huge precomputed weights at pad."""
    torch.manual_seed(0)
    feature_specs, groups = _make_simple_specs()
    tok = GroupNSTokenizer(
        feature_specs=feature_specs, groups=groups,
        emb_dim=8, d_model=16, emb_skip_threshold=0,
    )
    tok.eval()
    int_feats = torch.tensor([[5, 7, 0, 0]], dtype=torch.long)
    # caller pre-computed log1p; pad positions have huge transformed weights
    w1 = torch.log1p(torch.tensor(1.0)).item()
    w2 = torch.log1p(torch.tensor(2.0)).item()
    weights = torch.tensor([[w1, w2, 1e6, 1e6]])

    out = tok(int_feats, paired_dense={0: weights})

    emb = tok.embs[0]
    e5, e7 = emb(torch.tensor([5, 7]))
    w = torch.log1p(torch.tensor([1.0, 2.0]))
    expected_pool = ((w[0] * e5 + w[1] * e7) / w.sum()).unsqueeze(0)
    expected_token = F.silu(tok.group_projs[0](expected_pool)).unsqueeze(1)
    assert torch.allclose(out, expected_token, atol=1e-5)


def test_numerical_stability_huge_log1p_weights():
    """log1p(1.5e9) ≈ 21.13 must produce finite output."""
    torch.manual_seed(0)
    feature_specs, groups = _make_simple_specs()
    tok = GroupNSTokenizer(
        feature_specs=feature_specs, groups=groups,
        emb_dim=8, d_model=16, emb_skip_threshold=0,
    )
    tok.eval()
    int_feats = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    vals = torch.tensor([[100.0, 1.5e9, 50.0, 0.0]])  # fid 65/66 max
    weights = torch.log1p(vals.clamp(min=0))

    out = tok(int_feats, paired_dense={0: weights})
    assert torch.isfinite(out).all()


# ─────────────────────────────────────────────────────────────────
# RankMixer parity
# ─────────────────────────────────────────────────────────────────

def test_rankmixer_log1p_changes_output():
    torch.manual_seed(0)
    feature_specs, groups = _make_simple_specs()
    tok = RankMixerNSTokenizer(
        feature_specs=feature_specs, groups=groups,
        emb_dim=8, d_model=16, num_ns_tokens=1, emb_skip_threshold=0,
    )
    tok.eval()
    int_feats = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    vals = torch.tensor([[10.0, 20.0, 0.5, 0.0]])
    weights = torch.log1p(vals.clamp(min=0))

    out_uniform = tok(int_feats, paired_dense=None)
    out_log1p = tok(int_feats, paired_dense={0: weights})
    assert torch.isfinite(out_uniform).all() and torch.isfinite(out_log1p).all()
    assert not torch.allclose(out_uniform, out_log1p, atol=1e-5)


def test_rankmixer_log1p_correctness_vs_groupns():
    torch.manual_seed(0)
    feature_specs, groups = _make_simple_specs()
    tok_g = GroupNSTokenizer(
        feature_specs=feature_specs, groups=groups,
        emb_dim=8, d_model=16, emb_skip_threshold=0,
    )
    tok_r = RankMixerNSTokenizer(
        feature_specs=feature_specs, groups=groups,
        emb_dim=8, d_model=16, num_ns_tokens=1, emb_skip_threshold=0,
    )
    tok_r.embs[0].weight.data.copy_(tok_g.embs[0].weight.data)
    tok_g.eval(); tok_r.eval()

    int_feats = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    vals = torch.tensor([[10.0, 20.0, 0.5, 0.0]])
    weights = torch.log1p(vals.clamp(min=0))

    emb = tok_g.embs[0]
    e1, e2, e3 = emb(torch.tensor([1, 2, 3]))
    w = torch.log1p(torch.tensor([10.0, 20.0, 0.5]))
    expected_pool = ((w[0] * e1 + w[1] * e2 + w[2] * e3) / w.sum()).detach()

    out_r = tok_r(int_feats, paired_dense={0: weights})
    expected_token_r = F.silu(tok_r.token_projs[0](expected_pool.unsqueeze(0))).unsqueeze(1)
    assert torch.allclose(out_r, expected_token_r, atol=1e-5)


# ─────────────────────────────────────────────────────────────────
# fid 89-91 path: sigmoid weights
# ─────────────────────────────────────────────────────────────────

def test_sigmoid_weighted_pool_correctness():
    """Hand-compute sigmoid-weighted pool with bounded ~[-1,+1] vals."""
    torch.manual_seed(0)
    feature_specs, groups = _make_simple_specs()
    tok = GroupNSTokenizer(
        feature_specs=feature_specs, groups=groups,
        emb_dim=8, d_model=16, emb_skip_threshold=0,
    )
    tok.eval()
    int_feats = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    vals = torch.tensor([[-0.5, 0.3, 0.9, 0.0]])  # fid 89-91 typical range
    weights = torch.sigmoid(vals)

    emb = tok.embs[0]
    e1, e2, e3 = emb(torch.tensor([1, 2, 3]))
    w = torch.sigmoid(torch.tensor([-0.5, 0.3, 0.9]))
    expected_pool = ((w[0] * e1 + w[1] * e2 + w[2] * e3) / w.sum()).unsqueeze(0)
    expected_token = F.silu(tok.group_projs[0](expected_pool)).unsqueeze(1)

    out = tok(int_feats, paired_dense={0: weights})
    assert torch.allclose(out, expected_token, atol=1e-5)


def test_sigmoid_negative_values_still_contribute():
    """Sigmoid maps neg vals to small positive weights → pool != mean-pool."""
    torch.manual_seed(0)
    feature_specs, groups = _make_simple_specs()
    tok = GroupNSTokenizer(
        feature_specs=feature_specs, groups=groups,
        emb_dim=8, d_model=16, emb_skip_threshold=0,
    )
    tok.eval()
    int_feats = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    vals_all_neg = torch.tensor([[-0.5, -0.3, -0.9, 0.0]])
    weights = torch.sigmoid(vals_all_neg)

    out_uniform = tok(int_feats, paired_dense=None)
    out_sig = tok(int_feats, paired_dense={0: weights})
    assert not torch.allclose(out_uniform, out_sig, atol=1e-5)
    assert torch.isfinite(out_sig).all()


# ─────────────────────────────────────────────────────────────────
# PCVRHyFormer integration: pair_weight_mode dispatch
# ─────────────────────────────────────────────────────────────────

def _make_pcvr_inputs(B=2):
    from model import ModelInput
    return ModelInput(
        # fid 1=3; fid 62=[1,2,3,0]; fid 89=[5,6,7]
        user_int_feats=torch.tensor(
            [[3, 1, 2, 3, 0, 5, 6, 7],
             [5, 4, 0, 0, 0, 8, 9, 0]],
            dtype=torch.long),
        item_int_feats=torch.tensor([[7], [9]], dtype=torch.long),
        # 4 vals for fid 62 + 3 vals for fid 89
        user_dense_feats=torch.tensor(
            [[10.0, 20.0, 5.0, 0.0,  -0.5, 0.3, 0.9],
             [1.0, 0.0, 0.0, 0.0,    0.2, -0.4, 0.0]]),
        item_dense_feats=torch.zeros(B, 0),
        seq_data={'a': torch.zeros(B, 2, 3, dtype=torch.long)},
        seq_lens={'a': torch.tensor([2, 1], dtype=torch.long)},
        seq_time_buckets={'a': torch.zeros(B, 3, dtype=torch.long)},
    )


def _make_pcvr(pair_weight_mode='uniform', user_paired_dense_specs=None):
    from model import PCVRHyFormer
    return PCVRHyFormer(
        # fid 1 single, fid 62 multi (len=4), fid 89 multi (len=3)
        user_int_feature_specs=[(10, 0, 1), (5, 1, 4), (10, 5, 3)],
        item_int_feature_specs=[(20, 0, 1)],
        user_dense_dim=7,
        item_dense_dim=0,
        seq_vocab_sizes={'a': [10, 10]},
        # 1 user group; 1 item group; 1 user_dense token; 0 item_dense
        # T = 1*1 + (1+1+1) = 4 → divides d_model=16
        user_ns_groups=[[0, 1, 2]], item_ns_groups=[[0]],
        d_model=16, emb_dim=8, num_queries=1,
        num_hyformer_blocks=1, num_heads=2,
        num_time_buckets=0,
        ns_tokenizer_type='group',
        user_paired_dense_specs=user_paired_dense_specs,
        user_int_fids=[1, 62, 89],
        pair_weight_mode=pair_weight_mode,
    )


def test_pcvr_uniform_mode_baseline():
    """PCVRHyFormer with pair_weight_mode='uniform' must equal baseline (no specs)."""
    torch.manual_seed(42)
    m_baseline = _make_pcvr(pair_weight_mode='uniform')
    m_baseline.eval()
    torch.manual_seed(42)
    m_with_specs = _make_pcvr(
        pair_weight_mode='uniform',
        user_paired_dense_specs={62: (0, 4), 89: (4, 3)},
    )
    m_with_specs.eval()
    m_with_specs.load_state_dict(m_baseline.state_dict(), strict=False)

    inputs = _make_pcvr_inputs()
    with torch.no_grad():
        out_b = m_baseline(inputs)
        out_w = m_with_specs(inputs)
    assert torch.allclose(out_b, out_w, atol=1e-7)


def test_pcvr_log1p_only_count_fids():
    """log1p mode: only fid 62 (count) gets transformed; fid 89 (score) stays mean-pool."""
    torch.manual_seed(42)
    m_log1p = _make_pcvr(
        pair_weight_mode='log1p',
        user_paired_dense_specs={62: (0, 4), 89: (4, 3)},
    )
    m_log1p.eval()

    # fid_idx 1 (=fid 62) in count slice; fid_idx 2 (=fid 89) NOT in score slice (log1p mode)
    assert 1 in m_log1p._paired_count_idx_to_slice
    assert 2 not in m_log1p._paired_score_idx_to_slice

    inputs = _make_pcvr_inputs()
    with torch.no_grad():
        out_log1p = m_log1p(inputs)
    assert torch.isfinite(out_log1p).all()


def test_pcvr_full_mode_both_fid_groups():
    """full mode: fid 62 → log1p, fid 89 → sigmoid, both must be active."""
    torch.manual_seed(42)
    m_full = _make_pcvr(
        pair_weight_mode='full',
        user_paired_dense_specs={62: (0, 4), 89: (4, 3)},
    )
    m_full.eval()

    assert 1 in m_full._paired_count_idx_to_slice  # fid 62 → log1p
    assert 2 in m_full._paired_score_idx_to_slice  # fid 89 → sigmoid

    inputs = _make_pcvr_inputs()
    with torch.no_grad():
        out_full = m_full(inputs)
    assert torch.isfinite(out_full).all()

    # full mode must differ from log1p-only mode (because fid 89 is also weighted)
    torch.manual_seed(42)
    m_log1p = _make_pcvr(
        pair_weight_mode='log1p',
        user_paired_dense_specs={62: (0, 4), 89: (4, 3)},
    )
    m_log1p.eval()
    m_log1p.load_state_dict(m_full.state_dict(), strict=False)
    with torch.no_grad():
        out_log1p = m_log1p(inputs)
    assert not torch.allclose(out_full, out_log1p, atol=1e-6), \
        "full mode must differ from log1p mode when fid 89 has non-zero dense values"


# ─────────────────────────────────────────────────────────────────
# W2.6 v2: PairSetEncoder (bucket + 1-layer transformer + mean pool)
# See docs/superpowers/specs/2026-05-03-pair-set-encoder-design.md
# ─────────────────────────────────────────────────────────────────

def test_pair_set_encoder_forward_shape_count_fid():
    """PairSetEncoder for COUNT fid (62-66) outputs (B, emb_dim)."""
    from model import PairSetEncoder
    torch.manual_seed(0)
    enc = PairSetEncoder(fid=62, vocab=10, emb_dim=8, nhead=2)
    enc.eval()
    ids = torch.tensor([[1, 2, 3, 0], [4, 0, 0, 0]], dtype=torch.long)
    vals = torch.tensor([[10.0, 20.0, 5.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    out = enc(ids, vals)
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_pair_set_encoder_forward_shape_score_fid():
    """PairSetEncoder for SCORE fid (89-91) outputs (B, emb_dim)."""
    from model import PairSetEncoder
    torch.manual_seed(0)
    enc = PairSetEncoder(fid=89, vocab=10, emb_dim=8, nhead=2)
    enc.eval()
    ids = torch.tensor([[5, 6, 7], [8, 9, 0]], dtype=torch.long)
    vals = torch.tensor([[-0.5, 0.3, 0.9], [0.2, -0.4, 0.0]])
    out = enc(ids, vals)
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_pair_set_encoder_quantize_count_path():
    """COUNT fids: log1p(v) buckets — v=0 → bucket 0; large v → high bucket; v<0 clamped."""
    from model import PairSetEncoder
    enc = PairSetEncoder(fid=62, vocab=10, emb_dim=8, nhead=2)
    vals = torch.tensor([[0.0, 1e8, -1.0, 1.0]])
    bucket = enc._quantize(vals)
    # log1p(0)=0 → bucket 0
    assert bucket[0, 0].item() == 0
    # log1p(1)≈0.69 → 0.69/24*32 ≈ 0.92 → floor=0
    assert bucket[0, 3].item() == 0
    # log1p(-1) clamped to log1p(0)=0 → bucket 0
    assert bucket[0, 2].item() == 0
    # log1p(1e8)≈18.42 → 18.42/24*32 ≈ 24.55 → floor=24
    assert bucket[0, 1].item() == 24


def test_pair_set_encoder_quantize_score_path():
    """SCORE fids: (v+1)/2 buckets — v=0 → middle bucket; v=±1 → edge buckets."""
    from model import PairSetEncoder
    enc = PairSetEncoder(fid=89, vocab=10, emb_dim=8, nhead=2)
    vals = torch.tensor([[0.0, -1.0, 1.0, -0.5, 0.5]])
    bucket = enc._quantize(vals)
    # v=0 → (0+1)/2*32=16 → bucket 16
    assert bucket[0, 0].item() == 16
    # v=-1 → 0*32=0 → bucket 0
    assert bucket[0, 1].item() == 0
    # v=+1 → 1*32=32 → clamped to 31
    assert bucket[0, 2].item() == 31
    # v=-0.5 → 0.25*32=8 → bucket 8
    assert bucket[0, 3].item() == 8
    # v=+0.5 → 0.75*32=24 → bucket 24
    assert bucket[0, 4].item() == 24


def test_pair_set_encoder_all_padding_row_no_nan():
    """Fully-padded row (id=0 everywhere) must produce finite zero output (no NaN)."""
    from model import PairSetEncoder
    torch.manual_seed(0)
    enc = PairSetEncoder(fid=91, vocab=10, emb_dim=8, nhead=2)
    enc.eval()
    # Row 0: all valid; Row 1: all padding (simulates fid 91's 48% all_zero case)
    ids = torch.tensor([[1, 2, 3], [0, 0, 0]], dtype=torch.long)
    vals = torch.tensor([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]])
    out = enc(ids, vals)
    assert torch.isfinite(out).all(), "all-padded row must not produce NaN"
    # Fully-padded row mean-pool denom clamp(min=1) → output is masked_sum/1, masked_sum=0
    # so output should be all zeros pre-projection. Post-projection: bias term only.
    # We check the row is finite and bounded (smoke test).
    assert out[1].abs().max() < 100


def test_pair_set_encoder_padding_does_not_affect_valid_rows():
    """Adding padded positions to a row must not change pool of valid positions."""
    from model import PairSetEncoder
    torch.manual_seed(0)
    enc = PairSetEncoder(fid=62, vocab=10, emb_dim=8, nhead=2)
    enc.eval()
    # Two equivalent inputs: (a) length=3 valid; (b) length=5 with 2 padding tail
    ids_a = torch.tensor([[1, 2, 3]], dtype=torch.long)
    vals_a = torch.tensor([[10.0, 20.0, 5.0]])
    out_a = enc(ids_a, vals_a)

    ids_b = torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.long)
    vals_b = torch.tensor([[10.0, 20.0, 5.0, 0.0, 0.0]])
    out_b = enc(ids_b, vals_b)
    # Mean pool should be the same (padded positions excluded)
    # Note: transformer self-attn output for valid positions may differ slightly because
    # valid tokens see padded positions as keys (masked OK) — but mask should hide them.
    # Test: out_a ≈ out_b within tolerance.
    assert torch.allclose(out_a, out_b, atol=1e-5), \
        "padding tail must not change valid positions' mean pool"


# ─────────────────────────────────────────────────────────────────
# PCVRHyFormer integration: pair_weight_mode='transformer'
# ─────────────────────────────────────────────────────────────────

def test_pcvr_transformer_mode_instantiates_encoders():
    """transformer mode must instantiate nn.ModuleDict of PairSetEncoder."""
    torch.manual_seed(42)
    m = _make_pcvr(
        pair_weight_mode='transformer',
        user_paired_dense_specs={62: (0, 4), 89: (4, 3)},
    )
    # Both COUNT (62) and SCORE (89) should be in slice dicts under transformer mode
    assert 1 in m._paired_count_idx_to_slice  # fid_idx for fid 62
    assert 2 in m._paired_score_idx_to_slice  # fid_idx for fid 89
    # PairSetEncoder dict should contain both
    assert '1' in m.pair_set_encoders
    assert '2' in m.pair_set_encoders
    assert len(m.pair_set_encoders) == 2


def test_pcvr_transformer_mode_forward_finite():
    """transformer mode forward produces finite output."""
    torch.manual_seed(42)
    m = _make_pcvr(
        pair_weight_mode='transformer',
        user_paired_dense_specs={62: (0, 4), 89: (4, 3)},
    )
    m.eval()
    inputs = _make_pcvr_inputs()
    with torch.no_grad():
        out = m(inputs)
    assert torch.isfinite(out).all()


def test_pcvr_transformer_mode_differs_from_baseline():
    """transformer mode output differs from uniform mode (PairSetEncoder applied)."""
    torch.manual_seed(42)
    m_baseline = _make_pcvr(pair_weight_mode='uniform')
    m_baseline.eval()
    torch.manual_seed(42)
    m_trans = _make_pcvr(
        pair_weight_mode='transformer',
        user_paired_dense_specs={62: (0, 4), 89: (4, 3)},
    )
    m_trans.eval()

    inputs = _make_pcvr_inputs()
    with torch.no_grad():
        out_b = m_baseline(inputs)
        out_t = m_trans(inputs)
    # transformer mode adds extra params (PairSetEncoder), output must differ.
    assert not torch.allclose(out_b, out_t, atol=1e-5), \
        "transformer mode must produce different output than baseline mean-pool"


def test_pcvr_uniform_mode_no_pair_set_encoders():
    """uniform/none mode must NOT instantiate any PairSetEncoder (zero regression)."""
    m_uniform = _make_pcvr(pair_weight_mode='uniform')
    m_none = _make_pcvr(
        pair_weight_mode='uniform',
        user_paired_dense_specs={62: (0, 4), 89: (4, 3)},
    )
    assert len(m_uniform.pair_set_encoders) == 0
    assert len(m_none.pair_set_encoders) == 0


def test_pcvr_log1p_mode_no_pair_set_encoders():
    """log1p mode must NOT instantiate any PairSetEncoder (only v1 weighted-pool)."""
    m = _make_pcvr(
        pair_weight_mode='log1p',
        user_paired_dense_specs={62: (0, 4), 89: (4, 3)},
    )
    assert len(m.pair_set_encoders) == 0


# ─────────────────────────────────────────────────────────────────
# W2.6 v2: id_emb sharing with tokenizer + reset behavior (F15 interaction)
# ─────────────────────────────────────────────────────────────────

def test_pcvr_transformer_mode_shares_id_emb_with_tokenizer():
    """v2 PairSetEncoder.id_emb must be the SAME tensor as tokenizer.embs[fid_idx].

    Required so the existing reinit_high_cardinality_params path resets it (F15).
    """
    torch.manual_seed(42)
    m = _make_pcvr(
        pair_weight_mode='transformer',
        user_paired_dense_specs={62: (0, 4), 89: (4, 3)},
    )
    for fid_idx_str, encoder in m.pair_set_encoders.items():
        fid_idx = int(fid_idx_str)
        real_idx = m.user_ns_tokenizer._emb_index[fid_idx]
        tok_emb = m.user_ns_tokenizer.embs[real_idx]
        # PairSetEncoder must use tokenizer's emb via shared list (no own id_emb)
        assert encoder._shared_id_emb is not None
        assert encoder.id_emb is None
        assert encoder._shared_id_emb[0] is tok_emb, \
            f"fid_idx {fid_idx} shared_id_emb must be SAME object as tokenizer's emb"
        # data_ptr equal confirms no copy
        assert encoder._shared_id_emb[0].weight.data_ptr() == tok_emb.weight.data_ptr()


def test_pcvr_transformer_mode_id_emb_reset_via_reinit_path():
    """After reinit_high_cardinality_params(0), shared id_emb is reset (xavier);
    bucket_emb (owned) is NOT reset.
    """
    torch.manual_seed(42)
    m = _make_pcvr(
        pair_weight_mode='transformer',
        user_paired_dense_specs={62: (0, 4), 89: (4, 3)},
    )

    # Snapshot before reinit
    encoder = m.pair_set_encoders['1']  # fid 62
    id_emb_before = encoder._shared_id_emb[0].weight.data.clone()
    bucket_emb_before = encoder.bucket_emb.weight.data.clone()
    transformer_w_before = encoder.transformer.linear1.weight.data.clone()

    # Run reinit with threshold=0 (baseline behavior: vs > 0 → reset all built emb)
    reinit_ptrs = m.reinit_high_cardinality_params(cardinality_threshold=0)

    id_emb_after = encoder._shared_id_emb[0].weight.data
    bucket_emb_after = encoder.bucket_emb.weight.data
    transformer_w_after = encoder.transformer.linear1.weight.data

    # id_emb (shared with tokenizer) MUST be reset
    assert encoder._shared_id_emb[0].weight.data_ptr() in reinit_ptrs, \
        "id_emb (shared with tokenizer) must be in reinit_ptrs"
    assert not torch.allclose(id_emb_before, id_emb_after), \
        "shared id_emb must change after reinit (xavier_normal_)"

    # bucket_emb MUST NOT be reset (owned by PairSetEncoder, analogous to time_embedding)
    assert encoder.bucket_emb.weight.data_ptr() not in reinit_ptrs, \
        "bucket_emb must NOT be in reinit_ptrs (analogous to time_embedding)"
    assert torch.allclose(bucket_emb_before, bucket_emb_after), \
        "bucket_emb must be unchanged after reinit"

    # Transformer weights are dense params, NOT touched by reinit
    assert torch.allclose(transformer_w_before, transformer_w_after), \
        "transformer block weights must be unchanged after reinit"


def test_pair_set_encoder_no_double_param_registration():
    """When PairSetEncoder uses shared id_emb, that emb's params should appear ONCE
    in the parent model's parameters() (i.e. only via tokenizer, not via encoder).
    """
    torch.manual_seed(42)
    m = _make_pcvr(
        pair_weight_mode='transformer',
        user_paired_dense_specs={62: (0, 4), 89: (4, 3)},
    )
    # Collect all param data_ptrs and count duplicates.
    ptrs = [p.data_ptr() for p in m.parameters()]
    dup = [p for p in set(ptrs) if ptrs.count(p) > 1]
    assert not dup, f"Duplicate params (would be optimized 2x): {dup}"


def test_pair_set_encoder_standalone_owns_id_emb():
    """When id_emb_module is None, PairSetEncoder owns its own id_emb (test path)."""
    from model import PairSetEncoder
    enc = PairSetEncoder(fid=62, vocab=10, emb_dim=8, nhead=2)
    assert enc.id_emb is not None
    assert enc._shared_id_emb is None
    # id_emb appears in parameters
    ptrs = [p.data_ptr() for p in enc.parameters()]
    assert enc.id_emb.weight.data_ptr() in ptrs


def test_pair_set_encoder_shared_id_emb_path():
    """When id_emb_module is provided, PairSetEncoder uses it without registering."""
    from model import PairSetEncoder
    shared = torch.nn.Embedding(11, 8, padding_idx=0)
    enc = PairSetEncoder(fid=62, vocab=10, emb_dim=8, nhead=2, id_emb_module=shared)
    assert enc.id_emb is None
    assert enc._shared_id_emb is not None
    # Shared emb's params NOT in encoder's parameters
    ptrs = [p.data_ptr() for p in enc.parameters()]
    assert shared.weight.data_ptr() not in ptrs, \
        "shared id_emb must not be registered as encoder's submodule"
    # But forward still works using the shared table
    enc.eval()
    ids = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    vals = torch.tensor([[10.0, 20.0, 5.0, 0.0]])
    out = enc(ids, vals)
    assert out.shape == (1, 8)
    assert torch.isfinite(out).all()
