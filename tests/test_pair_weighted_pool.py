"""Unit tests for pair-weighted pool in NSTokenizers (W2.6 重写).

See docs/superpowers/specs/2026-05-03-pair-feature-design.md for design.
"""
import torch
import torch.nn.functional as F

from model import GroupNSTokenizer, RankMixerNSTokenizer


def _make_simple_specs():
    """单 fid，vocab=10，length=4: 模拟 fid 62 简化版。

    feature_specs: List[Tuple[vocab_size, offset, length]]
    groups: List[List[int]]  # group of fid_idx into feature_specs
    """
    return [(10, 0, 4)], [[0]]


def test_uniform_mode_matches_baseline_mean_pool():
    """When weight_mode='uniform' or paired_dense=None, output must be bit-identical to current mean-pool."""
    torch.manual_seed(0)
    feature_specs, groups = _make_simple_specs()
    tok = GroupNSTokenizer(
        feature_specs=feature_specs, groups=groups,
        emb_dim=8, d_model=16, emb_skip_threshold=0,
    )
    tok.eval()

    int_feats = torch.tensor([[1, 2, 3, 0], [4, 0, 0, 0]], dtype=torch.long)

    out_baseline = tok(int_feats)
    out_new = tok(int_feats, paired_dense=None, weight_mode='uniform')

    assert torch.allclose(out_baseline, out_new, atol=1e-7), \
        "uniform mode must be bit-identical to baseline"


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
    out_new = tok(int_feats, paired_dense=None, weight_mode='uniform')
    assert torch.allclose(out_baseline, out_new, atol=1e-7)


def test_log1p_weighted_pool_correctness():
    """Hand-compute log1p-weighted pool and assert match (GroupNSTokenizer)."""
    torch.manual_seed(0)
    feature_specs, groups = _make_simple_specs()
    tok = GroupNSTokenizer(
        feature_specs=feature_specs, groups=groups,
        emb_dim=8, d_model=16, emb_skip_threshold=0,
    )
    tok.eval()

    int_feats = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)  # 3 valid, 1 pad
    vals = torch.tensor([[10.0, 20.0, 0.5, 999.0]])  # last position pad → ignored

    emb = tok.embs[0]
    e1, e2, e3 = emb(torch.tensor([1, 2, 3]))
    expected_w = torch.log1p(torch.tensor([10.0, 20.0, 0.5]))
    expected_pool = (expected_w[0] * e1 + expected_w[1] * e2 + expected_w[2] * e3) / expected_w.sum()
    expected_pool = expected_pool.unsqueeze(0)  # (1, emb_dim)
    import torch.nn.functional as F
    expected_token = F.silu(tok.group_projs[0](expected_pool)).unsqueeze(1)  # (1, 1, d_model)

    out = tok(int_feats, paired_dense={0: vals}, weight_mode='log1p')

    assert torch.allclose(out, expected_token, atol=1e-5), \
        f"log1p weighted pool mismatch: got {out}, expected {expected_token}"


def test_fallback_to_mean_pool_when_dense_all_zero():
    """When all dense values are 0 but ids are valid, fall back to uniform mean-pool."""
    torch.manual_seed(0)
    feature_specs, groups = _make_simple_specs()
    tok = GroupNSTokenizer(
        feature_specs=feature_specs, groups=groups,
        emb_dim=8, d_model=16, emb_skip_threshold=0,
    )
    tok.eval()

    int_feats = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    vals_all_zero = torch.tensor([[0.0, 0.0, 0.0, 0.0]])

    out_uniform_mode = tok(int_feats, paired_dense=None, weight_mode='uniform')
    out_log1p_mode = tok(int_feats, paired_dense={0: vals_all_zero}, weight_mode='log1p')

    assert torch.allclose(out_uniform_mode, out_log1p_mode, atol=1e-7), \
        "log1p with all-zero dense must fall back to mean-pool"


def test_unknown_user_returns_zero():
    """When ids are all padding, both modes should return identical embedding."""
    torch.manual_seed(0)
    feature_specs, groups = _make_simple_specs()
    tok = GroupNSTokenizer(
        feature_specs=feature_specs, groups=groups,
        emb_dim=8, d_model=16, emb_skip_threshold=0,
    )
    tok.eval()

    int_feats = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
    vals_anything = torch.tensor([[10.0, 20.0, 30.0, 40.0]])

    out_uniform = tok(int_feats, paired_dense=None, weight_mode='uniform')
    out_log1p = tok(int_feats, paired_dense={0: vals_anything}, weight_mode='log1p')

    # padding_idx=0 means emb(0)=0; mask is all-zero; both branches collapse to
    # zero pool input → identical projections.
    assert torch.allclose(out_uniform, out_log1p, atol=1e-7)


def test_partial_padding_no_pollution():
    """Padded positions must not contribute, even if dense vals at pad positions are huge."""
    torch.manual_seed(0)
    feature_specs, groups = _make_simple_specs()
    tok = GroupNSTokenizer(
        feature_specs=feature_specs, groups=groups,
        emb_dim=8, d_model=16, emb_skip_threshold=0,
    )
    tok.eval()

    int_feats = torch.tensor([[5, 7, 0, 0]], dtype=torch.long)  # 2 valid, 2 pad
    vals = torch.tensor([[1.0, 2.0, 1e6, 1e6]])  # padded positions have huge vals

    out = tok(int_feats, paired_dense={0: vals}, weight_mode='log1p')

    emb = tok.embs[0]
    e5, e7 = emb(torch.tensor([5, 7]))
    w = torch.log1p(torch.tensor([1.0, 2.0]))
    expected_pool = ((w[0] * e5 + w[1] * e7) / w.sum()).unsqueeze(0)
    import torch.nn.functional as F
    expected_token = F.silu(tok.group_projs[0](expected_pool)).unsqueeze(1)

    assert torch.allclose(out, expected_token, atol=1e-5), \
        f"padded positions polluted pool: got {out}, expected {expected_token}"


def test_numerical_stability_huge_values():
    """log1p must handle 1.5e9 max (fid 65/66) without inf/nan."""
    torch.manual_seed(0)
    feature_specs, groups = _make_simple_specs()
    tok = GroupNSTokenizer(
        feature_specs=feature_specs, groups=groups,
        emb_dim=8, d_model=16, emb_skip_threshold=0,
    )
    tok.eval()

    int_feats = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    vals = torch.tensor([[100.0, 1.5e9, 50.0, 0.0]])  # fid 65/66 max magnitude

    out = tok(int_feats, paired_dense={0: vals}, weight_mode='log1p')

    assert torch.isfinite(out).all(), f"Output must be finite (no inf/nan); got {out}"


def test_rankmixer_log1p_changes_output():
    """RankMixer log1p weighted pool must differ from uniform when vals are non-uniform."""
    torch.manual_seed(0)
    feature_specs, groups = _make_simple_specs()
    tok = RankMixerNSTokenizer(
        feature_specs=feature_specs, groups=groups,
        emb_dim=8, d_model=16, num_ns_tokens=1, emb_skip_threshold=0,
    )
    tok.eval()

    int_feats = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    vals = torch.tensor([[10.0, 20.0, 0.5, 0.0]])

    out_uniform = tok(int_feats, paired_dense=None, weight_mode='uniform')
    out_log1p = tok(int_feats, paired_dense={0: vals}, weight_mode='log1p')

    assert torch.isfinite(out_uniform).all() and torch.isfinite(out_log1p).all()
    assert not torch.allclose(out_uniform, out_log1p, atol=1e-5), \
        "log1p with non-uniform vals must differ from uniform mode in RankMixer"


def test_rankmixer_log1p_correctness_vs_groupns():
    """When configured with shared embedding weights, RankMixer's pre-projection
    embedding (i.e., the multi-value pool) must equal GroupNS's pool."""
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
    # Force shared embedding weights so the multi-value pool inputs are identical
    tok_r.embs[0].weight.data.copy_(tok_g.embs[0].weight.data)
    tok_g.eval(); tok_r.eval()

    int_feats = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    vals = torch.tensor([[10.0, 20.0, 0.5, 0.0]])

    # Hand-compute the pool (same for both tokenizers because logic is shared)
    emb = tok_g.embs[0]
    e1, e2, e3 = emb(torch.tensor([1, 2, 3]))
    w = torch.log1p(torch.tensor([10.0, 20.0, 0.5]))
    expected_pool = ((w[0] * e1 + w[1] * e2 + w[2] * e3) / w.sum()).detach()

    # Verify by hooking into RankMixer's internal cat (it cats per-group then projects).
    # Since num_ns_tokens=1 and we have a single 8-dim emb, the chunk_dim should be 8
    # and the projection is on the full 8-dim pool. Its pre-silu input should be
    # tok_r.token_projs[0]'s linear applied to expected_pool.
    out_r = tok_r(int_feats, paired_dense={0: vals}, weight_mode='log1p')
    expected_token_r = F.silu(tok_r.token_projs[0](expected_pool.unsqueeze(0))).unsqueeze(1)
    assert torch.allclose(out_r, expected_token_r, atol=1e-5), \
        f"RankMixer log1p mismatch: got {out_r}, expected {expected_token_r}"
