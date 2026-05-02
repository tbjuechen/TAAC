"""Unit tests for pair-weighted pool in NSTokenizers (W2.6 重写).

See docs/superpowers/specs/2026-05-03-pair-feature-design.md for design.
"""
import torch

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
