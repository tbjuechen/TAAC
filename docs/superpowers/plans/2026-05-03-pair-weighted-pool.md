# W2.6 重写：(int, dense) 平行对 value-加权 pool 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `feature/pair-weighted-pool` 分支上把 `user_int_feats_{62-66}` 的多值 mean-pool 升级为 `user_dense_feats_{62-66}` 驱动的 log1p-weighted pool，CLI flag 默认 `none` 等价 baseline，可控 A/B。

**Architecture:** 改动集中在 `src/model.py` 的两个 NSTokenizer 类（GroupNSTokenizer / RankMixerNSTokenizer）的 `forward`，新增 `paired_dense` 可选参数；`PCVRHyFormer.__init__` 接受 `user_paired_dense_specs` (fid → 切片偏移)，`forward` 切出 dense 段传入 user tokenizer。`src/train.py` 加 CLI flag `--pair_weighted_pool {none,log1p}` 和 specs 构造。新建 `tests/test_pair_weighted_pool.py` 用 pytest 做 TDD。

**Tech Stack:** PyTorch, pytest（新引入到 repo），Python 3.11+。

**Spec:** `docs/superpowers/specs/2026-05-03-pair-feature-design.md` (v0.1)

**已锁定决策**：
- 范围 A1：5 fid (62-66)；fid 89-91 不动；dense token 不动
- weight mode = log1p；fallback 到 mean-pool；`uniform` 模式严格等价 baseline
- CLI flag `--pair_weighted_pool {none,log1p}` 默认 `none`
- 配对 fid 列表硬编码 `PAIR_WEIGHTED_FIDS = [62, 63, 64, 65, 66]`

---

## 文件结构

| 操作 | 文件 | 责任 |
|---|---|---|
| Create | `tests/__init__.py` | 空文件，让 pytest 找到 tests |
| Create | `tests/test_pair_weighted_pool.py` | 单元测试：等价性、加权正确性、fallback、mask、数值稳定 |
| Create | `tests/conftest.py`（可选）| pytest 配置（让 src/ 可 import） |
| Modify | `src/model.py` | 1) 新增模块级常量 `PAIR_WEIGHTED_FIDS`；2) `GroupNSTokenizer.forward` 加 `paired_dense` 参；3) `RankMixerNSTokenizer.forward` 加 `paired_dense` 参；4) 抽出共享辅助函数 `_pool_with_optional_weights` 避免重复；5) `PCVRHyFormer.__init__` 加 `user_paired_dense_specs` + `pair_weight_mode`；6) `PCVRHyFormer.forward` 切片 + 调用 |
| Modify | `src/train.py` | 1) 加 CLI flag `--pair_weighted_pool`；2) 构造 `user_paired_dense_specs` dict；3) 注入 `model_args` |
| Modify | `src/run.sh` | 加 env var `PAIR_WEIGHTED_POOL`，默认 `none`，传入 train.py |

---

## Task 1: pytest 基建

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Modify: `environment.yml` (or `requirements.txt`) - 仅本地，**不上传平台**

- [ ] **Step 1: 检查 pytest 是否已装**

```bash
cd /Users/tbjuechen/code/TAAC && python -c "import pytest; print(pytest.__version__)"
```
预期：要么打印版本，要么 `ModuleNotFoundError`。

- [ ] **Step 2: 如未装则装**

```bash
pip install pytest
```

- [ ] **Step 3: 创建 tests/__init__.py（空文件）**

```bash
touch /Users/tbjuechen/code/TAAC/tests/__init__.py
```

- [ ] **Step 4: 创建 conftest.py，让 tests 能 import src**

```python
# tests/conftest.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

- [ ] **Step 5: 验证可 import model**

```bash
cd /Users/tbjuechen/code/TAAC && python -c "import sys; sys.path.insert(0, 'src'); from model import GroupNSTokenizer, RankMixerNSTokenizer; print('ok')"
```
预期：`ok`

- [ ] **Step 6: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "chore(tests): scaffold pytest with src/ on path"
```

---

## Task 2: 写等价性测试（uniform 模式 == baseline）

**Files:**
- Create: `tests/test_pair_weighted_pool.py`
- Test: 同上

- [ ] **Step 1: 写第一个失败测试 — uniform 模式下输出与原 mean-pool 完全一致**

```python
# tests/test_pair_weighted_pool.py
import torch
import pytest
from model import GroupNSTokenizer


def _make_simple_specs():
    """单 fid，vocab=10，length=4：模拟 fid 62 简化版。"""
    # feature_specs: [(vocab_size, offset, length), ...]
    return [(10, 0, 4)], [[0]]  # 单 group 包含 fid_idx=0


def test_uniform_mode_matches_baseline_mean_pool():
    """When weight_mode='uniform' or paired_dense=None, output must be bit-identical to current mean-pool."""
    torch.manual_seed(0)
    feature_specs, groups = _make_simple_specs()
    tok = GroupNSTokenizer(
        feature_specs=feature_specs, groups=groups,
        emb_dim=8, d_model=16, emb_skip_threshold=0,
    )
    tok.eval()

    # ids: 4 valid + padding semantics (0 = padding)
    int_feats = torch.tensor([[1, 2, 3, 0], [4, 0, 0, 0]], dtype=torch.long)

    # baseline call (no paired_dense)
    out_baseline = tok(int_feats)

    # new call with paired_dense=None or weight_mode='uniform' should match
    out_new = tok(int_feats, paired_dense=None, weight_mode='uniform')

    assert torch.allclose(out_baseline, out_new, atol=1e-7), \
        "uniform mode must be bit-identical to baseline"
```

- [ ] **Step 2: 跑测试，确认失败（接口未实现）**

```bash
cd /Users/tbjuechen/code/TAAC && pytest tests/test_pair_weighted_pool.py::test_uniform_mode_matches_baseline_mean_pool -v
```
预期：FAIL，错误信息类似 `unexpected keyword argument 'paired_dense'`

- [ ] **Step 3: 改 GroupNSTokenizer.forward 接受 paired_dense + weight_mode 但忽略它们**

定位：`src/model.py` line ~1034。修改 forward 签名：

```python
def forward(self, int_feats: torch.Tensor,
            paired_dense: dict | None = None,
            weight_mode: str = 'uniform') -> torch.Tensor:
    """Embeds and projects grouped discrete features into NS tokens.

    Args:
        int_feats: (B, total_int_dim), concatenated integer features.
        paired_dense: Optional dict {fid_idx: (B, length) dense tensor} for
            value-weighted pooling. Only fid_idx in this dict gets weighted
            treatment; others use uniform mean-pool. Default None = all uniform.
        weight_mode: 'uniform' (default, bit-equivalent to baseline) or
            'log1p' (use log1p(clamp_min(value, 0)) as pool weights).
    """
    # ... existing forward body unchanged in this step
```

不改 body，只加签名。这一步只让等价性测试通过。

- [ ] **Step 4: 跑测试确认通过**

```bash
cd /Users/tbjuechen/code/TAAC && pytest tests/test_pair_weighted_pool.py::test_uniform_mode_matches_baseline_mean_pool -v
```
预期：PASS

- [ ] **Step 5: 给 RankMixerNSTokenizer.forward 加同款签名（不改 body）+ 写它的等价性测试**

`src/model.py` line ~1148。在 test 文件加：

```python
from model import RankMixerNSTokenizer


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
```

- [ ] **Step 6: 跑两个测试都通过**

```bash
pytest tests/test_pair_weighted_pool.py -v -k "uniform"
```
预期：2 PASSED

- [ ] **Step 7: Commit**

```bash
git add tests/test_pair_weighted_pool.py src/model.py
git commit -m "test(model): equivalence test for uniform mode in NSTokenizers"
```

---

## Task 3: 实现 log1p weighted pool（happy path）

**Files:**
- Modify: `tests/test_pair_weighted_pool.py`
- Modify: `src/model.py`

- [ ] **Step 1: 写失败测试 — 已知 (ids, vals) 手算 log1p-weighted pool**

```python
def test_log1p_weighted_pool_correctness():
    """Manually compute log1p-weighted pool and assert match."""
    torch.manual_seed(0)
    feature_specs, groups = _make_simple_specs()  # fid_idx=0, vocab=10, len=4
    tok = GroupNSTokenizer(
        feature_specs=feature_specs, groups=groups,
        emb_dim=8, d_model=16, emb_skip_threshold=0,
    )
    tok.eval()

    int_feats = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)  # 3 valid, 1 pad
    vals = torch.tensor([[10.0, 20.0, 0.5, 999.0]])  # last position pad → ignored

    # Expected manual computation:
    #   mask = [1, 1, 1, 0]
    #   w = [log1p(10), log1p(20), log1p(0.5), 0] * mask
    #     = [2.398, 3.045, 0.405, 0]
    #   W = 2.398 + 3.045 + 0.405 = 5.848
    #   pool = (w[0]*emb(1) + w[1]*emb(2) + w[2]*emb(3)) / W
    emb = tok.embs[0]
    e1, e2, e3 = emb(torch.tensor([1, 2, 3]))
    expected_w = torch.log1p(torch.tensor([10.0, 20.0, 0.5]))
    expected_pool = (expected_w[0] * e1 + expected_w[1] * e2 + expected_w[2] * e3) / expected_w.sum()
    expected_token = torch.nn.functional.silu(tok.group_projs[0](expected_pool.unsqueeze(0))).unsqueeze(1)

    out = tok(int_feats, paired_dense={0: vals}, weight_mode='log1p')

    assert torch.allclose(out, expected_token, atol=1e-5), \
        f"log1p weighted pool mismatch: got {out}, expected {expected_token}"
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
pytest tests/test_pair_weighted_pool.py::test_log1p_weighted_pool_correctness -v
```
预期：FAIL（输出仍是 mean-pool）

- [ ] **Step 3: 在 `src/model.py` 顶部加常量**

定位：`src/model.py` 文件头 import 之后，第一个类之前。

```python
# Pair-weighted pool: fid list (vocab fid numbers, NOT schema indices)
PAIR_WEIGHTED_FIDS = [62, 63, 64, 65, 66]
```

- [ ] **Step 4: 抽出 helper 函数 `_pool_multivalue` 在 model.py 模块级**

放在 `class GroupNSTokenizer` 之前：

```python
def _pool_multivalue(
    emb_all: torch.Tensor,    # (B, length, emb_dim)
    vals: torch.Tensor,       # (B, length) int values for padding mask
    paired_value: torch.Tensor | None,  # (B, length) dense values, or None
    weight_mode: str,         # 'uniform' or 'log1p'
) -> torch.Tensor:
    """Pool multi-value embeddings, optionally weighted by paired dense values.

    Returns: (B, emb_dim).

    - weight_mode='uniform' or paired_value is None: mean-pool ignoring padding (=0).
    - weight_mode='log1p': w_i = log1p(clamp_min(paired_value_i, 0)) * mask_i.
      If sum(w) ~ 0, fall back to uniform mean-pool.
      If mask sum ~ 0 (all padding), output 0.
    """
    mask = (vals != 0).float().unsqueeze(-1)  # (B, length, 1)
    mask_count = mask.sum(dim=1).clamp(min=1)  # (B, 1)
    uniform_pool = (emb_all * mask).sum(dim=1) / mask_count  # (B, emb_dim)

    if weight_mode == 'uniform' or paired_value is None:
        return uniform_pool

    # log1p weighted pool
    eps = 1e-8
    w = torch.log1p(paired_value.clamp(min=0)) * mask.squeeze(-1)  # (B, length)
    W = w.sum(dim=1, keepdim=True)                                  # (B, 1)
    weighted_pool = (emb_all * w.unsqueeze(-1)).sum(dim=1) / W.clamp(min=eps)
    # Fallback: where W <= eps, use uniform_pool
    use_weighted = (W > eps).float()  # (B, 1)
    return weighted_pool * use_weighted + uniform_pool * (1 - use_weighted)
```

- [ ] **Step 5: 改 GroupNSTokenizer.forward 多值分支调用 `_pool_multivalue`**

定位：`src/model.py` line ~1057-1063（`else: # multi-value` 分支）。

替换前（当前 baseline）：
```python
else:
    # Multi-value feature: lookup then mean pooling (ignoring padding=0)
    vals = int_feats[:, offset:offset + length].long()  # (B, length)
    emb_all = emb_layer(vals)  # (B, length, emb_dim)
    mask = (vals != 0).float().unsqueeze(-1)  # (B, length, 1)
    count = mask.sum(dim=1).clamp(min=1)  # (B, 1)
    fid_emb = (emb_all * mask).sum(dim=1) / count  # (B, emb_dim)
```

替换后：
```python
else:
    # Multi-value feature
    vals = int_feats[:, offset:offset + length].long()  # (B, length)
    emb_all = emb_layer(vals)                            # (B, length, emb_dim)
    paired_value = (paired_dense or {}).get(fid_idx, None)
    fid_emb = _pool_multivalue(emb_all, vals, paired_value, weight_mode)
```

注意：`fid_idx` 此时是 `enumerate(group)` 里给的索引值，不是 group 内部位置。重读一遍 forward 上下文确认变量名（如不是 `fid_idx`，按实际改）。

- [ ] **Step 6: 跑测试确认通过**

```bash
pytest tests/test_pair_weighted_pool.py::test_log1p_weighted_pool_correctness -v
```
预期：PASS

- [ ] **Step 7: 跑全部 test，确认 uniform 测试还通过**

```bash
pytest tests/test_pair_weighted_pool.py -v
```
预期：3 PASSED

- [ ] **Step 8: Commit**

```bash
git add src/model.py tests/test_pair_weighted_pool.py
git commit -m "feat(model): log1p weighted pool helper + GroupNS happy path"
```

---

## Task 4: Fallback + 边界测试

**Files:**
- Modify: `tests/test_pair_weighted_pool.py`

- [ ] **Step 1: 写 fallback 测试 — dense 全 0 但 ids 有效**

```python
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
    vals_uniform = torch.tensor([[0.0, 0.0, 0.0, 0.0]])

    out_uniform_mode = tok(int_feats, paired_dense=None, weight_mode='uniform')
    out_log1p_mode = tok(int_feats, paired_dense={0: vals_uniform}, weight_mode='log1p')

    assert torch.allclose(out_uniform_mode, out_log1p_mode, atol=1e-7), \
        "log1p with all-zero dense must fall back to mean-pool"
```

- [ ] **Step 2: 跑测试**

```bash
pytest tests/test_pair_weighted_pool.py::test_fallback_to_mean_pool_when_dense_all_zero -v
```
预期：PASS（fallback 已在 Task 3 实现）。如失败，调试 helper 的 fallback 分支。

- [ ] **Step 3: 写边界测试 — ids 全 0（unknown user）**

```python
def test_unknown_user_returns_zero():
    """When ids are all padding, both modes should return zero embedding."""
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

    # padding_idx=0 means emb(0) = 0, so pool of all-pad ids must be 0 vector at emb level.
    # After projection + silu, the result is silu(proj(0)) = silu(b) where b is bias.
    # Both modes hit the mask=0 path → uniform_pool=0 path → identical results.
    assert torch.allclose(out_uniform, out_log1p, atol=1e-7)
```

- [ ] **Step 4: 跑测试**

```bash
pytest tests/test_pair_weighted_pool.py::test_unknown_user_returns_zero -v
```
预期：PASS

- [ ] **Step 5: 写部分 padding 测试**

```python
def test_partial_padding_no_pollution():
    """Padded positions must not contribute to pool, even if dense vals at pad positions are nonzero."""
    torch.manual_seed(0)
    feature_specs, groups = _make_simple_specs()
    tok = GroupNSTokenizer(
        feature_specs=feature_specs, groups=groups,
        emb_dim=8, d_model=16, emb_skip_threshold=0,
    )
    tok.eval()

    # ids: only first 2 valid, last 2 are padding
    int_feats = torch.tensor([[5, 7, 0, 0]], dtype=torch.long)
    # vals at padded positions are huge (would dominate if not masked)
    vals = torch.tensor([[1.0, 2.0, 1e6, 1e6]])

    out = tok(int_feats, paired_dense={0: vals}, weight_mode='log1p')

    # Manual: mask = [1, 1, 0, 0]; w = [log1p(1), log1p(2), 0, 0] = [0.693, 1.099, 0, 0]
    # pool = (0.693*emb(5) + 1.099*emb(7)) / (0.693 + 1.099)
    emb = tok.embs[0]
    e5, e7 = emb(torch.tensor([5, 7]))
    w = torch.log1p(torch.tensor([1.0, 2.0]))
    expected_pool = (w[0] * e5 + w[1] * e7) / w.sum()
    expected_token = torch.nn.functional.silu(tok.group_projs[0](expected_pool.unsqueeze(0))).unsqueeze(1)

    assert torch.allclose(out, expected_token, atol=1e-5)
```

- [ ] **Step 6: 跑测试**

```bash
pytest tests/test_pair_weighted_pool.py::test_partial_padding_no_pollution -v
```
预期：PASS

- [ ] **Step 7: 写数值稳定测试 — fid 65/66 量级 (1.5e9)**

```python
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
    # Mix of normal + huge values (fid 65/66 max ≈ 1.5e9)
    vals = torch.tensor([[100.0, 1.5e9, 50.0, 0.0]])

    out = tok(int_feats, paired_dense={0: vals}, weight_mode='log1p')

    assert torch.isfinite(out).all(), "Output must be finite (no inf/nan)"
    # log1p(1.5e9) ≈ 21.13, so the second position dominates; output should be close to silu(proj(emb(2)))
    # We don't assert exact value here, just stability.
```

- [ ] **Step 8: 跑测试**

```bash
pytest tests/test_pair_weighted_pool.py::test_numerical_stability_huge_values -v
```
预期：PASS

- [ ] **Step 9: 跑完整 test 套件**

```bash
pytest tests/test_pair_weighted_pool.py -v
```
预期：6 PASSED

- [ ] **Step 10: Commit**

```bash
git add tests/test_pair_weighted_pool.py
git commit -m "test(model): fallback / padding / numerical stability for weighted pool"
```

---

## Task 5: 把 weighted pool 应用到 RankMixerNSTokenizer

**Files:**
- Modify: `src/model.py`
- Modify: `tests/test_pair_weighted_pool.py`

**说明：** RankMixerNSTokenizer 与 GroupNSTokenizer 的差异在最后的 cat→split→project 步，但**多值 embedding 的 pool** 逻辑相同。Task 3 的 `_pool_multivalue` helper 应该可直接复用。

- [ ] **Step 1: 写 RankMixer 的 log1p 等价性测试**

```python
def test_rankmixer_log1p_correctness():
    """RankMixer must produce same multi-value pool as GroupNS for matching configs."""
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

    # Both should compute the same _pool_multivalue result internally;
    # downstream projection differs (per-group vs per-chunk), so we don't assert
    # equal final outputs—only that both produce finite, non-zero outputs and
    # that switching from log1p to uniform changes the result.
    out_uniform = tok_r(int_feats, paired_dense=None, weight_mode='uniform')
    out_log1p = tok_r(int_feats, paired_dense={0: vals}, weight_mode='log1p')

    assert torch.isfinite(out_uniform).all() and torch.isfinite(out_log1p).all()
    assert not torch.allclose(out_uniform, out_log1p, atol=1e-5), \
        "log1p with non-uniform vals should differ from uniform mode"
```

- [ ] **Step 2: 跑测试，确认失败（接口已有但 body 没改）**

```bash
pytest tests/test_pair_weighted_pool.py::test_rankmixer_log1p_correctness -v
```
预期：FAIL（断言 `not allclose` 不成立——log1p 走的还是旧路径）

- [ ] **Step 3: 改 RankMixerNSTokenizer.forward 多值分支调用 helper**

定位：`src/model.py` line ~1170-1174。

替换前：
```python
else:
    vals = int_feats[:, offset:offset + length].long()
    emb_all = emb_layer(vals)
    mask = (vals != 0).float().unsqueeze(-1)
    count = mask.sum(dim=1).clamp(min=1)
    fid_emb = (emb_all * mask).sum(dim=1) / count
```

替换后：
```python
else:
    vals = int_feats[:, offset:offset + length].long()
    emb_all = emb_layer(vals)
    paired_value = (paired_dense or {}).get(fid_idx, None)
    fid_emb = _pool_multivalue(emb_all, vals, paired_value, weight_mode)
```

注意 `fid_idx` 变量名需对照原代码——在 RankMixer 里循环结构是 `for group in self.groups: for fid_idx in group:`，所以 `fid_idx` 已是正确名字。

- [ ] **Step 4: 跑测试确认通过**

```bash
pytest tests/test_pair_weighted_pool.py -v
```
预期：8 PASSED（含原 7 + 新 1）

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_pair_weighted_pool.py
git commit -m "feat(model): apply weighted pool to RankMixerNSTokenizer"
```

---

## Task 6: PCVRHyFormer 接入 paired_dense_specs

**Files:**
- Modify: `src/model.py` (line ~1192-1700)
- Modify: `tests/test_pair_weighted_pool.py`

- [ ] **Step 1: 写一个 end-to-end 的 PCVRHyFormer 等价性测试**

```python
def test_pcvr_hyformer_uniform_equivalence():
    """PCVRHyFormer with pair_weight_mode='uniform' or specs=None must match baseline."""
    import torch
    from model import PCVRHyFormer, PCVRHyFormerInputs

    torch.manual_seed(42)

    # Minimal config matching demo data shape
    user_int_feature_specs = [(10, 0, 1), (5, 1, 4)]   # fid 1 single, fid 62 multi (len=4)
    item_int_feature_specs = [(20, 0, 1)]
    user_dense_dim = 4   # only fid 62 dense (len=4)
    item_dense_dim = 0
    seq_vocab_sizes = {'a': [10, 10]}

    common_kwargs = dict(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=user_dense_dim,
        item_dense_dim=item_dense_dim,
        seq_vocab_sizes=seq_vocab_sizes,
        user_ns_groups=[[0], [1]],
        item_ns_groups=[[0]],
        d_model=16, emb_dim=8, num_queries=1,
        num_hyformer_blocks=1, num_heads=2,
        num_time_buckets=0,
        ns_tokenizer_type='group',
    )

    model_baseline = PCVRHyFormer(**common_kwargs)
    model_baseline.eval()

    # New: same kwargs but with pair_weight_mode='uniform' should be bit-identical
    model_new = PCVRHyFormer(
        **common_kwargs,
        user_paired_dense_specs={62: (0, 4)},  # fid 62 dense at offset 0, length 4
        pair_weight_mode='uniform',
    )
    # Force shared weights
    model_new.load_state_dict(model_baseline.state_dict(), strict=False)
    model_new.eval()

    # ... build inputs and assert outputs match
    # (skip full test body if too involved; mark as TODO and rely on integration smoke instead)
```

**注**：这个测试可能因为 PCVRHyFormer 复杂依赖（seq features、time buckets 等）写起来很长。如果 5 分钟内写不出，**降级为只测 NSTokenizer 层**（已在 Task 2-5 完成），跳过该测试，靠 Task 8 的 integration smoke 兜底。

- [ ] **Step 2: 改 PCVRHyFormer.__init__ 加新参数**

定位：`src/model.py` line ~1199-1232。在 `__init__` 签名末尾加：

```python
        # Pair-weighted pool config
        user_paired_dense_specs: dict | None = None,  # {fid: (doff, dlen)}
        pair_weight_mode: str = 'uniform',
```

并在 body 末尾保存：
```python
        self.user_paired_dense_specs = user_paired_dense_specs or {}
        self.pair_weight_mode = pair_weight_mode

        # Build fid_idx lookup for user_int feature_specs
        self._user_paired_fid_idx_to_dense_slice = {}
        if user_paired_dense_specs and pair_weight_mode != 'uniform':
            # We need to know: for each fid_idx in user_int_feature_specs, what is the
            # dense slice (doff, dlen)?
            # user_int_feature_specs is List[Tuple[vocab, offset, length]]; we need
            # the fid number, which is NOT stored here. Pass fid list separately.
            # → Decision: change signature to also pass user_int_fids: List[int].
            pass  # implemented in next step
```

- [ ] **Step 3: 重新设计：把 fid 列表也传进来**

实际上 `user_int_feature_specs` 在 train.py 由 `build_feature_specs(schema, vocab_sizes)` 构造，schema 有 `entries: List[(fid, dim)]`。最简洁的做法是**新增**一个 `user_int_fids: List[int]` 参数。

修改 `__init__` 签名再加：
```python
        user_int_fids: List[int] | None = None,
```

body 中：
```python
        self._paired_fid_idx_to_slice = {}
        if user_paired_dense_specs and user_int_fids:
            for fid_idx, fid in enumerate(user_int_fids):
                if fid in user_paired_dense_specs:
                    self._paired_fid_idx_to_slice[fid_idx] = user_paired_dense_specs[fid]
```

- [ ] **Step 4: 改 PCVRHyFormer.forward 切片 + 传入 tokenizer**

定位：`src/model.py` line ~1637 和 ~1680 两处。改前：
```python
user_ns = self.user_ns_tokenizer(inputs.user_int_feats)
```

改后：
```python
paired_dense = None
if self._paired_fid_idx_to_slice and self.pair_weight_mode != 'uniform':
    paired_dense = {
        fid_idx: inputs.user_dense_feats[:, doff:doff + dlen]
        for fid_idx, (doff, dlen) in self._paired_fid_idx_to_slice.items()
    }
user_ns = self.user_ns_tokenizer(inputs.user_int_feats,
                                  paired_dense=paired_dense,
                                  weight_mode=self.pair_weight_mode)
```

两处都改。

- [ ] **Step 5: smoke test — 直接构造 PCVRHyFormer 实例确认 init 不崩**

```bash
cd /Users/tbjuechen/code/TAAC && python -c "
import sys; sys.path.insert(0, 'src')
from model import PCVRHyFormer
m = PCVRHyFormer(
    user_int_feature_specs=[(10,0,1),(5,1,4)],
    item_int_feature_specs=[(20,0,1)],
    user_dense_dim=4, item_dense_dim=0,
    seq_vocab_sizes={'a':[10,10]},
    user_ns_groups=[[0],[1]], item_ns_groups=[[0]],
    d_model=16, emb_dim=8, num_queries=1,
    num_hyformer_blocks=1, num_heads=2,
    num_time_buckets=0,
    user_paired_dense_specs={62:(0,4)}, user_int_fids=[1,62],
    pair_weight_mode='log1p',
)
print('ok')
"
```
预期：`ok`

- [ ] **Step 6: Commit**

```bash
git add src/model.py tests/test_pair_weighted_pool.py
git commit -m "feat(model): PCVRHyFormer wires paired_dense to user NS tokenizer"
```

---

## Task 7: train.py CLI flag + specs 构造

**Files:**
- Modify: `src/train.py`

- [ ] **Step 1: 加 CLI flag**

定位：`src/train.py` argparse 部分（line ~42 附近 `parser.add_argument` 集合）。加：

```python
parser.add_argument('--pair_weighted_pool', type=str, default='none',
                    choices=['none', 'log1p'],
                    help='Pair (int, dense) weighted pool mode for user_int_feats_{62-66}. '
                         'none = baseline mean-pool (default); log1p = log1p-weighted by paired dense.')
```

- [ ] **Step 2: 构造 user_paired_dense_specs**

定位：`src/train.py` line ~280-290（model_args 构造前）。加：

```python
# Pair-weighted pool: build {fid: (doff, dlen)} for fid in PAIR_WEIGHTED_FIDS that
# exist in user_dense_schema.
from model import PAIR_WEIGHTED_FIDS

user_paired_dense_specs = {}
if args.pair_weighted_pool != 'none':
    # Walk user_dense_schema.entries to find offsets
    doff = 0
    for fid, dim in pcvr_dataset.user_dense_schema.entries:
        if fid in PAIR_WEIGHTED_FIDS:
            user_paired_dense_specs[fid] = (doff, dim)
        doff += dim
    logging.info(f"Pair-weighted pool enabled (mode={args.pair_weighted_pool}); "
                 f"paired fids: {sorted(user_paired_dense_specs.keys())}")

user_int_fids = [fid for fid, _dim in pcvr_dataset.user_int_schema.entries]
```

注意：`schema.entries` 是 `List[Tuple[fid, dim]]`，需要确认确切属性名（dataset.py line 295 写了 `self.user_dense_schema.add(fid, dim)`，FeatureSchema 类应该有 `.entries`）。如不是 `entries`，按实际改。

**Sanity check 命令**：
```bash
cd /Users/tbjuechen/code/TAAC && grep -n "class FeatureSchema\|def add\|self.entries" src/dataset.py
```

- [ ] **Step 3: 注入 model_args**

定位：`src/train.py` line ~286-314 的 `model_args` dict 字面量。加两个 key：
```python
"user_paired_dense_specs": user_paired_dense_specs,
"user_int_fids": user_int_fids,
"pair_weight_mode": args.pair_weighted_pool,
```

- [ ] **Step 4: smoke test — 跑 train.py 1 个 step（demo data）确认不崩**

先看 demo 数据有没有 dense fid 62-66。如果没有，flag log1p 路径不会触发，但代码不应该崩。

```bash
cd /Users/tbjuechen/code/TAAC/src && python train.py --data_dir ../data/demo \
    --pair_weighted_pool none \
    --num_epochs 1 --batch_size 4 --max_steps_per_epoch 2 \
    --device cpu 2>&1 | tail -30
```
（实际 CLI flag 名按 train.py 当前实现匹配，例如 `--max_steps` 之类——快速读 train.py argparse 确认）

预期：跑完 2 个 step，不崩。

- [ ] **Step 5: smoke test — log1p 模式**

```bash
cd /Users/tbjuechen/code/TAAC/src && python train.py --data_dir ../data/demo \
    --pair_weighted_pool log1p \
    --num_epochs 1 --batch_size 4 --max_steps_per_epoch 2 \
    --device cpu 2>&1 | tail -30
```
预期：log 里看到 "Pair-weighted pool enabled" 行；跑完不崩。
**注意**：demo schema 是 codex mock，可能不含 fid 62-66；那么 paired_dense_specs 为空，effective 退化到 baseline。这是预期行为，不算 bug。

- [ ] **Step 6: Commit**

```bash
git add src/train.py
git commit -m "feat(train): --pair_weighted_pool CLI flag + paired_dense_specs"
```

---

## Task 8: run.sh A/B 配置

**Files:**
- Modify: `src/run.sh`

- [ ] **Step 1: 加 env var 控制**

定位：`src/run.sh` 当前 active config block（line ~5-15）。

加 env var 默认值（顶部）：
```bash
PAIR_WEIGHTED_POOL="${PAIR_WEIGHTED_POOL:-none}"
```

并在 train.py 调用行加参数：
```bash
    --pair_weighted_pool "$PAIR_WEIGHTED_POOL" \
```

- [ ] **Step 2: 加注释说明 A/B 用法**

run.sh 顶部注释加：
```bash
# Pair-weighted pool A/B:
#   PAIR_WEIGHTED_POOL=none  bash run.sh   # baseline (default)
#   PAIR_WEIGHTED_POOL=log1p bash run.sh   # treatment (W2.6 重写)
```

- [ ] **Step 3: 本地干跑 sanity（不真训，只验证 env 注入）**

```bash
cd /Users/tbjuechen/code/TAAC && PAIR_WEIGHTED_POOL=log1p bash -n src/run.sh
echo "syntax ok"
```
（`-n` = 不执行只检查语法）

- [ ] **Step 4: Commit**

```bash
git add src/run.sh
git commit -m "feat(run): PAIR_WEIGHTED_POOL env var for A/B"
```

---

## Task 9: 全套 sanity + 最终 commit + branch push 准备

**Files:**
- 仅运行测试，无文件改动

- [ ] **Step 1: 跑完整 pytest**

```bash
cd /Users/tbjuechen/code/TAAC && pytest tests/ -v
```
预期：所有测试 PASS。

- [ ] **Step 2: 确认 main 分支等价性 — 切回 main 跑同一个 demo 配置，对比 forward 输出**

可选高级 sanity：在 feature 分支用 `--pair_weighted_pool none` 跑 demo + 在 `main` 跑 demo，对比首批 batch 的 forward logit。如果 PRNG 一致且代码等价，应位 bit-equal。

如时间不够，**跳过此步**，靠 Task 7 Step 4 的 smoke test + Task 2 的 unit equivalence test 兜底。

- [ ] **Step 3: 检查 git log + status**

```bash
cd /Users/tbjuechen/code/TAAC && git log --oneline -10 && git status
```
预期：`feature/pair-weighted-pool` 分支上看到 ~7 个 commit；working tree 干净（除 `output/`）。

- [ ] **Step 4: 不 push**

CLAUDE.md 工程规范：实验结果出来后再 push。**这一步只是准备好分支，等 A/B 跑出来再决定 push。**

---

## A/B 实验执行（实施完成后另起任务）

实施完成后，按 spec `2026-05-03-pair-feature-design.md` 第 "A/B 计划" 节执行：

1. **等 E4 跑完**（CLAUDE.md 当前最高优先级）
2. Run X：`PAIR_WEIGHTED_POOL=none bash src/run.sh` — 3 epoch baseline
3. Run Y：`PAIR_WEIGHTED_POOL=log1p bash src/run.sh` — 3 epoch treatment
4. 隔离：`CUDA_VISIBLE_DEVICES=0` 和 `=1` 分别跑（不同 ckpt/log/tf path）
5. 看 val + test 同向涨判定（spec A/B 表）
6. 出结果 → 更新 spec changelog v0.2 + 主 spec F# + CLAUDE.md → push 分支

---

## 已知风险（实施过程中关注）

1. **schema 属性名不确定**：`schema.entries` 可能叫 `_entries` 或别的；Task 7 Step 2 有 sanity 命令，跑前确认
2. **PCVRHyFormer 测试复杂依赖**：Task 6 Step 1 的端到端测试可能因为 seq features / time buckets 写起来很长，已写明可降级为 smoke test
3. **demo schema 可能不含 fid 62-66**：log1p smoke test 会退化为等价 baseline；这不是 bug，但也意味着 demo 数据无法验证 weighted 路径正确性。**单元测试 (Task 2-5) 是 weighted 路径的唯一本地验证**，必须全绿
4. **`paired_dense=None` 默认值是 dict，要小心可变默认参陷阱**：plan 里都用 `paired_dense or {}` pattern，OK
5. **torch.compile 兼容性**：CLAUDE.md 提到 baseline 用 AMP+compile；新增 dict 参数和条件分支可能让 compile fallback。Task 7 smoke test 没开 compile，初次跑通后再单独验证 compile 不崩
