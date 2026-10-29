from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.wsa1_5.attention import WORLD_GROUPS as GROUPS, VISIBLE, build_causal_attention_mask, masked_attention
from lerobot.policies.wsa1_5.mot import CausalMoT, ExpertState


class SmallBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.num_heads = 2
        self.self_attn.attn_head_dim = 4
        for name in ["q", "k", "v"]:
            setattr(self.self_attn, name, nn.Linear(width, 8))
        self.self_attn.o = nn.Linear(8, width)
        self.self_attn.norm_q = nn.Identity()
        self.self_attn.norm_k = nn.Identity()
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.modulation = nn.Parameter(torch.randn(1, 6, width) * 0.2)
        self.ffn = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))


def fixture():
    torch.manual_seed(7)
    widths = dict(zip(GROUPS, [8, 12, 16, 10]))
    states = {g: ExpertState(torch.randn(2, 3, w, requires_grad=True), torch.zeros(2, 6, w),
                             torch.ones(2, 3, dtype=torch.bool)) for g, w in widths.items()}
    experts = {g: SimpleNamespace(blocks=nn.ModuleList([SmallBlock(w) for _ in range(3)]))
               for g, w in widths.items()}
    return states, experts


def test_dense_mask():
    lengths = {"O": 2, "R": 3, "M": 1, "G": 4}
    mask = build_causal_attention_mask(lengths)
    slices = {"O": slice(0, 2), "R": slice(2, 5), "M": slice(5, 6), "G": slice(6, 10)}
    for q in GROUPS:
        for k in GROUPS:
            assert bool(mask[slices[q], slices[k]].all()) == (k in VISIBLE[q])
    flags = {g: torch.ones(2, n, dtype=torch.bool) for g, n in lengths.items()}
    flags["R"][1, 0] = False
    padded = build_causal_attention_mask(lengths, valid_tokens=flags)
    assert not padded[1, :, 2].any() and not padded[1, :, :, 2].any()
    assert build_causal_attention_mask({"O": 1}).item()
    with pytest.raises(ValueError):
        build_causal_attention_mask({"X": 1})


@pytest.mark.parametrize("backend", ["sdpa", "math", "reference"])
def test_empty_rows_and_reference(backend):
    q = torch.randn(2, 2, 5, 4, requires_grad=True)
    k, v = torch.randn(2, 2, 3, 4), torch.randn(2, 2, 3, 4)
    mask = torch.ones(2, 1, 5, 3, dtype=torch.bool)
    mask[0, :, 0] = False
    actual = masked_attention(q, k, v, mask, backend=backend)
    expected = masked_attention(q, k, v, mask, backend="reference")
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert not actual[0, :, 0].any()
    actual.sum().backward()
    assert torch.isfinite(q.grad).all()


@pytest.mark.parametrize("checkpointing", [False, True])
def test_all_causal_dependencies_and_gradients(checkpointing):
    states, experts = fixture()
    mot = CausalMoT(gradient_checkpointing=checkpointing)
    original, _ = mot(states, experts)
    for source in GROUPS:
        changed = {g: ExpertState(s.tokens.clone(), s.modulation, s.valid) for g, s in states.items()}
        changed[source].tokens = changed[source].tokens + torch.randn_like(changed[source].tokens) * 10
        altered, _ = mot(changed, experts)
        for query in GROUPS:
            if source not in VISIBLE[query]:
                torch.testing.assert_close(original[query], altered[query], atol=0, rtol=0)
            else:
                assert not torch.allclose(original[query], altered[query])
    for query in GROUPS:
        grads = torch.autograd.grad(original[query].square().sum(), [states[g].tokens for g in GROUPS],
                                    retain_graph=True, allow_unused=True)
        for source, grad in zip(GROUPS, grads, strict=True):
            if source not in VISIBLE[query]:
                assert grad is None or not grad.any()
            else:
                assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_padding_never_visible():
    states, experts = fixture()
    states["O"].valid[:, -1] = False
    before, _ = CausalMoT()(states, experts)
    states["O"].tokens = states["O"].tokens.detach().clone()
    states["O"].tokens[:, -1] = 10000
    after, _ = CausalMoT()(states, experts)
    for group in GROUPS:
        torch.testing.assert_close(before[group], after[group], atol=0, rtol=0)


def test_checkpoint_gradients_match():
    states, experts = fixture()
    eager, _ = CausalMoT()(states, experts)
    checked, _ = CausalMoT(gradient_checkpointing=True)(states, experts)
    params = [p for expert in experts.values() for p in expert.blocks.parameters()]
    eager_grads = torch.autograd.grad(sum(v.square().sum() for v in eager.values()), params, retain_graph=True)
    checked_grads = torch.autograd.grad(sum(v.square().sum() for v in checked.values()), params)
    for a, b in zip(eager_grads, checked_grads, strict=True):
        torch.testing.assert_close(a, b)
