"""Batch isolation checks with synthetic views, tiny random encoders and no pretrained assets."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.wsa1_5.experts import FrozenWanVAE
from lerobot.policies.wsa1_5.modeling_wsa1_5 import WSA1_5_Policy
from lerobot.policies.wsa1_5.teacher import OmegaTeacher
from lerobot.policies.wsa1_5.text_embeddings import encode_text
from lerobot.policies.wsa1_5.vendor.wan_video_text_encoder import WanTextEncoder
from lerobot.policies.wsa1_5.vendor.wan_video_vae import VideoVAE38_, WanVideoVAE
from test_wsa1_5_observation_encoding import make_policy


@pytest.mark.parametrize("all_valid", [True, False])
def test_omega_batches_all_samples_with_wsa_repeated_missing_cameras(monkeypatch, all_valid):
    valid = (torch.ones(8, 3, dtype=torch.bool) if all_valid else torch.tensor([
        [True, True, True], [True, False, True], [True, True, True], [False, False, False],
        [True, False, True], [False, True, False], [True, False, False],
    ]))
    views = [[torch.full((3, 16 + v * 4, 20), 0.1 + sample * 0.1 + v * 0.01)
              if valid[sample, v] else torch.full((3, 16, 16), float("nan"))
              for v in range(3)] for sample in range(len(valid))]
    teacher = OmegaTeacher("unused", layers=(4,), dtype=torch.float32)
    calls, loads = [], []

    def aggregator(images):
        assert images.shape[2:] == (3, 256, 256)
        calls.append(tuple(images.shape[:2]))
        # Depend on other views of the SAME sample to detect accidental batch/view flattening.
        base = images.mean((2, 3, 4)) + images.mean((1, 2, 3, 4))[:, None]
        feature = base[:, :, None, None].expand(-1, -1, 273, 2048)
        return {4: feature}, 17

    teacher.model = SimpleNamespace(aggregator=aggregator)
    monkeypatch.setattr(teacher, "_load", lambda device: loads.append(device))
    result, mask = teacher.features(views, valid, torch.device("cpu"))
    assert calls == ([(8, 3)] if all_valid else [(6, 3)])
    assert len(loads) == 1
    for sample, flags in enumerate(valid.tolist()):
        if not any(flags):
            continue
        primary = views[sample][flags.index(True)].mean()
        expected_views = torch.stack([views[sample][v].mean() if flag else primary for v, flag in enumerate(flags)])
        expected = expected_views + expected_views.mean()
        for view, flag in enumerate(flags):
            if flag:
                torch.testing.assert_close(result[4][sample, view], expected[view].expand(273, 2048))
    for sample in range(len(valid)):
        expected, expected_mask = teacher.features([views[sample]], valid[sample:sample + 1], torch.device("cpu"))
        torch.testing.assert_close(result[4][sample:sample + 1], expected[4])
        torch.testing.assert_close(mask[sample:sample + 1], expected_mask)
    assert not result[4][~mask].any() and not result[4].is_inference()


@pytest.mark.parametrize("frames", [1, 9])
def test_wan_vae_real_tiny_codec_batches_match_individual_encoding(frames):
    # Construct only a tiny random codec; never instantiate the pretrained 5B VAE.
    torch.manual_seed(18)
    wrapper = WanVideoVAE.__new__(WanVideoVAE)
    nn.Module.__init__(wrapper)
    wrapper.model = VideoVAE38_(dim=4, z_dim=2, dec_dim=4, num_res_blocks=1)
    wrapper.scale = [torch.tensor([0.2, -0.1]), torch.tensor([0.8, 1.1])]
    vae = FrozenWanVAE(wrapper)
    videos = torch.randn(3, 3, frames, 32, 32)
    calls = []
    hook = wrapper.model.encoder.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[0]))
    batch = vae.encode(videos)
    assert calls == [3] * ((frames - 1) // 4 + 1)
    assert all(value is None for value in wrapper.model._enc_feat_map)
    expected = torch.cat([vae.encode(video[None]) for video in videos])
    torch.testing.assert_close(batch, expected, atol=1e-5, rtol=1e-5)
    # Different batch/temporal lengths must never reuse the previous causal cache.
    changed = vae.encode(videos[1:2, :, :1])
    torch.testing.assert_close(changed, expected[1:2, :, :1], atol=1e-5, rtol=1e-5)
    hook.remove()


def test_wan_vae_keeps_structural_errors():
    wrapper = WanVideoVAE.__new__(WanVideoVAE)
    nn.Module.__init__(wrapper)
    with pytest.raises(ValueError, match="non-empty"):
        wrapper.encode(torch.zeros(0, 3, 1, 16, 16), "cpu")
    with pytest.raises(NotImplementedError, match="Tiled"):
        wrapper.encode(torch.zeros(1, 3, 1, 16, 16), "cpu", tiled=True)


def test_online_text_encoder_runs_once_for_the_whole_batch(monkeypatch):
    policy, _ = make_policy(monkeypatch)
    policy.config.max_text_length = 4
    policy.text_encoder = WanTextEncoder(vocab=16, dim=8, dim_attn=8, dim_ffn=16,
                                         num_heads=2, num_layers=2, num_buckets=8, dropout=0).eval().requires_grad_(False)
    texts = ["a", "b", "c"]
    ids = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0], [6, 7, 8, 9]])
    token_calls, encoder_calls = [], []

    def tokenizer(batch, **kwargs):
        token_calls.append(list(batch))
        tokens = torch.stack([ids[texts.index(text)] for text in batch])
        return tokens, tokens != 0

    policy.tokenizer = tokenizer
    monkeypatch.setattr(policy, "_encode_text", WSA1_5_Policy._encode_text.__get__(policy))
    hook = policy.text_encoder.register_forward_pre_hook(lambda module, args: encoder_calls.append(args[0].shape[0]))
    context, valid = policy._encode_text(texts)
    assert token_calls == [texts] and encoder_calls == [3]
    expected = [encode_text(policy.text_encoder, tokenizer, [text], "cpu", 4, 8) for text in texts]
    torch.testing.assert_close(context, torch.cat([value[0] for value in expected]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(valid, torch.cat([value[1] for value in expected]))
    assert not context[~valid].any() and not context.requires_grad
    hook.remove()


def test_motion_placeholder_reuses_real_encoding_and_invalidates_cache(monkeypatch):
    policy, vae = make_policy(monkeypatch)
    original_encode = vae.encode

    def nonzero_encoding(video):
        return original_encode(video) + 0.75

    monkeypatch.setattr(vae, "encode", nonzero_encoding)
    with torch.inference_mode():
        first = policy._placeholder(8, "cpu")
    second = policy._placeholder(2, torch.device("cpu"))
    assert vae.calls == [(1, 3, 1, 32, 32)]
    assert (first == 0.75).all() and (second == 0.75).all()
    assert not policy._motion_placeholder_latent.is_inference()
    assert not second.requires_grad and not any("placeholder" in key for key in policy.state_dict())
    policy.reset()
    policy._placeholder(1, "cpu")
    assert len(vae.calls) == 1
    state = policy.state_dict()
    policy.load_state_dict(state, strict=True)
    policy._placeholder(3, "cpu")
    assert len(vae.calls) == 2
    policy.to("cpu")
    policy._placeholder(4, "cpu")
    assert len(vae.calls) == 3
    policy.model_dtype = torch.float64
    assert policy._placeholder(2, "cpu").dtype == torch.float64
    assert len(vae.calls) == 4
    policy.config.motion_width = 64
    assert policy._placeholder(2, "cpu").shape == (2, 4, 1, 2, 4)
    assert vae.calls[-1] == (1, 3, 1, 32, 64)
