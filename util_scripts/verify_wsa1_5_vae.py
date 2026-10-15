"""SERVER ONLY: verify official Wan VAE temporal behavior and condition causality.

The local development agent must not execute this program with a checkpoint.
"""
import argparse
import json
from pathlib import Path

import torch

from lerobot.policies.wsa1_5.initialization import load_report, read_checkpoint
from lerobot.policies.wsa1_5.temporal import TemporalLayout
from lerobot.policies.wsa1_5.vendor.wan_video_vae import WanVideoVAE38


def verify(vae, device, height=192, width=160):
    layout = TemporalLayout()
    torch.manual_seed(42)
    future = torch.randn(1, 32, 3, height, width, device=device)
    motion = layout.prepend_motion_placeholder(future)
    observed = torch.randn(1, 3, height, width, device=device)
    rgb = layout.assemble_rgb(observed, future)
    with torch.inference_mode():
        direct = vae.encode(future.transpose(1, 2), device=device)
        direct_decoded = vae.model.decode(direct, vae.scale)
        padded = vae.encode(motion, device=device)
        decoded = vae.model.decode(padded, vae.scale)
        isolated = vae.encode(motion[:, :, :1], device=device)
        perturbed = motion.clone()
        perturbed[:, :, 1:] = torch.randn_like(perturbed[:, :, 1:]) * 2
        perturbed_z = vae.encode(perturbed, device=device)
        rgb_z = vae.encode(rgb, device=device)
        observed_z = vae.encode(rgb[:, :, :1], device=device)
    layout.check_latents(padded, height, width)
    assert direct.shape[2] == 8 and direct_decoded.shape[2] == 29, "Official 32-frame behavior changed"
    assert decoded.shape[2] == 33
    torch.testing.assert_close(padded[:, :, :1], isolated, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(padded[:, :, :1], perturbed_z[:, :, :1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rgb_z[:, :, :1], observed_z, atol=1e-5, rtol=1e-5)
    return {"direct_32_latents": list(direct.shape), "direct_32_decoded": list(direct_decoded.shape),
            "padded_33_latents": list(padded.shape), "padded_33_decoded": list(decoded.shape),
            "condition_future_perturbation_error": float((padded[:, :, :1] - perturbed_z[:, :, :1]).abs().max()),
            "condition_separate_encode_error": float((rgb_z[:, :, :1] - observed_z).abs().max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--execute-on-server", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("outputs/wsa1_5_server_vae.json"))
    args = parser.parse_args()
    if not args.execute_on_server:
        parser.error("Requires --execute-on-server; local development does not load or run models")
    vae = WanVideoVAE38()
    state = read_checkpoint(args.vae_path)
    if not all(key.startswith("model.") for key in state):
        state = {f"model.{key}": value for key, value in state.items()}
    load_report(vae, state, name="Wan2.2 VAE", strict=True)
    vae = vae.to(args.device).eval().requires_grad_(False)
    report = verify(vae, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
