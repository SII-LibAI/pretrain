"""Fixed-square Omega targets and a frozen server-side teacher."""
import logging

import torch
import torch.nn.functional as F

from .initialization import load_report, read_checkpoint


def preprocess_omega_views(views):
    """Resize raw [0,1] RGB directly to 256x256 with bilinear interpolation."""
    if not views:
        raise ValueError("Omega requires at least one valid view")
    tensors = []
    for view in views:
        if view.ndim != 3 or view.shape[0] != 3 or not torch.isfinite(view).all():
            raise ValueError("Teacher views must be finite native [3,H,W] RGB tensors")
        if view.min() < 0 or view.max() > 1:
            raise ValueError("Teacher preprocessing requires raw RGB in [0,1]")
        tensors.append(F.interpolate(view.detach().float()[None], size=(256, 256),
                                     mode="bilinear", align_corners=False)[0])
    return torch.stack(tensors)


class OmegaTeacher:
    """Not an nn.Module child: teacher weights are excluded from policy checkpoints.

    The training entry loads weights before optimization; standalone target
    extraction retains lazy loading. Inference never initializes VGGTOmega.
    Calling train() on the student cannot enable teacher dropout or gradients.
    """
    def __init__(self, checkpoint_path, layers=(4, 11, 17, 23), use_checkpoint=False, dtype=torch.bfloat16):
        self.checkpoint_path, self.layers = checkpoint_path, tuple(layers)
        self.use_checkpoint = use_checkpoint
        if dtype not in {torch.float32, torch.bfloat16, torch.float16}:
            raise ValueError("Unsupported Omega compute dtype")
        self.dtype = dtype
        self.model = None

    def _load(self, device):
        if self.model is None:
            if not self.checkpoint_path:
                raise ValueError("Enabled 3D supervision requires an Omega checkpoint")
            logging.info("[WSA teacher] loading VGGT-Omega from %s", self.checkpoint_path)
            from vggt_omega.models import VGGTOmega
            # Teacher initialization must not consume the student's RNG:
            # whether endpoint labels exist cannot alter its noise trajectory.
            rng_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=rng_devices):
                model = VGGTOmega(enable_alignment=True, use_checkpoint=self.use_checkpoint, autocast=False)
            load_report(model, read_checkpoint(self.checkpoint_path), name="VGGT-Omega", strict=True)
            if model.aggregator.patch_token_start != 17 or not set(self.layers).issubset(model.aggregator.cached_layer_indices):
                raise ValueError("Omega token order/cached layers differ from audited official source")
            self.model = model.to(device=device, dtype=self.dtype).eval().requires_grad_(False)
            logging.info("[WSA teacher] VGGT-Omega ready device=%s compute_dtype=%s frozen=True", device, self.dtype)
        else:
            self.model.to(device).eval()

    def features(self, views_by_sample, view_valid, device):
        if not views_by_sample or view_valid.dtype != torch.bool or view_valid.shape != (len(views_by_sample), 3):
            raise ValueError("Omega target view validity must be boolean [B,3]")
        valid_rows = view_valid.tolist()
        samples = [sample for sample, valid in enumerate(valid_rows) if any(valid)]
        with torch.inference_mode():
            results = {layer: torch.zeros(len(views_by_sample), 3, 273, 2048, device=device) for layer in self.layers}
            mask = view_valid.to(device)[..., None].expand(-1, -1, 273)
            if samples:
                self._load(device)
                images = []
                for sample in samples:
                    valid, views = valid_rows[sample], views_by_sample[sample]
                    primary = views[valid.index(True)]
                    # WSA-L repeats the first valid camera; the original mask still gates the loss.
                    images.append(preprocess_omega_views([views[index] if flag else primary
                                                          for index, flag in enumerate(valid)]))
                images = torch.stack(images).to(device)
                with torch.autocast(device_type=device.type, dtype=self.dtype,
                                    enabled=self.dtype != torch.float32 and device.type in {"cpu", "cuda"}):
                    hidden, start = self.model.aggregator(images)
                if start != 17:
                    raise ValueError("Expected Omega camera + 16 register tokens before patch tokens")
                for layer in self.layers:
                    feature = hidden[layer]
                    if feature is None or feature.shape != (len(samples), 3, 273, 2048):
                        raise ValueError(f"Missing/wrong Omega intermediate layer {layer}")
                    if not torch.isfinite(feature).all():
                        raise FloatingPointError(f"Non-finite Omega intermediate layer {layer}")
                    results[layer][samples] = feature.float().masked_fill(~mask[samples, ..., None], 0)
        # Normal tensors can safely be saved by autograd's student loss.
        return {layer: value.clone() for layer, value in results.items()}, mask.clone()
