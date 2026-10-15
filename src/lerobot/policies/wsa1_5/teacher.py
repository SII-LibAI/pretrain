"""Official Omega preprocessing and a lazily loaded, frozen server-side teacher."""
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .initialization import load_report, read_checkpoint
from .vendor.omega_preprocessing import (
    _crop_to_supported_aspect_ratio, _max_size_target_shape, _pad_images_to_common_size,
)

OMEGA_PREPROCESS_VERSION = "omega.48b23c8.max_size.256.patch16.v1"


@dataclass(frozen=True)
class ViewGeometry:
    height: int
    width: int
    pad_top: int
    pad_left: int
    canvas_height: int
    canvas_width: int


def preprocess_omega_views(views):
    """[0,1] decoded native RGB -> exact official PIL crop/resize/white padding.

    The official loader accepts filenames. This entry point converts decoded
    uint8-equivalent views to PIL, then calls the same geometry helper functions.
    Preprocessing may run without constructing any model.
    """
    if not views:
        raise ValueError("Omega requires at least one valid view")
    tensors, sizes = [], []
    for view in views:
        if view.ndim != 3 or view.shape[0] != 3 or not torch.isfinite(view).all():
            raise ValueError("Teacher views must be finite native [3,H,W] RGB tensors")
        if view.min() < 0 or view.max() > 1:
            raise ValueError("Teacher preprocessing requires raw RGB in [0,1]")
        pixels = (view.detach().cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        image = _crop_to_supported_aspect_ratio(Image.fromarray(pixels))
        width, height = image.size
        target_h, target_w = _max_size_target_shape(height / width, 256, 16)
        image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        tensors.append(torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255)
        sizes.append((target_h, target_w))
    shapes = set(sizes)
    if len(shapes) > 1:
        tensors = _pad_images_to_common_size(tensors, shapes)
    common_h, common_w = tensors[0].shape[-2:]
    geometry = [ViewGeometry(h, w, (common_h - h) // 2, (common_w - w) // 2, common_h, common_w)
                for h, w in sizes]
    return torch.stack(tensors), geometry


class FeatureAligner3D:
    """Copy camera/register tokens; sample patches on a 16x16 content grid."""
    def __call__(self, features, geometry, patch_token_start=17):
        if patch_token_start != 17 or features.ndim != 4 or features.shape[1] != len(geometry):
            raise ValueError("Expected Omega [B,V,N,D] with camera + 16 register tokens")
        aligned, masks = [], []
        for view_index, geo in enumerate(geometry):
            grid_h, grid_w = geo.canvas_height // 16, geo.canvas_width // 16
            if features.shape[2] != 17 + grid_h * grid_w:
                raise ValueError("Omega feature length disagrees with actual preprocessed patch grid")
            special = features[:, view_index, :17]
            patches = features[:, view_index, 17:].transpose(1, 2).reshape(features.shape[0], -1, grid_h, grid_w)
            centers = (torch.arange(16, device=features.device, dtype=torch.float32) + 0.5) / 16
            yy = 2 * (geo.pad_top + centers * geo.height) / geo.canvas_height - 1
            xx = 2 * (geo.pad_left + centers * geo.width) / geo.canvas_width - 1
            gy, gx = torch.meshgrid(yy, xx, indexing="ij")
            grid = torch.stack([gx, gy], -1)[None].expand(features.shape[0], -1, -1, -1)
            aligned_patches = F.grid_sample(patches.float(), grid, mode="bilinear", padding_mode="border",
                                           align_corners=False).flatten(2).transpose(1, 2).to(features.dtype)
            pixel_valid = features.new_zeros(1, 1, geo.canvas_height, geo.canvas_width, dtype=torch.float32)
            pixel_valid[:, :, geo.pad_top:geo.pad_top + geo.height, geo.pad_left:geo.pad_left + geo.width] = 1
            patch_valid = F.avg_pool2d(pixel_valid, 16, 16)
            sampled_valid = F.grid_sample(patch_valid.expand(features.shape[0], -1, -1, -1), grid,
                                          mode="bilinear", padding_mode="border", align_corners=False)
            valid = torch.cat([torch.ones(features.shape[0], 17, dtype=torch.bool, device=features.device),
                               sampled_valid.flatten(1) >= 1 - 1e-6], 1)
            aligned.append(torch.cat([special, aligned_patches], 1))
            masks.append(valid)
        return torch.stack(aligned, 1), torch.stack(masks, 1)


class OmegaTeacher:
    """Not an nn.Module child: teacher weights are excluded from policy checkpoints.

    Lazy construction occurs only during training target extraction. Inference
    never imports or initializes VGGTOmega. Calling train() on the student cannot
    enable teacher dropout or gradients.
    """
    def __init__(self, checkpoint_path, layers=(4, 11, 17, 23), use_checkpoint=False, dtype=torch.bfloat16):
        self.checkpoint_path, self.layers = checkpoint_path, tuple(layers)
        self.use_checkpoint = use_checkpoint
        if dtype not in {torch.float32, torch.bfloat16, torch.float16}:
            raise ValueError("Unsupported Omega compute dtype")
        self.dtype = dtype
        self.model = None
        self.aligner = FeatureAligner3D()

    def _load(self, device):
        if self.model is None:
            if not self.checkpoint_path:
                raise ValueError("Uncached 3D targets require an Omega checkpoint")
            from vggt_omega.models import VGGTOmega
            # Lazy teacher initialization must not consume the student's RNG:
            # whether endpoint labels exist cannot alter its noise trajectory.
            rng_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=rng_devices):
                model = VGGTOmega(enable_alignment=True, use_checkpoint=self.use_checkpoint, autocast=False)
            load_report(model, read_checkpoint(self.checkpoint_path), name="VGGT-Omega", strict=True)
            if model.aggregator.patch_token_start != 17 or not set(self.layers).issubset(model.aggregator.cached_layer_indices):
                raise ValueError("Omega token order/cached layers differ from audited official source")
            self.model = model.eval().requires_grad_(False)
        self.model.to(device).eval()

    def features(self, views_by_sample, view_valid, device):
        if view_valid.dtype != torch.bool or view_valid.shape != (len(views_by_sample), 3):
            raise ValueError("Omega target view validity must be boolean [B,3]")
        results = {layer: [] for layer in self.layers}
        masks = []
        with torch.inference_mode():
            for views, valid in zip(views_by_sample, view_valid, strict=True):
                indices = [index for index, flag in enumerate(valid.tolist()) if flag]
                targets = {layer: torch.zeros(3, 273, 2048, device=device) for layer in self.layers}
                mask = torch.zeros(3, 273, dtype=torch.bool, device=device)
                if indices:
                    self._load(device)
                    # Invalid placeholder cameras must not affect valid cameras
                    # through teacher inter-frame attention.
                    images, geometry = preprocess_omega_views([views[index] for index in indices])
                    with torch.autocast(device_type=device.type, dtype=self.dtype,
                                        enabled=self.dtype != torch.float32 and device.type in {"cpu", "cuda"}):
                        hidden, start = self.model.aggregator(images[None].to(device))
                    for layer in self.layers:
                        feature = hidden[layer]
                        if feature is None or feature.shape[-1] != 2048:
                            raise ValueError(f"Missing/wrong Omega intermediate layer {layer}")
                        if not torch.isfinite(feature).all():
                            raise FloatingPointError(f"Non-finite Omega intermediate layer {layer}")
                        aligned, aligned_valid = self.aligner(feature, geometry, start)
                        targets[layer][indices] = aligned[0].float()
                        mask[indices] = aligned_valid[0]
                masks.append(mask)
                for layer in self.layers:
                    results[layer].append(targets[layer])
            stacked = {layer: torch.stack(values) for layer, values in results.items()}
            stacked_mask = torch.stack(masks)
        # Normal tensors can safely be saved by autograd's student loss.
        return {layer: value.clone() for layer, value in stacked.items()}, stacked_mask.clone()
