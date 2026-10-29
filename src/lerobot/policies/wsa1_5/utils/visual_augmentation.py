"""LingBotVLA's training RGB augmentation, replayed across views and time."""

import math
from dataclasses import dataclass, fields
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
import yaml
from PIL import Image, ImageOps
from torchvision.transforms import (
    InterpolationMode,
    functional as TF,  # noqa: N812
)

CAMERA_WEIGHTS = {"camera_top": 0.50, "camera_wrist_left": 0.25, "camera_wrist_right": 0.25}
PROJECT_ROOT = Path(__file__).resolve().parents[5]


@lru_cache(maxsize=128)
def _load_crop(path):
    with Image.open(path) as source:
        return TF.pil_to_tensor(ImageOps.exif_transpose(source).convert("RGBA")).float() / 255


@lru_cache(maxsize=4)
def _crop_bank(directory):
    root = Path(directory).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    if not root.is_dir():
        raise FileNotFoundError(f"Object crop directory does not exist: {root}")
    # Only object subdirectories; exclude the root overview/contact sheet.
    paths = tuple(str(p.resolve()) for p in sorted(root.glob("*/*"))
                  if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if not paths:
        raise ValueError(f"No object crop images found under {root}/*/")
    for path in paths:
        _load_crop(path)
    return paths


@dataclass
class VisualAugmentationConfig:
    keep_original_prob: float = 0.10
    color_prob: float = 0.80
    brightness: tuple = (0.25, 1.10)
    contrast: tuple = (0.5, 1.5)
    saturation: tuple = (0.4, 1.6)
    temperature: tuple = (0.90, 1.10)
    affine_prob: float = 0.50
    scale: tuple = (0.90, 1.10)
    translate: float = 0.06
    gaussian_noise_prob: float = 0.50
    noise_std: tuple = (0.005, 0.03)
    salt_pepper_prob: float = 0.15
    salt_pepper_amount: tuple = (0.001, 0.01)
    gaussian_blur_prob: float = 0.25
    blur_sigma: tuple = (0.2, 1.2)
    motion_blur_prob: float = 0.10
    motion_kernel: tuple = (3, 5)
    jpeg_prob: float = 0.30
    jpeg_quality: tuple = (35, 90)
    occlusion_prob: float = 0.0
    occlusion_area: tuple = (0.02, 0.10)
    occlusion_aspect: tuple = (0.5, 2.0)
    paste_prob: float = 0.0
    paste_area: tuple = (0.01, 0.05)
    crops_dir: str = "assets/RoboTwin_Clean_Object_Crops"

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "crops_dir":
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("crops_dir must be a nonempty path string")
            elif item.name.endswith("_prob") or item.name == "translate":
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"{item.name} must be in [0, 1]")
            else:
                if len(value) != 2 or not all(math.isfinite(x) for x in value) or not 0 < value[0] <= value[1]:
                    raise ValueError(f"Invalid range for {item.name}: {value}")
        if self.gaussian_blur_prob + self.motion_blur_prob > 1:
            raise ValueError("Blur probabilities must sum to <= 1 (mutually exclusive)")
        if self.salt_pepper_amount[1] > 1:
            raise ValueError("salt_pepper_amount must be <= 1")
        if not 1 <= self.jpeg_quality[0] <= self.jpeg_quality[1] <= 100:
            raise ValueError("JPEG quality must be in [1, 100]")
        for name in ("jpeg_quality", "motion_kernel"):
            if any(not isinstance(x, int) for x in getattr(self, name)):
                raise ValueError(f"{name} must contain integers")
        if any(x % 2 == 0 for x in self.motion_kernel):
            raise ValueError("motion_kernel endpoints must be odd")
        if self.temperature[1] >= 2:
            raise ValueError("temperature must be < 2 for positive blue gain")
        if self.occlusion_area[1] > 1 or self.paste_area[1] > 1:
            raise ValueError("Overlay area fractions must be <= 1")


def load_visual_augmentation(path, *, image_augment=True, object_paste=True):
    if not image_augment and not object_paste:
        return None
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as stream:
        options = yaml.safe_load(stream)
    if not isinstance(options, dict):
        raise ValueError("Image augmentation YAML must contain a mapping")
    config = VisualAugmentationConfig(**options)
    if not image_augment:
        for item in fields(config):
            if item.name.endswith("_prob") and item.name not in {"keep_original_prob", "paste_prob"}:
                setattr(config, item.name, 0.0)
    if not object_paste:
        config.paste_prob = 0.0
    if config.paste_prob > 0:
        _crop_bank(config.crops_dir)
    return config


def sample_pixel_augmentation(config, view_shapes=None):
    def draw(bounds):
        return float(torch.empty(()).uniform_(*bounds))

    def chance(prob):
        return bool(torch.rand(()) < prob)

    params = {"pixel_config": config, "keep_original": chance(config.keep_original_prob)}
    if params["keep_original"]:
        return params
    blur = float(torch.rand(()))
    params.update(
        color=chance(config.color_prob),
        brightness=draw(config.brightness), contrast=draw(config.contrast),
        saturation=draw(config.saturation), temperature=draw(config.temperature),
        affine=chance(config.affine_prob), scale=draw(config.scale),
        translate=(draw((-config.translate, config.translate)), draw((-config.translate, config.translate))),
        gaussian_noise=chance(config.gaussian_noise_prob), noise_std=draw(config.noise_std),
        salt_pepper=chance(config.salt_pepper_prob), salt_pepper_amount=draw(config.salt_pepper_amount),
        blur="gaussian" if blur < config.gaussian_blur_prob else (
            "motion" if blur < config.gaussian_blur_prob + config.motion_blur_prob else None),
        blur_sigma=draw(config.blur_sigma),
        motion_kernel=int(torch.randint(config.motion_kernel[0] // 2, config.motion_kernel[1] // 2 + 1, ())) * 2 + 1,
        motion_vertical=chance(0.5), jpeg=chance(config.jpeg_prob),
        jpeg_quality=int(torch.randint(config.jpeg_quality[0], config.jpeg_quality[1] + 1, ())),
        seed=int(torch.randint(2**31, ())),
    )
    view_shapes = view_shapes if view_shapes is not None else dict.fromkeys(CAMERA_WEIGHTS, (256, 256))
    available = [(key, CAMERA_WEIGHTS[key.rsplit(".", 1)[-1]]) for key in view_shapes
                 if key.rsplit(".", 1)[-1] in CAMERA_WEIGHTS]

    def choose_view():
        index = int(torch.multinomial(torch.tensor([weight for _, weight in available]), 1))
        return available[index][0]

    def box(view, area_range, aspect):
        height, width = view_shapes[view]
        area = draw(area_range) * height * width
        box_width, box_height = math.sqrt(area * aspect), math.sqrt(area / aspect)
        fit = min(1.0, width / box_width, height / box_height)
        box_width, box_height = box_width * fit / width, box_height * fit / height
        return (draw((0, max(0, 1 - box_width))), draw((0, max(0, 1 - box_height))), box_width, box_height)

    params.update(occlusion_view=None, paste_view=None, pastes=[])
    if available and config.occlusion_prob > 0 and chance(config.occlusion_prob):
        view = choose_view()
        params.update(occlusion_view=view, occlusion_box=box(view, config.occlusion_area, draw(config.occlusion_aspect)),
                      occlusion_mode=("solid", "mosaic", "noise")[int(torch.randint(3, ()))])
    if available and config.paste_prob > 0 and chance(config.paste_prob):
        view = choose_view()
        paths = _crop_bank(config.crops_dir)
        count = int(torch.randint(1, min(2, len(paths)) + 1, ()))
        params["paste_view"] = view
        for index in torch.randperm(len(paths))[:count].tolist():
            path = paths[index]
            crop = _load_crop(path)
            params["pastes"].append({"path": path, "box": box(view, config.paste_area, crop.shape[2] / crop.shape[1])})
    return params


def _pixel_box(box, height, width):
    left, top, box_width, box_height = box
    w, h = max(1, min(width, round(box_width * width))), max(1, min(height, round(box_height * height)))
    x, y = min(width - w, max(0, round(left * width))), min(height - h, max(0, round(top * height)))
    return x, y, w, h


def apply_pixel_augmentation(visual, params, view_key=None):
    if params["keep_original"]:
        return visual
    if visual.ndim not in (3, 4) or visual.shape[-3] != 3:
        raise ValueError("Expected RGB (C,H,W) or (T,C,H,W)")
    if visual.dtype != torch.uint8 and not visual.is_floating_point():
        raise ValueError("Expected uint8 or floating point RGB")
    was_image = visual.ndim == 3
    image = visual.unsqueeze(0) if was_image else visual
    factor = 255.0 if visual.dtype == torch.uint8 or float(image.max()) > 1 else 1.0
    image = image.float() / factor
    height, width = image.shape[-2:]
    generator = torch.Generator(device=image.device).manual_seed(params["seed"])
    if params["color"]:
        image = TF.adjust_brightness(image, params["brightness"])
        image = TF.adjust_contrast(image, params["contrast"])
        image = TF.adjust_saturation(image, params["saturation"])
        temperature = params["temperature"]
        image = (image * image.new_tensor([temperature, 1.0, 2.0 - temperature])[None, :, None, None]).clamp(0, 1)
    if params["affine"]:
        tx, ty = params["translate"]
        image = TF.affine(image, angle=0, translate=[round(tx * width), round(ty * height)],
                          scale=params["scale"], shear=[0, 0], interpolation=InterpolationMode.BILINEAR, fill=0)
    overlay_generator = torch.Generator(device=image.device).manual_seed(params["seed"] + 1)
    if params.get("occlusion_view") is not None and params["occlusion_view"] == view_key:
        x, y, w, h = _pixel_box(params["occlusion_box"], height, width)
        region = image[..., y:y+h, x:x+w]
        if params["occlusion_mode"] == "mosaic":
            fill = F.interpolate(F.adaptive_avg_pool2d(region, (min(4, h), min(4, w))), (h, w), mode="nearest")
        else:
            size = (1, 3, 1, 1) if params["occlusion_mode"] == "solid" else (1, 3, h, w)
            fill = torch.rand(size, device=image.device, generator=overlay_generator)
        image[..., y:y+h, x:x+w] = fill
    if params.get("paste_view") is not None and params["paste_view"] == view_key:
        for paste in params["pastes"]:
            x, y, w, h = _pixel_box(paste["box"], height, width)
            crop = _load_crop(paste["path"]).unsqueeze(0).to(image.device)
            crop = F.interpolate(crop, (h, w), mode="bilinear", align_corners=False, antialias=True)
            alpha = crop[:, 3:4]
            region = image[..., y:y+h, x:x+w]
            image[..., y:y+h, x:x+w] = crop[:, :3] * alpha + region * (1 - alpha)
    if params["gaussian_noise"]:
        noise = torch.randn((1, 3, height, width), generator=generator, device=image.device)
        image = (image + noise * params["noise_std"]).clamp(0, 1)
    if params["salt_pepper"]:
        noise = torch.rand((1, 1, height, width), generator=generator, device=image.device)
        amount = params["salt_pepper_amount"]
        image = torch.where(noise < amount / 2, 0.0, torch.where(noise > 1 - amount / 2, 1.0, image))
    if params["blur"] == "gaussian":
        image = TF.gaussian_blur(image, [5, 5], [params["blur_sigma"]] * 2)
    elif params["blur"] == "motion":
        kernel = params["motion_kernel"]
        size = (kernel, 1) if params["motion_vertical"] else (1, kernel)
        pad = (0, 0, kernel // 2, kernel // 2) if params["motion_vertical"] else (kernel // 2, kernel // 2, 0, 0)
        image = F.avg_pool2d(F.pad(image, pad, mode="replicate"), size, stride=1)
    if params["jpeg"]:
        frames = []
        for frame in image:
            array = (frame.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
            with BytesIO() as buffer:
                Image.fromarray(array).save(buffer, format="JPEG", quality=params["jpeg_quality"])
                buffer.seek(0)
                with Image.open(buffer) as decoded:
                    frames.append(TF.pil_to_tensor(decoded.convert("RGB")))
        image = torch.stack(frames).to(device=image.device, dtype=torch.float32) / 255
    image = image.clamp(0, 1) * factor
    image = image.round().to(visual.dtype) if visual.dtype == torch.uint8 else image.to(visual.dtype)
    return image.squeeze(0) if was_image else image
