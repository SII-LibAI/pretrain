"""Strict tensor contracts at data, cache and loss boundaries."""
import torch


def require_finite(value, name):
    if not torch.is_tensor(value) or not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} must be a finite tensor")


def require_bool(value, name):
    if not torch.is_tensor(value) or value.dtype != torch.bool:
        raise TypeError(f"{name} must be an explicit boolean tensor")
