"""Seam-aware tiling used by the public inference entry point."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F


def _starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile <= overlap:
        raise ValueError("tile must be larger than overlap")
    if length <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _weight(tile: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    window = torch.hann_window(tile, periodic=False, device=device, dtype=dtype).clamp_min(1e-3)
    return (window[:, None] * window[None, :])[None, None]


def tiled_predict(
    image: torch.Tensor,
    predict: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    tile: int = 768,
    overlap: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict SR image and text-mask logits using overlapped, weighted tiles.

    The model operates on already-upsampled RGB input and must preserve its spatial
    resolution. Only batch size one is supported because the diffusion path is
    intentionally single-image.
    """
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("Expected a single RGB tensor with shape (1, 3, H, W).")
    _, _, height, width = image.shape
    y_starts, x_starts = _starts(height, tile, overlap), _starts(width, tile, overlap)
    pad_h, pad_w = max(tile - height, 0), max(tile - width, 0)
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
        y_starts, x_starts = [0], [0]

    image_sum = torch.zeros((1, 3, image.shape[-2], image.shape[-1]), device=image.device, dtype=image.dtype)
    mask_sum = torch.zeros((1, 1, image.shape[-2], image.shape[-1]), device=image.device, dtype=image.dtype)
    weights = torch.zeros((1, 1, image.shape[-2], image.shape[-1]), device=image.device, dtype=image.dtype)
    blend = _weight(tile, image.device, image.dtype)

    for top in y_starts:
        for left in x_starts:
            patch = image[:, :, top : top + tile, left : left + tile]
            sr_patch, mask_patch = predict(patch)
            if sr_patch.shape != patch.shape or mask_patch.shape != (1, 1, tile, tile):
                raise RuntimeError("Model output shape does not match tiled inference contract.")
            image_sum[:, :, top : top + tile, left : left + tile] += sr_patch * blend
            mask_sum[:, :, top : top + tile, left : left + tile] += mask_patch * blend
            weights[:, :, top : top + tile, left : left + tile] += blend

    sr = image_sum / weights
    mask_logits = mask_sum / weights
    return sr[:, :, :height, :width], mask_logits[:, :, :height, :width]
