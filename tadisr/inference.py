"""Public, device-configurable inference API for the CogView4 TADiSR release."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .checkpoint import inspect_checkpoint


@dataclass(frozen=True)
class CogView4Config:
    base_model: Path
    checkpoint: Path
    prompt_embeddings: Path
    device: str = "cuda"


def load_cogview4(config: CogView4Config):
    """Construct and validate the CogView4 TADiSR pipeline before moving it to GPU."""
    report = inspect_checkpoint(config.checkpoint, variant="cogview4")
    if not report.valid:
        raise RuntimeError("Invalid checkpoint: " + "; ".join(report.errors))
    from tadisr_pipelines import CogView4Pipeline

    pipeline = CogView4Pipeline(
        ckpt_dir=str(config.base_model),
        prompt_path=str(config.prompt_embeddings),
        device=config.device,
    )
    pipeline.load_model(str(config.checkpoint))
    pipeline.set_eval()
    return pipeline


def load_rgb(path: str | Path):
    import torch

    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def save_rgb(image, path: str | Path) -> None:
    array = image.squeeze(0).detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray((array * 255.0).round().astype(np.uint8)).save(path)


def save_mask(mask_logits, path: str | Path) -> None:
    mask = mask_logits.sigmoid().squeeze().detach().float().cpu().clamp(0, 1).numpy()
    Image.fromarray((mask * 255.0).round().astype(np.uint8), mode="L").save(path)
