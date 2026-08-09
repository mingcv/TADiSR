#!/usr/bin/env python
"""Run TADiSR CogView4 inference on one image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Low-resolution RGB image.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", required=True, help="Local zai-org/CogView4-6B directory.")
    parser.add_argument("--prompt-embeddings", required=True,
                        help="Output from prepare_prompt_embeddings.py for the fixed quality prompt.")
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--tile", type=int, default=768)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.scale <= 0:
        parser.error("--scale must be positive")
    import torch
    import torch.nn.functional as F

    from tadisr.inference import CogView4Config, load_cogview4, load_rgb, save_mask, save_rgb
    from tadisr.tiling import tiled_predict

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_cogview4(CogView4Config(
        base_model=Path(args.base_model),
        checkpoint=Path(args.checkpoint),
        prompt_embeddings=Path(args.prompt_embeddings),
        device=args.device,
    ))
    image = load_rgb(args.input).to(args.device)
    target = (round(image.shape[-2] * args.scale), round(image.shape[-1] * args.scale))
    image = F.interpolate(image, size=target, mode="bicubic", align_corners=False)
    with torch.inference_mode():
        sr, mask_logits = tiled_predict(image, model, args.tile, args.overlap)
    save_rgb(sr, output_dir / "sr.png")
    save_mask(mask_logits, output_dir / "text_mask.png")


if __name__ == "__main__":
    main()
