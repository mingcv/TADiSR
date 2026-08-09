#!/usr/bin/env python
"""Pre-compute the fixed quality-prompt embedding used by TADiSR CogView4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Local zai-org/CogView4-6B directory.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--prompt",
        default=(
            "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, "
            "hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, "
            "extreme meticulous detailing, skin pore detailing, hyper sharpness, "
            "perfect without deformations,text"
        ),
    )
    parser.add_argument(
        "--negative-prompt",
        default="blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth",
    )
    args = parser.parse_args()

    from tadisr.pipelines import CogView4TextEncoderPipeline

    encoder = CogView4TextEncoderPipeline(ckpt_dir=args.base_model, device=args.device)
    encoder.prepare_text_embed(args.prompt, args.negative_prompt, args.output)


if __name__ == "__main__":
    main()
