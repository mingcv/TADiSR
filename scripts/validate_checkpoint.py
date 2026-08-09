#!/usr/bin/env python
"""Validate a released TADiSR checkpoint without constructing the base model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", choices=("auto", "cogview4", "kolors"), default="auto")
    parser.add_argument(
        "--strict-decoder",
        action="store_true",
        help="Construct the joint decoder on CPU and require an exact state-dict match.",
    )
    args = parser.parse_args()

    from tadisr.checkpoint import inspect_checkpoint, strict_load_joint_decoder

    report = inspect_checkpoint(args.checkpoint, args.variant)
    if not report.valid:
        raise SystemExit(1)
    result = report.to_dict()
    if args.strict_decoder:
        strict_load_joint_decoder(args.checkpoint, args.variant)
        result["strict_decoder_load"] = True
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
