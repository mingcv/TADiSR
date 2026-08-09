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
    args = parser.parse_args()

    from tadisr.checkpoint import inspect_checkpoint

    report = inspect_checkpoint(args.checkpoint, args.variant)
    print(json.dumps(report.to_dict(), indent=2))
    if not report.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
