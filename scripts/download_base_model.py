#!/usr/bin/env python
"""Download a public base model to a local directory."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("cogview4", "kolors"), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    repo_id = {"cogview4": "zai-org/CogView4-6B", "kolors": "Kwai-Kolors/Kolors"}[args.model]
    snapshot_download(repo_id=repo_id, local_dir=args.output_dir)


if __name__ == "__main__":
    main()
