#!/usr/bin/env python
"""Verify a local release checkpoint against checkpoints/manifest.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True, help="Model id from the manifest.")
    parser.add_argument("--manifest", default="checkpoints/manifest.json")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    record = next((item for item in manifest["models"] if item["id"] == args.model), None)
    if record is None:
        raise SystemExit(f"Unknown model id: {args.model}")
    path = Path(args.checkpoint)
    actual = sha256(path)
    if path.stat().st_size != record["bytes"] or actual != record["sha256"]:
        raise SystemExit(f"Checksum mismatch: {actual}")
    print(f"OK {record['id']} {actual}")


if __name__ == "__main__":
    main()
