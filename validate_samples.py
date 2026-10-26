"""Run pinned CPU inference on a directory of rendered PNG/JPEG samples.

Usage: python validate_samples.py SAMPLE_DIRECTORY --output results.json
This does not send images to Hugging Face; only model files are downloaded.
"""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from modality_check import REVISION, load_modality_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checker = load_modality_check()
    rows = []
    for path in sorted(args.directory.iterdir()):
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        with Image.open(path) as image:
            decision = checker.classify(image)
            row = dict(filename=path.name, size=list(image.size), mode=image.mode,
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       label=decision.label, granularity=decision.granularity,
                       score=decision.score, out_of_distribution=decision.out_of_distribution,
                       accepted=decision.accepted)
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(dict(revision=REVISION,
                          providers=checker.session.get_providers(), results=rows), indent=2),
                          encoding="utf-8")


if __name__ == "__main__":
    main()
