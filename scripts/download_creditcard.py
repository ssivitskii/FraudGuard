"""Download the ULB dataset mirror used by the official TensorFlow tutorial."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
EXPECTED_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"


def main() -> None:
    dest = Path("data/raw/creditcard.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        temporary = dest.with_suffix(".part")
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(dest)
    digest = file_sha256(dest)
    if digest != EXPECTED_SHA256:
        raise ValueError("ULB mirror checksum mismatch; refusing an unverified dataset")
    manifest = {
        "dataset": "ULB credit card transactions, September 2013",
        "url": URL,
        "tutorial": "https://www.tensorflow.org/tutorials/structured_data/imbalanced_data",
        "original": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        "sha256": digest,
        "bytes": dest.stat().st_size,
        "redistribution": "Raw data excluded from Git; consult original dataset terms.",
    }
    Path("data/creditcard_source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
