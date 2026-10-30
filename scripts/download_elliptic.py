"""Download the author's public Elliptic dataset without account credentials."""

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from scripts.download_creditcard import file_sha256

URL = "https://www.kaggle.com/api/v1/datasets/download/ellipticco/elliptic-data-set"
EXPECTED_SHA256 = {
    "elliptic_txs_features.csv": "fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0",
    "elliptic_txs_edgelist.csv": "a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e",
    "elliptic_txs_classes.csv": "93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493",
}


def download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for name, expected in EXPECTED_SHA256.items():
        path = data_dir / name
        if not path.exists():
            missing.append(name)
        elif file_sha256(path) != expected:
            raise ValueError(f"Checksum mismatch for {name}; existing file was not overwritten")
    if missing:
        with tempfile.TemporaryDirectory(prefix="fraudguard-elliptic-") as temporary:
            archive = Path(temporary) / "elliptic.zip"
            with urllib.request.urlopen(URL, timeout=60) as response, archive.open("wb") as out:
                shutil.copyfileobj(response, out)
            with zipfile.ZipFile(archive) as zipped:
                for name in missing:
                    path = data_dir / name
                    partial = path.with_suffix(".part")
                    try:
                        with (
                            zipped.open(f"elliptic_bitcoin_dataset/{name}") as source,
                            partial.open("wb") as out,
                        ):
                            shutil.copyfileobj(source, out)
                        if file_sha256(partial) != EXPECTED_SHA256[name]:
                            raise ValueError(f"Checksum mismatch for {name}")
                        partial.replace(path)
                    finally:
                        partial.unlink(missing_ok=True)
    print(f"Verified {len(EXPECTED_SHA256)} Elliptic CSV files in {data_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/elliptic"))
    download(parser.parse_args().data_dir)


if __name__ == "__main__":
    main()
