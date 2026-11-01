"""Convert a checksum-pinned official Handbook archive to a local replay CSV."""

import argparse
import io
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from scripts.download_creditcard import file_sha256

SOURCE_COMMIT = "6e67dbd0a3bfe0d7ec33abc4bce5f37cd4ff0d6a"
URL = f"https://codeload.github.com/Fraud-Detection-Handbook/simulated-data-raw/zip/{SOURCE_COMMIT}"
ARCHIVE_SHA256 = "d2d38596905f3e2d8d482c49d59d38aad2b1a726f8bc00dcfe6d29d3a93c043a"
CSV_SHA256 = "7f63c8e8d663144b137ffce42e6f2d5ffc20d4b1c489d8edd8be793cd4c0c020"
START_DATE = date(2018, 4, 1)
END_DATE = date(2018, 7, 15)
COLUMNS = [
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_TIME_SECONDS",
    "TX_TIME_DAYS",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
]


def convert_archive(archive: Path, destination: Path) -> None:
    # Pickle can execute code: only read bytes from this verified official archive.
    if file_sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Handbook archive checksum mismatch; no pickle was read")
    partial = destination.with_suffix(".part")
    try:
        with zipfile.ZipFile(archive) as zipped, partial.open("w", encoding="utf-8") as out:
            day = START_DATE
            while day <= END_DATE:
                name = f"simulated-data-raw-{SOURCE_COMMIT}/data/{day}.pkl"
                frame = pd.read_pickle(io.BytesIO(zipped.read(name)))
                if list(frame.columns) != COLUMNS or not frame.TX_DATETIME.dt.date.eq(day).all():
                    raise ValueError(f"Unexpected Handbook schema or dates in {name}")
                frame.to_csv(
                    out,
                    index=False,
                    header=day == START_DATE,
                    date_format="%Y-%m-%d %H:%M:%S",
                )
                day += timedelta(days=1)
        if file_sha256(partial) != CSV_SHA256:
            raise ValueError("Converted Handbook CSV checksum mismatch")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def download(data_dir: Path, archive: Path | None = None) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / "transactions.csv"
    if destination.exists():
        if file_sha256(destination) != CSV_SHA256:
            raise ValueError("Existing Handbook CSV checksum mismatch; file was not overwritten")
    elif archive is not None:
        convert_archive(archive, destination)
    else:
        with tempfile.TemporaryDirectory(prefix="fraudguard-handbook-download-") as temporary:
            archive = Path(temporary) / "source.zip"
            with urllib.request.urlopen(URL, timeout=60) as response, archive.open("wb") as out:
                shutil.copyfileobj(response, out)
            convert_archive(archive, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/handbook"))
    parser.add_argument("--archive", type=Path, help="Reuse the checksum-pinned source archive")
    args = parser.parse_args()
    print(f"Verified Handbook CSV: {download(args.data_dir, args.archive)}")


if __name__ == "__main__":
    main()
