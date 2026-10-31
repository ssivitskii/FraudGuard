import hashlib
import io
import zipfile
from datetime import date

import pandas as pd
import pytest

from scripts import download_handbook as downloader


def test_rejects_unverified_archive_before_reading_pickle(tmp_path, monkeypatch):
    archive = tmp_path / "unverified.zip"
    archive.write_bytes(b"not the official archive")

    def unexpected_read(*args, **kwargs):
        pytest.fail("Unverified pickle must not be read")

    monkeypatch.setattr(pd, "read_pickle", unexpected_read)
    with pytest.raises(ValueError, match="archive checksum"):
        downloader.download(tmp_path / "data", archive)
    assert not (tmp_path / "data/transactions.csv").exists()


def test_verified_conversion_and_cache_integrity(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        [[1, pd.Timestamp("2018-04-01 12:00:00"), 2, 3, 4.5, 43200, 0, 1, 2]],
        columns=downloader.COLUMNS,
    )
    buffer = io.BytesIO()
    frame.to_pickle(buffer)
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(
            f"simulated-data-raw-{downloader.SOURCE_COMMIT}/data/2018-04-01.pkl",
            buffer.getvalue(),
        )
    csv = frame.to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S").encode()
    monkeypatch.setattr(downloader, "END_DATE", date(2018, 4, 1))
    monkeypatch.setattr(
        downloader, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(downloader, "CSV_SHA256", hashlib.sha256(csv).hexdigest())
    destination = downloader.download(tmp_path / "data", archive)
    assert destination.read_bytes() == csv
    archive.unlink()
    assert downloader.download(tmp_path / "data") == destination
    destination.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="not overwritten"):
        downloader.download(tmp_path / "data")
    assert destination.read_bytes() == b"corrupted"


def test_conversion_failure_preserves_destination(tmp_path, monkeypatch):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w"):
        pass
    monkeypatch.setattr(
        downloader, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    destination = tmp_path / "transactions.csv"
    destination.write_bytes(b"existing data")
    with pytest.raises(KeyError):
        downloader.convert_archive(archive, destination)
    assert destination.read_bytes() == b"existing data"
    assert not destination.with_suffix(".part").exists()
