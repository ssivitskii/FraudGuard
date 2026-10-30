import hashlib
import io
import zipfile

import pytest

from scripts import download_elliptic


@pytest.fixture
def archive(monkeypatch):
    name = "elliptic_txs_classes.csv"
    data = b"txId,class\n1,unknown\n"
    digest = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(download_elliptic, "EXPECTED_SHA256", {name: digest})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zipped:
        zipped.writestr(f"elliptic_bitcoin_dataset/{name}", data)
        zipped.writestr("../unexpected.txt", b"must not be extracted")
    payload = buffer.getvalue()
    monkeypatch.setattr(
        download_elliptic.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(payload)
    )
    return name, data


def test_download_verifies_only_expected_member_and_reuses_cache(tmp_path, archive, monkeypatch):
    name, data = archive
    destination = tmp_path / "raw"
    download_elliptic.download(destination)
    assert (destination / name).read_bytes() == data
    assert not (tmp_path / "unexpected.txt").exists()

    def network_is_not_needed(*args, **kwargs):
        pytest.fail("verified files should not be downloaded again")

    monkeypatch.setattr(download_elliptic.urllib.request, "urlopen", network_is_not_needed)
    download_elliptic.download(destination)


def test_corrupt_existing_file_is_not_overwritten(tmp_path, archive):
    name, _ = archive
    path = tmp_path / name
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="existing file was not overwritten"):
        download_elliptic.download(tmp_path)
    assert path.read_bytes() == b"corrupt"


def test_bad_download_is_not_published(tmp_path, archive, monkeypatch):
    name, _ = archive
    monkeypatch.setattr(download_elliptic, "EXPECTED_SHA256", {name: "0" * 64})
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_elliptic.download(tmp_path)
    assert not (tmp_path / name).exists()
    assert not list(tmp_path.glob("*.part"))
