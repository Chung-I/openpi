import pathlib
import subprocess

import pytest

import openpi.shared.download as download


@pytest.fixture(scope="session", autouse=True)
def set_openpi_data_home(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("openpi_data")
    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("OPENPI_DATA_HOME", str(temp_dir))
        yield


def test_download_local(tmp_path: pathlib.Path):
    local_path = tmp_path / "local"
    local_path.touch()

    result = download.maybe_download(str(local_path))
    assert result == local_path

    with pytest.raises(FileNotFoundError):
        download.maybe_download("bogus")


def test_download_gs_dir():
    remote_path = "gs://openpi-assets/testdata/random"

    local_path = download.maybe_download(remote_path)
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path)
    assert new_local_path == local_path


def test_download_gs():
    remote_path = "gs://openpi-assets/testdata/random/random_512kb.bin"

    local_path = download.maybe_download(remote_path)
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path)
    assert new_local_path == local_path


def test_download_fsspec():
    remote_path = "gs://big_vision/paligemma_tokenizer.model"

    local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert new_local_path == local_path


# The tests below need no network. They cover the gsutil path directly, which the
# network tests above cannot: on a machine WITHOUT gsutil installed, `_download_gsutil`
# falls back to fsspec immediately and the gsutil branch is never executed. That is why
# a real single-object failure (see below) went unnoticed in CI.


def test_gsutil_failure_falls_back_to_fsspec(tmp_path, monkeypatch):
    """gsutil is invoked as `{url}/*`, so a single OBJECT matches nothing and it exits 1.

    Not hypothetical: every DROID RLDS config references
    `gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json`, one object in a bucket
    routed through gsutil. Without this fallback, DROID training dies before step 0 on
    any machine that has gsutil installed and lacks a warm cache.
    """
    monkeypatch.setattr(download.shutil, "which", lambda _: "/usr/bin/gsutil")

    def _fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "gsutil")

    monkeypatch.setattr(download.subprocess, "run", _fail)

    called: list[tuple[str, pathlib.Path]] = []
    monkeypatch.setattr(download, "_download_fsspec", lambda url, path, **kw: called.append((url, path)))

    target = tmp_path / "out"
    download._download_gsutil("gs://openpi-assets/droid/ranges.json", target)  # noqa: SLF001

    assert called == [("gs://openpi-assets/droid/ranges.json", target)]


def test_gsutil_success_does_not_fall_back(tmp_path, monkeypatch):
    monkeypatch.setattr(download.shutil, "which", lambda _: "/usr/bin/gsutil")
    monkeypatch.setattr(download.subprocess, "run", lambda *a, **k: None)

    called: list = []
    monkeypatch.setattr(download, "_download_fsspec", lambda *a, **k: called.append(a))

    download._download_gsutil("gs://openpi-assets/checkpoints/x", tmp_path / "out")  # noqa: SLF001

    assert not called, "fsspec must not be used when gsutil succeeds"


def test_missing_gsutil_falls_back_to_fsspec(tmp_path, monkeypatch):
    monkeypatch.setattr(download.shutil, "which", lambda _: None)

    called: list = []
    monkeypatch.setattr(download, "_download_fsspec", lambda url, path, **kw: called.append((url, path)))

    target = tmp_path / "out"
    download._download_gsutil("gs://openpi-assets/droid/ranges.json", target)  # noqa: SLF001

    assert called == [("gs://openpi-assets/droid/ranges.json", target)]
