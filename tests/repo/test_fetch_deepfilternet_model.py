"""fetch_deepfilternet_model.py — archive extraction, verification and the
CLI, with the download replaced by an in-memory zip (no network)."""

import io
import zipfile

import pytest

import fetch_deepfilternet_model as fetch


def archive(entries, prefix="DeepFilterNet3/"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(prefix, "")                                   # directory entry
        for name, data in entries.items():
            zf.writestr(prefix + name, data)
    return buf.getvalue()


GOOD = {"config.ini": "[df]\nsr = 48000\n", "checkpoints/model_120.ckpt.best": b"\x00" * 64}


@pytest.fixture
def dest(tmp_path, monkeypatch):
    # extract() prints paths relative to PROJECT_ROOT.
    monkeypatch.setattr(fetch, "PROJECT_ROOT", tmp_path)
    return tmp_path / "Models" / "DFN3"


def test_extract_strips_the_model_prefix(dest):
    fetch.extract(archive(GOOD), "DeepFilterNet3", dest)
    assert (dest / "config.ini").read_text() == GOOD["config.ini"]
    assert (dest / "checkpoints" / "model_120.ckpt.best").stat().st_size == 64
    assert not (dest / "DeepFilterNet3").exists()


def test_extract_rejects_an_archive_for_another_model(dest):
    with pytest.raises(RuntimeError, match="no 'DeepFilterNet3/' directory"):
        fetch.extract(archive(GOOD, prefix="Other/"), "DeepFilterNet3", dest)


def test_extract_blocks_parent_traversal(dest):
    with pytest.raises(RuntimeError, match="escapes destination"):
        fetch.extract(archive({"../../evil.txt": "x"}), "DeepFilterNet3", dest)
    assert not (dest.parent.parent / "evil.txt").exists()


@pytest.mark.xfail(strict=True, raises=pytest.fail.Exception, reason=(
    "zip-slip: the guard compares string prefixes, so '../DFN3evil/x' resolves "
    "to a SIBLING directory whose path still starts with '.../DFN3'. Compare "
    "with Path.is_relative_to(dest.resolve()) instead."))
def test_extract_blocks_sibling_prefix_traversal(dest):
    with pytest.raises(RuntimeError, match="escapes destination"):
        fetch.extract(archive({"../DFN3evil/pwn.txt": "x"}), "DeepFilterNet3", dest)


def test_verify_accepts_a_complete_model(dest, capsys):
    fetch.extract(archive(GOOD), "DeepFilterNet3", dest)
    fetch.verify(dest)
    assert "Model sample rate: 48000 Hz" in capsys.readouterr().out


@pytest.mark.parametrize("entries, missing", [
    ({"checkpoints/x": b"1"}, "config.ini"),
    ({"config.ini": "sr = 48000"}, "checkpoints"),
])
def test_verify_names_what_is_missing(dest, entries, missing):
    fetch.extract(archive(entries), "DeepFilterNet3", dest)
    with pytest.raises(RuntimeError, match=missing):
        fetch.verify(dest)


def test_download_uses_the_upstream_url(monkeypatch):
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"zipbytes"

    def urlopen(request):
        seen["url"] = request.full_url
        seen["agent"] = request.get_header("User-agent")
        return Response()

    monkeypatch.setattr(fetch.urllib.request, "urlopen", urlopen)
    assert fetch.download("DeepFilterNet3") == b"zipbytes"
    assert seen["url"].endswith("/models/DeepFilterNet3.zip") and seen["agent"]


def test_cli_skips_a_populated_destination(dest, monkeypatch):
    fetch.extract(archive(GOOD), "DeepFilterNet3", dest)
    monkeypatch.setattr(fetch, "download", lambda m: pytest.fail("must not download"))
    monkeypatch.setattr("sys.argv", ["fetch", "--dest", str(dest)])
    assert fetch.main() == 0


def test_cli_force_redownloads(dest, monkeypatch):
    fetch.extract(archive({**GOOD, "stale.txt": "old"}), "DeepFilterNet3", dest)
    monkeypatch.setattr(fetch, "download", lambda m: archive(GOOD))
    monkeypatch.setattr("sys.argv", ["fetch", "--dest", str(dest), "--force"])
    assert fetch.main() == 0
    assert not (dest / "stale.txt").exists() and (dest / "config.ini").exists()
