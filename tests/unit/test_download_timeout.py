"""Unit tests for timeout-bounded model downloads (no network)."""

import socket
from unittest.mock import MagicMock, patch


def test_codeformer_download_timeout_cleans_partial(tmp_path):
    from processing.codeformer_restoration import _download_model

    dest_dir = tmp_path / "models"
    with patch(
        "urllib.request.urlopen", side_effect=socket.timeout("stalled")
    ) as mock_open:
        try:
            _download_model("https://example.invalid/m.pth", dest_dir, "m.pth")
        except Exception:
            pass
        else:
            raise AssertionError("expected download failure to raise")
    assert mock_open.call_args.kwargs.get("timeout") == 300
    assert not (dest_dir / "m.pth").exists()


def test_codeformer_download_success_writes_file(tmp_path):
    from processing.codeformer_restoration import _download_model

    resp = MagicMock()
    resp.read.side_effect = [b"data", b""]
    resp.__enter__.return_value = resp
    dest_dir = tmp_path / "models"
    with patch("urllib.request.urlopen", return_value=resp):
        out = _download_model("https://example.invalid/m.pth", dest_dir, "m.pth")
    assert out.read_bytes() == b"data"


def test_setup_script_download_has_timeout(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
    import download_models

    resp = MagicMock()
    resp.headers = {}
    resp.read.side_effect = [b"x" * 1024, b""]
    resp.__enter__.return_value = resp
    dest = tmp_path / "f.bin"
    with patch("urllib.request.urlopen", return_value=resp) as mock_open:
        try:
            download_models._download("https://example.invalid/f.bin", dest, 1)
        except ValueError:
            pass  # size floor may trip; timeout kwarg is what matters here
    assert mock_open.call_args.kwargs.get("timeout") == 300
