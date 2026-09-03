"""Unit tests for scripts/download_models.py (no network access)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def _load_module():
    import download_models

    return download_models


def test_model_urls_point_upstream_not_this_repo():
    dm = _load_module()
    assert "thebiglaskowski/faceoff" not in dm.INSWAPPER_URL
    assert "thebiglaskowski/faceoff" not in dm.BUFFALO_L_URL
    assert dm.INSWAPPER_URL.startswith("https://")
    assert dm.BUFFALO_L_URL.startswith("https://")


def test_fetch_inswapper_skips_when_present(tmp_path):
    dm = _load_module()
    dest = tmp_path / "inswapper_128.onnx"
    dest.write_bytes(b"fake-model")
    with patch.object(dm, "_download") as mock_dl:
        assert dm.fetch_inswapper(tmp_path, dm.INSWAPPER_URL) == dest
        mock_dl.assert_not_called()


def test_fetch_buffalo_l_skips_when_present(tmp_path):
    dm = _load_module()
    dest_dir = tmp_path / "models" / "buffalo_l"
    dest_dir.mkdir(parents=True)
    (dest_dir / "det_10g.onnx").write_bytes(b"fake-model")
    with patch.object(dm, "_download") as mock_dl:
        assert dm.fetch_buffalo_l(tmp_path, dm.BUFFALO_L_URL) == dest_dir
        mock_dl.assert_not_called()


def test_download_rejects_undersized_payload(tmp_path):
    """An error page (small body) must not be kept as a model file."""
    dm = _load_module()
    dest = tmp_path / "inswapper_128.onnx"
    resp = MagicMock()
    resp.headers = {"Content-Length": "12"}
    resp.read.side_effect = [b"404: Not", b" Found", b""]
    resp.__enter__.return_value = resp
    with (
        patch.object(dm.urllib.request, "urlopen", return_value=resp),
        pytest.raises(ValueError, match="expected at least"),
    ):
        dm._download("https://example.invalid/model.onnx", dest, 400 * 1024 * 1024)
    assert not dest.exists()
    assert not dest.with_suffix(dest.suffix + ".part").exists()


def test_main_force_redownloads(tmp_path):
    dm = _load_module()
    (tmp_path / "inswapper_128.onnx").write_bytes(b"fake-model")
    marker_dir = tmp_path / "models" / "buffalo_l"
    marker_dir.mkdir(parents=True)
    (marker_dir / "det_10g.onnx").write_bytes(b"fake-model")
    with (
        patch.object(dm, "fetch_inswapper") as mock_swap,
        patch.object(dm, "fetch_buffalo_l") as mock_buffalo,
    ):
        assert dm.main(["--models", "all", "--force"]) == 0
        mock_swap.assert_called_once()
        assert mock_swap.call_args.kwargs.get("force") is True
        mock_buffalo.assert_called_once()


def test_missing_model_error_points_at_download_script():
    from core.media_processor import MediaProcessor

    with (
        patch("core.media_processor.config") as mock_config,
        patch("core.media_processor.is_tensorrt_runtime_available", return_value=False),
    ):
        mock_config.inswapper_model_path = "/nonexistent/swap.onnx"
        mock_config.buffalo_model_path = "/nonexistent/buffalo_l"
        with pytest.raises(FileNotFoundError, match="download_models"):
            MediaProcessor()
