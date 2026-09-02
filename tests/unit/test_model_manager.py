"""Unit tests for processing/model_manager.py (no network, no GPU)."""

from unittest.mock import MagicMock, patch


def _load():
    from processing import model_manager

    return model_manager


def test_status_reports_missing(monkeypatch, tmp_path):
    mm = _load()
    monkeypatch.chdir(tmp_path)
    rows = mm.get_model_status()
    assert rows[0][0] == "inswapper_128.onnx"
    assert "Missing" in rows[0][2]
    assert "Missing" in rows[1][2]


def test_status_reports_installed(monkeypatch, tmp_path):
    mm = _load()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "inswapper_128.onnx").write_bytes(b"x" * 1024)
    buffalo = tmp_path / "models" / "buffalo_l"
    buffalo.mkdir(parents=True)
    (buffalo / "det_10g.onnx").write_bytes(b"y" * 1024)
    rows = mm.get_model_status()
    assert "Installed" in rows[0][2]
    assert "Installed" in rows[1][2]


def test_download_maps_selection_to_cli_flag():
    mm = _load()
    with patch.object(mm.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="done", stderr="")
        log = mm.run_model_download("inswapper_128.onnx (face swap)")
        cmd = mock_run.call_args.args[0]
        assert "--models" in cmd
        assert cmd[cmd.index("--models") + 1] == "inswapper"
        assert "finished" in log


def test_download_failure_reports_exit_code():
    mm = _load()
    with patch.object(mm.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="boom")
        log = mm.run_model_download("All models (recommended)")
        assert "code 1" in log
        assert "boom" in log


def test_download_timeout_is_clean():
    mm = _load()
    with patch.object(
        mm.subprocess,
        "run",
        side_effect=mm.subprocess.TimeoutExpired(cmd="x", timeout=1),
    ):
        assert "timed out" in mm.run_model_download("All models (recommended)")


def test_verify_flags_missing(monkeypatch, tmp_path):
    mm = _load()
    monkeypatch.chdir(tmp_path)
    report = mm.verify_models()
    assert "missing" in report.lower()


def test_clear_cache_handles_empty(monkeypatch, tmp_path):
    mm = _load()
    monkeypatch.chdir(tmp_path)
    with patch("utils.model_cache.get_cache_info") as mock_info:
        mock_info.return_value = {
            "num_files": 0,
            "total_size_mb": 0.0,
            "cache_dir": str(tmp_path / "cache"),
        }
        assert "Cleared 0" in mm.clear_trt_cache()
