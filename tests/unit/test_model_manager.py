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


def test_stream_failure_reports_exit_code():
    mm = _load()
    from unittest.mock import MagicMock

    proc = MagicMock()
    proc.stdout = iter(["boom\n"])
    proc.poll.return_value = 1
    proc.returncode = 1
    with patch.object(mm.subprocess, "Popen", return_value=proc):
        out = list(mm.stream_model_download("All models (recommended)"))
    assert "code 1" in out[-1]
    assert "boom" in out[-1]


def test_verify_flags_missing(monkeypatch, tmp_path):
    mm = _load()
    monkeypatch.chdir(tmp_path)
    report = mm.verify_models()
    assert "missing" in report.lower()


def test_download_cmd_uses_unbuffered_child():
    mm = _load()
    cmd = mm._download_cmd("inswapper_128.onnx (face swap)", force=True)
    assert "-u" in cmd
    assert "--force" in cmd
    assert cmd[cmd.index("--models") + 1] == "inswapper"


def test_stream_yields_live_lines_and_final_status():
    mm = _load()
    from unittest.mock import MagicMock

    proc = MagicMock()
    proc.stdout = iter(["line1\n", "line2\n"])
    proc.poll.return_value = 0
    proc.returncode = 0
    with patch.object(mm.subprocess, "Popen", return_value=proc) as mock_popen:
        out = list(mm.stream_model_download("All models (recommended)"))
        assert "-u" in mock_popen.call_args.args[0]
    assert "line1" in out[0]
    assert out[-1].startswith("✅ Download finished.")
    assert "line2" in out[-1]


def test_stream_cancel_terminates_proc():
    mm = _load()
    from unittest.mock import MagicMock

    def gen():
        yield "chunk1\n"
        mm.request_download_cancel()
        yield "chunk2\n"

    proc = MagicMock()
    proc.stdout = gen()
    proc.poll.return_value = None
    proc.wait.return_value = 0
    with patch.object(mm.subprocess, "Popen", return_value=proc):
        out = list(mm.stream_model_download("All models (recommended)"))
    proc.terminate.assert_called_once()
    assert "cancelled" in out[-1].lower()


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
