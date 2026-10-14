"""Unit tests for utils/temp_manager.py (no GPU)."""

import time as time_module


def _manager(tmp_path):
    from utils.temp_manager import TempManager

    return TempManager(str(tmp_path / "temp"))


def test_temp_file_names_unique_when_clock_frozen(tmp_path, monkeypatch):
    """Same-millisecond callers must not be handed the same temp path."""
    monkeypatch.setattr(time_module, "time", lambda: 1_700_000_000.0)
    manager = _manager(tmp_path)

    first = manager.get_temp_file("image", suffix=".png")
    second = manager.get_temp_file("image", suffix=".png")

    assert first != second
    assert first.parent == second.parent
    assert first.suffix == second.suffix == ".png"


def test_temp_directories_unique_when_clock_frozen(tmp_path, monkeypatch):
    """Nested temp dirs at the same timestamp must not share a directory."""
    monkeypatch.setattr(time_module, "time", lambda: 1_700_000_000.0)
    manager = _manager(tmp_path)

    with manager.temp_directory("video") as outer:
        (outer / "outer.txt").write_text("outer")
        with manager.temp_directory("video") as inner:
            assert inner != outer
            assert (outer / "outer.txt").exists()
        # Inner cleanup must not delete the outer directory or its files.
        assert (outer / "outer.txt").read_text() == "outer"
