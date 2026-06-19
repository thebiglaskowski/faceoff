"""Tests for output compression utilities."""

from unittest.mock import patch

from utils.compression import EXTERNAL_GIFSICLE, _find_gifsicle


def test_find_gifsicle_prefers_system_path_on_linux():
    with patch("utils.compression.sys.platform", "linux"):
        with patch("utils.compression.shutil.which", return_value="/usr/bin/gifsicle") as which:
            assert _find_gifsicle() == "/usr/bin/gifsicle"
            which.assert_called_once_with("gifsicle")


def test_find_gifsicle_uses_bundled_exe_on_windows():
    with patch("utils.compression.sys.platform", "win32"):
        with patch.object(type(EXTERNAL_GIFSICLE), "exists", return_value=True):
            assert _find_gifsicle() == str(EXTERNAL_GIFSICLE)


def test_find_gifsicle_skips_windows_exe_on_linux_even_if_present():
    with patch("utils.compression.sys.platform", "linux"):
        with patch("utils.compression.shutil.which", return_value="/usr/bin/gifsicle") as which:
            with patch.object(type(EXTERNAL_GIFSICLE), "exists", return_value=True):
                assert _find_gifsicle() == "/usr/bin/gifsicle"
                which.assert_called_once_with("gifsicle")