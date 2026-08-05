"""Unit tests for progress tracking."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_null_progress_bar_supports_every_attribute_the_pipeline_uses():
    """NullProgressBar must satisfy the whole tqdm surface the codebase calls.

    NullProgressBar is used whenever progress bars are suppressed, which
    _should_show() decides by checking for a TTY. Any headless launch — nohup,
    systemd, Docker without -t, output redirected to a log — therefore takes
    this path, while an interactive terminal never does. Drift between this
    stub and tqdm is invisible to developers running in a terminal and crashes
    every GIF/video job for everyone else, so it is asserted rather than
    assumed.
    """
    from utils.progress import NullProgressBar

    bar = NullProgressBar()

    # Methods called on progress bars in processing/streaming_media.py et al.
    for name in ("update", "refresh", "close", "set_description", "set_postfix"):
        assert callable(getattr(bar, name, None)), f"NullProgressBar lacks {name}()"

    # Exercise them the way the pipeline does — none may raise.
    bar.update(1)
    bar.set_description("x")
    bar.set_postfix(a=1)
    bar.total = 100
    bar.n = 100
    bar.refresh()
    bar.close()

    assert bar.n == 100
    assert bar.total == 100


@pytest.mark.unit
def test_null_progress_bar_covers_methods_grepped_from_source():
    """Guard against future drift: scan real call sites, assert each exists."""
    from utils.progress import NullProgressBar

    called = set()
    pattern = re.compile(r"\bpbar\.([a-z_]+)")
    for sub in ("processing", "core", "ui", "utils"):
        for py in (ROOT / sub).rglob("*.py"):
            called.update(pattern.findall(py.read_text(encoding="utf-8")))

    missing = [m for m in sorted(called) if not hasattr(NullProgressBar(), m)]
    assert not missing, (
        f"NullProgressBar is missing {missing}, which source code calls on a "
        f"progress bar. Headless runs (no TTY) would crash with AttributeError."
    )
