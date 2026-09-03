"""Unit tests for the compare-with-original view (no GPU)."""

import gradio as gr
import numpy as np
import pytest


def _load():
    from ui.helpers import controls

    return controls


def test_fill_compare_passes_values_through():
    mod = _load()
    assert mod.fill_compare_view("/tmp/target.png") == "/tmp/target.png"
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    assert mod.fill_compare_view(arr) is arr


def test_fill_compare_skips_when_no_target():
    mod = _load()
    assert isinstance(mod.fill_compare_view(None), type(gr.skip()))


@pytest.mark.parametrize("checked", [True, False])
def test_toggle_compare_view_flips_visibility(checked):
    mod = _load()
    update = mod.toggle_compare_view(checked)
    assert update["visible"] is checked


def _build(create):
    with gr.Blocks():
        return create()


def test_image_tab_exposes_compare_components():
    from ui.components.image_tab import create_image_tab

    comps = _build(create_image_tab)
    assert comps["compare_toggle"].value is False
    assert comps["compare_img"].visible is False


def test_video_tab_exposes_compare_components():
    from ui.components.video_tab import create_video_tab

    comps = _build(create_video_tab)
    assert comps["compare_toggle_vid"].value is False
    assert comps["compare_vid"].visible is False


def test_gif_tab_exposes_compare_components():
    from ui.components.gif_tab import create_gif_tab

    comps = _build(create_gif_tab)
    assert comps["compare_toggle_gif"].value is False
    assert comps["compare_gif"].visible is False
