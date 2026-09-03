"""Unit tests for job cancellation (no GPU)."""


def _load():
    from processing import job_control

    return job_control


def test_cancel_flag_roundtrip():
    jc = _load()
    jc.clear_job_cancel()
    assert jc.job_cancel_requested() is False
    jc.request_job_cancel()
    assert jc.job_cancel_requested() is True
    jc.clear_job_cancel()
    assert jc.job_cancel_requested() is False


def test_raise_if_cancelled_only_when_set():
    import pytest

    jc = _load()
    jc.clear_job_cancel()
    jc.raise_if_cancelled()  # must not raise
    jc.request_job_cancel()
    with pytest.raises(jc.JobCancelled):
        jc.raise_if_cancelled()
    jc.clear_job_cancel()


def test_swap_wrapper_toggles_button():
    import gradio as gr

    from ui.helpers.controls import swap_with_button_toggle

    calls = []

    def fake_fn(*args, **kwargs):
        calls.append(args)
        return "RESULT"

    out = list(swap_with_button_toggle(fake_fn)("a", "b"))
    assert calls == [("a", "b")]
    assert out[0][1]["interactive"] is False
    assert out[0][2]["visible"] is False  # error box cleared on run start
    assert out[1][0] == "RESULT"
    assert out[1][1] == {"interactive": True, "__type__": "update"}
    assert out[1][2]["visible"] is False  # stays hidden on success


def test_swap_wrapper_forwards_progress():
    from unittest.mock import MagicMock

    from ui.helpers.controls import swap_with_button_toggle

    seen = {}

    def fake_fn(*args, **kwargs):
        seen.update(kwargs)
        return "R"

    fake_progress = MagicMock()
    out = list(swap_with_button_toggle(fake_fn)("x", progress=fake_progress))
    assert seen.get("progress") is fake_progress
    assert out[-1][0] == "R"


def test_swap_wrapper_reenables_on_error():
    from ui.helpers.controls import swap_with_button_toggle

    def boom(*args, **kwargs):
        raise ValueError("nope")

    gen = swap_with_button_toggle(boom)()
    first = next(gen)
    assert first[1]["interactive"] is False
    second = next(gen)  # re-enable yield from the except branch
    assert second[1]["interactive"] is True
    assert second[2]["visible"] is True  # unexpected error renders inline
    assert "nope" in second[2]["value"]
    try:
        next(gen)
    except ValueError:
        pass
    else:
        raise AssertionError("expected the error to propagate")


def test_swap_wrapper_renders_gr_error_inline_without_raising():
    import gradio as gr

    from ui.helpers.controls import swap_with_button_toggle

    def missing_input(*args, **kwargs):
        raise gr.Error("Please upload a source image.")

    out = list(swap_with_button_toggle(missing_input)())
    assert len(out) == 2  # start yield + inline error; no re-raise
    assert out[1][1]["interactive"] is True
    assert out[1][2]["visible"] is True
    assert "Please upload a source image." in out[1][2]["value"]


def test_swap_error_text_falls_back_when_empty():
    from ui.helpers.controls import swap_error_text

    assert swap_error_text(ValueError()) == "⚠️ Swap failed."
    assert "boom" in swap_error_text(ValueError("boom"))


def test_all_swap_tabs_wire_inline_error_box():
    import gradio as gr

    from ui.components.gallery_tab import create_gallery_tab
    from ui.components.gif_tab import create_gif_tab, wire_gif_tab
    from ui.components.image_tab import create_image_tab, wire_image_tab
    from ui.components.video_tab import create_video_tab, wire_video_tab

    with gr.Blocks():
        gallery_components = create_gallery_tab()
        for create, wire in (
            (create_image_tab, wire_image_tab),
            (create_video_tab, wire_video_tab),
            (create_gif_tab, wire_gif_tab),
        ):
            comps = create()
            assert comps["swap_error"].visible is False
            wire(comps, gallery_components)  # must not raise
