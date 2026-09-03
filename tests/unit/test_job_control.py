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

    def fake_fn(*args):
        calls.append(args)
        return "RESULT"

    out = list(swap_with_button_toggle(fake_fn)("a", "b"))
    assert calls == [("a", "b")]
    assert out[0][1]["interactive"] is False
    assert out[1] == ("RESULT", {"interactive": True, "__type__": "update"})


def test_swap_wrapper_reenables_on_error():
    from ui.helpers.controls import swap_with_button_toggle

    def boom(*args):
        raise ValueError("nope")

    gen = swap_with_button_toggle(boom)()
    first = next(gen)
    assert first[1]["interactive"] is False
    second = next(gen)  # re-enable yield from the except branch
    assert second[1]["interactive"] is True
    try:
        next(gen)
    except ValueError:
        pass
    else:
        raise AssertionError("expected the error to propagate")
