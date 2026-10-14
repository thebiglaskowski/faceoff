"""Unit tests for face-mapping UI helpers (no GPU)."""

from unittest.mock import MagicMock


def _helper():
    from ui.helpers import face_mapping

    return face_mapping


def test_add_mapping_parses_dropdown_labels():
    mod = _helper()
    manager = MagicMock()
    manager.get_display_text.return_value = "0 → 1"
    status, text = mod.add_face_mapping(
        "Source Face 0", "Target Face 1", "No mappings", manager
    )
    manager.add.assert_called_once_with(0, 1)
    assert "Added mapping" in status
    assert text == "0 → 1"


def test_add_mapping_none_selection_warns():
    mod = _helper()
    manager = MagicMock()
    status, text = mod.add_face_mapping(None, "Target Face 1", "No mappings", manager)
    assert "Select both" in status
    assert text == "No mappings"
    manager.add.assert_not_called()


def test_add_mapping_garbage_selection_warns_without_raising():
    mod = _helper()
    manager = MagicMock()
    for bad in ("", "???", "Source Face X"):
        status, text = mod.add_face_mapping(
            bad, "Target Face 1", "No mappings", manager
        )
        assert "Detect Faces" in status
        assert text == "No mappings"
    manager.add.assert_not_called()


def test_swap_frame_ignores_negative_mapping_indices():
    """Negative mapping indices must not silently swap the last face."""
    import numpy as np

    from processing.frame_batch import _swap_frame

    calls = []

    class FakeProcessor:
        def swap_face(self, frame, dst, src):
            calls.append((dst, src))
            return frame

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    _swap_frame(frame, [object()], [object()], [(-1, -1)], processor=FakeProcessor())
    assert calls == []


def test_clear_mapping_resets_display():
    mod = _helper()
    manager = MagicMock()
    status, text = mod.clear_face_mappings(manager)
    manager.clear.assert_called_once()
    assert text == "No mappings"


def _select(index):
    from unittest.mock import MagicMock

    evt = MagicMock()
    evt.index = index
    return evt


def test_click_thumbnail_selects_dropdown_value():
    import gradio as gr

    mod = _helper()
    out = mod.select_mapping_index(_select(2), "Source Face")
    assert out["value"] == "Source Face 2"
    assert mod.select_mapping_index(_select(None), "Source Face") == gr.skip()


def test_click_target_maps_immediately():
    mod = _helper()
    calls = []

    def fake_add(s, t, c):
        calls.append((s, t, c))
        return "Added mapping: 0 → 1", "0 → 1"

    tgt, st, mp = mod.pick_target_and_map(
        _select(1), "Source Face 0", "cur", fake_add, "Target Face"
    )
    assert calls == [(0, 1, "cur")]
    assert tgt["value"] == "Target Face 1"
    assert mp == "0 → 1"


def test_click_target_without_source_warns():
    mod = _helper()

    def fake_add(s, t, c):  # pragma: no cover - must not run
        raise AssertionError("must not add without a source")

    _, status, current = mod.pick_target_and_map(
        _select(1), None, "cur", fake_add, "Target Face"
    )
    assert "source face first" in status
    assert current == "cur"


def test_auto_map_single_pair_only():
    mod = _helper()
    calls = []

    def fake_add(s, t, c):
        calls.append((s, t))
        return "Added", "0 → 0"

    status, mappings = mod.auto_map_single_pair(["a"], ["b"], "cur", fake_add)
    assert calls == [(0, 0)]
    assert mappings == "0 → 0"

    import gradio as gr

    assert mod.auto_map_single_pair(["a", "b"], ["c"], "cur", fake_add) == (
        gr.skip(),
        gr.skip(),
    )
    assert mod.auto_map_single_pair([], [], "cur", fake_add) == (
        gr.skip(),
        gr.skip(),
    )
    assert len(calls) == 1
