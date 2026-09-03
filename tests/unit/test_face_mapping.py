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


def test_clear_mapping_resets_display():
    mod = _helper()
    manager = MagicMock()
    status, text = mod.clear_face_mappings(manager)
    manager.clear.assert_called_once()
    assert text == "No mappings"
