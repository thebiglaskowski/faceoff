"""Unit tests for gallery bulk delete / clear-all (no GPU)."""

from pathlib import Path


def _seed(tmp_path, media_type, names):
    from utils.output_metadata import metadata_path_for

    target = tmp_path / "outputs" / media_type
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        (target / name).write_bytes(b"data")
        metadata_path_for(target / name).write_bytes(b'{"sidecar": true}')
    return target


def test_delete_all_removes_type_and_sidecars(monkeypatch, tmp_path):
    from ui.helpers import gallery_utils as gu

    monkeypatch.chdir(tmp_path)
    _seed(tmp_path, "image", ["a.png", "b.jpg"])
    _seed(tmp_path, "gif", ["keep.gif"])

    deleted, failed = gu.delete_all_files("image")
    assert (deleted, failed) == (2, 0)
    assert list((tmp_path / "outputs" / "image").iterdir()) == []
    assert (tmp_path / "outputs" / "gif" / "keep.gif").exists()


def test_delete_all_empty_is_zero(monkeypatch, tmp_path):
    from ui.helpers import gallery_utils as gu

    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "video").mkdir(parents=True)
    assert gu.delete_all_files("video") == (0, 0)


def test_delete_all_ignores_outside_files(monkeypatch, tmp_path):
    from ui.helpers import gallery_utils as gu

    monkeypatch.chdir(tmp_path)
    _seed(tmp_path, "image", ["a.png"])
    stray = tmp_path / "stray.png"
    stray.write_bytes(b"do not touch")

    deleted, failed = gu.delete_all_files("image")
    assert (deleted, failed) == (1, 0)
    assert stray.exists()


def test_clear_all_arms_before_deleting(monkeypatch, tmp_path):
    from ui.components import gallery_tab as gt

    monkeypatch.chdir(tmp_path)
    _seed(tmp_path, "image", ["a.png", "b.png"])

    status, gallery, _count, btn, armed = gt.clear_all_media("Images", "24", "")
    assert "again" in status["value"]
    assert btn["value"] == gt.CONFIRM_CLEAR_LABEL
    assert armed == "Images"
    # Nothing deleted by the arming click.
    assert len(list((tmp_path / "outputs" / "image").glob("*.png"))) == 2

    status, gallery, _count, btn, armed = gt.clear_all_media("Images", "24", "Images")
    assert "Deleted 2" in status["value"]
    assert btn["value"] == gt.CLEAR_ALL_LABEL
    assert armed == ""
    assert gallery["value"] == []
    assert list((tmp_path / "outputs" / "image").iterdir()) == []


def test_clear_all_type_switch_rearms(monkeypatch, tmp_path):
    from ui.components import gallery_tab as gt

    monkeypatch.chdir(tmp_path)
    _seed(tmp_path, "image", ["a.png"])
    _seed(tmp_path, "gif", ["b.gif"])

    _status, _g, _c, _b, armed = gt.clear_all_media("Images", "24", "")
    assert armed == "Images"
    # Confirming while viewing another type must NOT delete images.
    status, _g, _c, _b, armed = gt.clear_all_media("GIFs", "24", armed)
    assert "again" in status["value"]
    assert armed == "GIFs"
    assert (tmp_path / "outputs" / "image" / "a.png").exists()


def test_clear_all_empty_and_unknown(monkeypatch, tmp_path):
    from ui.components import gallery_tab as gt

    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "image").mkdir(parents=True)

    status, _g, _c, btn, armed = gt.clear_all_media("Images", "24", "")
    assert "Nothing to delete" in status["value"]
    assert btn["value"] == gt.CLEAR_ALL_LABEL
    assert armed == ""

    status, _g, _c, _b, armed = gt.clear_all_media("???", "24", "")
    assert "Unknown media type" in status["value"]
    assert armed == ""


def test_disarm_resets_label_and_state():
    from ui.components.gallery_tab import CLEAR_ALL_LABEL, disarm_clear_all

    btn, armed = disarm_clear_all()
    assert btn["value"] == CLEAR_ALL_LABEL
    assert armed == ""


def test_gallery_tab_exposes_bulk_components():
    import gradio as gr

    from ui.components.gallery_tab import (
        CLEAR_ALL_LABEL,
        create_gallery_tab,
        wire_gallery_tab,
    )

    with gr.Blocks():
        comps = create_gallery_tab()
    assert comps["clear_all_btn"].value == CLEAR_ALL_LABEL
    assert comps["bulk_status"].visible is False
    with gr.Blocks():
        comps = create_gallery_tab()
        wire_gallery_tab(comps)  # must not raise


def test_single_delete_status_is_visible(monkeypatch, tmp_path):
    from ui.components import gallery_tab as gt

    monkeypatch.chdir(tmp_path)
    target = tmp_path / "outputs" / "image"
    target.mkdir(parents=True)
    (target / "gone.png").write_bytes(b"png")

    status, _g, _c, _b, _s, _m = gt.delete_and_refresh("gone.png", "Images", "24")
    assert status["visible"] is True
    assert "Deleted" in status["value"]
    assert not (target / "gone.png").exists()
