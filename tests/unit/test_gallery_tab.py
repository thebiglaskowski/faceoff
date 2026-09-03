"""Unit tests for gallery tab states (no GPU)."""


def _load():
    from ui.components import gallery_tab

    return gallery_tab


def test_bad_limit_falls_back_without_raising(monkeypatch, tmp_path):
    mod = _load()
    monkeypatch.chdir(tmp_path)
    gallery_update, count_update = mod.update_gallery("Images", "not-a-number")
    assert gallery_update["value"] == []


def test_empty_gallery_hints_at_next_step(monkeypatch, tmp_path):
    mod = _load()
    monkeypatch.chdir(tmp_path)
    _, count_update = mod.update_gallery("GIFs", "24")
    assert "run a swap" in count_update["value"]


def test_unknown_media_type_defaults_to_images(monkeypatch, tmp_path):
    mod = _load()
    monkeypatch.chdir(tmp_path)
    gallery_update, _ = mod.update_gallery("???", "24")
    assert gallery_update["value"] == []
