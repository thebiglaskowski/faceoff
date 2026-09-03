"""Unit tests for gallery file deletion confinement (no GPU)."""


def _load():
    from ui.helpers import gallery_utils

    return gallery_utils


def test_legit_gallery_delete_works(monkeypatch, tmp_path):
    mod = _load()
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "outputs" / "gif"
    target.mkdir(parents=True)
    victim = target / "swapped_1.gif"
    victim.write_bytes(b"gif")
    ok, msg = mod.delete_file(str(target / "swapped_1.gif"), "gif")
    assert ok, msg
    assert not victim.exists()


def test_traversal_delete_blocked(monkeypatch, tmp_path):
    mod = _load()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "gif").mkdir(parents=True)
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"do not delete")
    ok, msg = mod.delete_file(
        str(tmp_path / "outputs" / "gif" / ".." / ".." / "secret.txt"), "gif"
    )
    assert not ok
    assert "mismatch" in msg
    assert secret.exists()


def test_wrong_media_type_blocked(monkeypatch, tmp_path):
    mod = _load()
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "outputs" / "image"
    target.mkdir(parents=True)
    pic = target / "a.png"
    pic.write_bytes(b"png")
    ok, _ = mod.delete_file(str(pic), "gif")
    assert not ok
    assert pic.exists()
