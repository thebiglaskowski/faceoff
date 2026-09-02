"""Unit tests for utils/legacy_torch.py (no GPU, no weights)."""

from types import SimpleNamespace


def _load():
    from utils import legacy_torch

    return legacy_torch


def _frame(name, parent=None):
    return SimpleNamespace(f_globals={"__name__": name}, f_back=parent)


def test_legacy_packages_detected():
    lt = _load()
    assert lt.module_needs_legacy_load("basicsr.models.archs")
    assert lt.module_needs_legacy_load("gfpgan")
    assert lt.module_needs_legacy_load("facexlib.utils")
    assert not lt.module_needs_legacy_load("processing.hat_enhancement")
    assert not lt.module_needs_legacy_load("torch.serialization")
    assert not lt.module_needs_legacy_load("")


def test_stack_walk_finds_legacy_caller():
    lt = _load()
    coln = _frame("torch.serialization")
    mid = _frame("processing.enhancement", coln)
    leaf = _frame("basicsr.archs.rrdbnet_arch", mid)
    assert lt._legacy_caller_on_stack(leaf) is True
    assert lt._legacy_caller_on_stack(mid) is False
    assert lt._legacy_caller_on_stack(None) is False


def test_installed_shim_preserves_explicit_kwarg(monkeypatch):
    lt = _load()
    import torch

    seen = {}

    def recorder(*args, **kwargs):
        seen.update(kwargs)
        return "ok"

    lt.install_legacy_torch_load_shim()
    monkeypatch.setattr(lt, "_original_load", recorder)
    # Called from this test module (not a legacy package): kwargs untouched.
    assert torch.load("whatever.pt", weights_only=True) == "ok"
    assert seen == {"weights_only": True}


def test_install_is_idempotent():
    lt = _load()
    import torch

    lt.install_legacy_torch_load_shim()
    first = torch.load
    lt.install_legacy_torch_load_shim()
    assert torch.load is first
