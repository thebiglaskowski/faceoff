"""Unit tests for the Settings tab and comment-preserving config save."""

import shutil

import pytest
import yaml


@pytest.fixture()
def isolated_config(tmp_path):
    """Point the Config singleton at a scratch copy of config.yaml.

    The real path is derived from the module location, never from
    ``Config._config_path`` — other test modules rebind that attribute
    without restoring it. Teardown points the singleton back at the
    real file and reloads, healing any leaked path for later tests.
    """
    from pathlib import Path

    import utils.config_manager as config_module
    from utils.config_manager import Config, config

    real_path = Path(config_module.__file__).parent.parent / "config.yaml"
    scratch = tmp_path / "config.yaml"
    shutil.copy(real_path, scratch)
    Config._config_path = scratch
    config.reload()
    yield scratch
    Config._config_path = real_path
    config.reload()


def _doc(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_spec_covers_every_config_leaf():
    """The editor must expose all options (except system model catalogs)."""
    import gradio as gr  # noqa: F401  (ensures UI deps resolve in tests)

    from ui.components.settings_tab import SECTIONS, iter_editable_fields

    assert SECTIONS  # sections exist for grouping
    doc = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    skipped = {
        ("enhancement", "models"),
        ("hat_enhancement", "models"),
        ("swinir", "models"),
    }
    covered = {tuple(spec.path) for spec in iter_editable_fields()}
    assert len(covered) == len(list(iter_editable_fields())), "duplicate spec path"

    def _leaves(node, prefix=()):
        if isinstance(node, dict):
            for key, value in node.items():
                yield from _leaves(value, prefix + (key,))
        elif isinstance(node, list) and all(
            isinstance(item, (str, int, float, bool)) for item in node
        ):
            yield prefix
        else:
            yield prefix

    missing = [
        path
        for path in _leaves(doc)
        if not any(path[: len(skip)] == skip for skip in skipped)
        and path not in covered
    ]
    assert missing == []


def test_widget_count_matches_save_arity():
    import gradio as gr

    from ui.components.settings_tab import (
        create_settings_tab,
        total_inputs,
        wire_settings_tab,
    )

    with gr.Blocks():
        comps = create_settings_tab()
        assert len(comps["field_widgets"]) == total_inputs()
        wire_settings_tab(comps)  # must not raise


def test_surgery_preserves_comments(isolated_config):
    from utils.config_manager import config

    before = isolated_config.read_text(encoding="utf-8")
    config.write_values(scalars={("gpu", "batch_size"): 4})
    after = isolated_config.read_text(encoding="utf-8")
    assert "Frames per batch" in after  # trailing comment survived
    assert "Wave 3 phase 1" in after  # untouched section comments survived
    assert _doc(isolated_config)["gpu"]["batch_size"] == 4
    assert config.batch_size == 4  # in-memory singleton reloaded


def test_surgery_block_style_and_pair_and_list(isolated_config):
    from utils.config_manager import config

    config.write_values(
        scalars={("gpu", "multi_gpu_video_enabled"): False},
        pairs={("face_detection", "face_analysis_det_size"): (320, 320)},
        lists={("file_formats", "images"): [".png"]},
    )
    doc = _doc(isolated_config)
    assert doc["gpu"]["multi_gpu_video_enabled"] is False
    assert doc["face_detection"]["face_analysis_det_size"] == [320, 320]
    assert doc["file_formats"]["images"] == [".png"]
    assert "Enable multi-GPU" in isolated_config.read_text(encoding="utf-8")


def test_surgery_rejects_unknown_path_without_touching_file(isolated_config):
    from utils.config_manager import config

    before = isolated_config.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        config.write_values(scalars={("gpu", "nope_not_real"): 1})
    assert isolated_config.read_text(encoding="utf-8") == before


def test_surgery_rejects_type_mismatch_without_touching_file(isolated_config):
    from utils.config_manager import config

    before = isolated_config.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        config.write_values(scalars={("gpu", "batch_size"): "lots"})
    with pytest.raises(ValueError):
        config.write_values(scalars={("enhancement", "models"): True})
    assert isolated_config.read_text(encoding="utf-8") == before


def test_apply_round_trip(isolated_config):
    from ui.components.settings_tab import (
        apply_settings_values,
        collect_current_values,
        value_offset,
    )
    from utils.config_manager import config

    values = collect_current_values()
    values[value_offset(("gpu", "batch_size"))] = 4
    values[value_offset(("ui", "theme"))] = "neon"

    message = apply_settings_values(*values)
    assert message.startswith("✅ Saved 2 settings")
    assert "restart" not in message.lower()
    doc = _doc(isolated_config)
    assert doc["gpu"]["batch_size"] == 4
    assert doc["ui"]["theme"] == "neon"
    assert config.batch_size == 4


def test_apply_rejects_bad_value_without_saving(isolated_config):
    from ui.components.settings_tab import (
        apply_settings_values,
        collect_current_values,
        value_offset,
    )

    values = collect_current_values()
    values[value_offset(("gpu", "batch_size"))] = 99999  # outside 1-64

    before = isolated_config.read_text(encoding="utf-8")
    message = apply_settings_values(*values)
    assert message.startswith("❌")
    assert "nothing saved" in message
    assert isolated_config.read_text(encoding="utf-8") == before


def test_apply_flags_restart_keys(isolated_config):
    from ui.components.settings_tab import (
        apply_settings_values,
        collect_current_values,
        value_offset,
    )

    values = collect_current_values()
    values[value_offset(("ui", "server_port"))] = 7861
    message = apply_settings_values(*values)
    assert message.startswith("✅")
    assert "restart" in message.lower()


def test_discard_returns_current_values_plus_status(isolated_config):
    from ui.components.settings_tab import (
        discard_settings_values,
        total_inputs,
    )

    out = discard_settings_values()
    assert len(out) == total_inputs() + 1
    assert "Discarded" in out[-1]
