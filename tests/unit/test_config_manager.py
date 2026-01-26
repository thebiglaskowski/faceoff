"""
Unit tests for utils/config_manager.py

Tests:
- Config singleton pattern
- Config loading from YAML
- Default value fallbacks
- Property accessors
- get_model_options helper
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestConfigSingleton:
    """Tests for Config singleton pattern."""

    def test_singleton_returns_same_instance(self, reset_config):
        """Should return the same instance on multiple calls."""
        from utils.config_manager import Config

        config1 = Config()
        config2 = Config()

        assert config1 is config2

    def test_singleton_reset(self, reset_config):
        """reset_config fixture should reset singleton."""
        from utils.config_manager import Config

        config1 = Config()
        Config._instance = None

        config2 = Config()

        # After reset, should be different instances
        # (though they'll have same values if loading same file)
        assert config1 is not config2


class TestConfigLoading:
    """Tests for configuration loading."""

    def test_load_from_yaml(self, temp_config, reset_config):
        """Should load config from YAML file."""
        from utils.config_manager import Config

        config = Config()

        # Values from temp_config fixture
        assert config.max_file_size_mb == 100
        assert config.batch_size == 2

    def test_load_missing_file(self, reset_config, tmp_path):
        """Should use defaults when config file missing."""
        from utils.config_manager import Config

        # Point to non-existent file
        Config._config_path = tmp_path / "nonexistent.yaml"
        config = Config()

        # Should use defaults
        assert config.max_file_size_mb == 500  # Default value
        assert config.batch_size == 4  # Default value

    def test_load_invalid_yaml(self, reset_config, tmp_path):
        """Should use defaults when YAML is invalid."""
        from utils.config_manager import Config

        # Create invalid YAML file
        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text("invalid: yaml: content: [")

        Config._config_path = bad_config
        config = Config()

        # Should fall back to defaults
        assert config.max_file_size_mb == 500

    def test_reload_config(self, temp_config, reset_config):
        """Should reload config when reload() is called."""
        from utils.config_manager import Config

        config = Config()
        original_value = config.max_file_size_mb

        # Modify config file
        new_content = temp_config.read_text().replace(
            "max_file_size_mb: 100",
            "max_file_size_mb: 200"
        )
        temp_config.write_text(new_content)

        config.reload()

        assert config.max_file_size_mb == 200


class TestConfigGet:
    """Tests for Config.get() method."""

    def test_get_single_key(self, temp_config, reset_config):
        """Should get value with single key."""
        from utils.config_manager import Config

        config = Config()
        result = config.get('limits')

        assert isinstance(result, dict)
        assert 'max_file_size_mb' in result

    def test_get_nested_keys(self, temp_config, reset_config):
        """Should get value with nested keys."""
        from utils.config_manager import Config

        config = Config()
        result = config.get('gpu', 'batch_size')

        assert result == 2  # From temp_config fixture

    def test_get_with_default(self, temp_config, reset_config):
        """Should return default when key not found."""
        from utils.config_manager import Config

        config = Config()
        result = config.get('nonexistent', 'key', default='fallback')

        assert result == 'fallback'

    def test_get_returns_none_without_default(self, temp_config, reset_config):
        """Should return None when key not found and no default."""
        from utils.config_manager import Config

        config = Config()
        result = config.get('nonexistent', 'key')

        assert result is None

    def test_get_deep_nested(self, temp_config, reset_config):
        """Should handle deeply nested keys."""
        from utils.config_manager import Config

        config = Config()
        result = config.get('enhancement', 'defaults', 'tile_size')

        assert result == 128  # From temp_config fixture


class TestConfigProperties:
    """Tests for Config property accessors."""

    def test_limits_properties(self, temp_config, reset_config):
        """Should access limits properties correctly."""
        from utils.config_manager import Config

        config = Config()

        assert config.max_file_size_mb == 100
        assert config.max_video_duration_sec == 60
        assert config.max_image_pixels == 4194304
        assert config.max_gif_frames == 100

    def test_gpu_properties(self, temp_config, reset_config):
        """Should access GPU properties correctly."""
        from utils.config_manager import Config

        config = Config()

        assert config.batch_size == 2
        assert config.max_batch_size == 8
        assert config.workers_per_gpu == 2
        assert config.tensorrt_enabled is False

    def test_face_detection_properties(self, temp_config, reset_config):
        """Should access face detection properties correctly."""
        from utils.config_manager import Config

        config = Config()

        assert config.face_confidence_threshold == 0.5
        assert config.adaptive_detection_enabled is True
        assert config.detection_scale == 0.5
        assert config.min_detection_resolution == 320

    def test_enhancement_properties(self, temp_config, reset_config):
        """Should access enhancement properties correctly."""
        from utils.config_manager import Config

        config = Config()

        assert config.default_tile_size == 128
        assert config.default_outscale == 2
        assert config.default_use_fp32 is False

    def test_memory_properties(self, temp_config, reset_config):
        """Should access memory properties correctly."""
        from utils.config_manager import Config

        config = Config()

        assert config.auto_clear_cache is False
        assert config.clear_cache_threshold_mb == 512
        assert config.reduce_batch_on_oom is True
        assert config.min_batch_size == 1

    def test_ui_properties(self, temp_config, reset_config):
        """Should access UI properties correctly."""
        from utils.config_manager import Config

        config = Config()

        assert config.ui_server_name == "127.0.0.1"
        assert config.ui_server_port == 7861
        assert config.ui_share is False

    def test_property_aliases(self, temp_config, reset_config):
        """Should have working property aliases."""
        from utils.config_manager import Config

        config = Config()

        # Aliases for backwards compatibility
        assert config.server_name == config.ui_server_name
        assert config.server_port == config.ui_server_port
        assert config.share == config.ui_share
        assert config.theme == config.ui_theme


class TestConfigDefaults:
    """Tests for default value behavior."""

    def test_all_properties_have_defaults(self, reset_config, tmp_path):
        """All properties should work even with empty config."""
        from utils.config_manager import Config

        # Point to non-existent file
        Config._config_path = tmp_path / "empty.yaml"
        config = Config()

        # All these should return defaults without error
        assert config.max_file_size_mb == 500
        assert config.batch_size == 4
        assert config.face_confidence_threshold == 0.5
        assert config.default_tile_size == 256
        assert config.auto_clear_cache is True
        assert config.ui_server_port == 7860

    def test_partial_config_uses_defaults(self, reset_config, tmp_path):
        """Should use defaults for missing sections."""
        from utils.config_manager import Config

        # Create minimal config
        partial_config = tmp_path / "partial.yaml"
        partial_config.write_text("""
limits:
  max_file_size_mb: 250
""")

        Config._config_path = partial_config
        config = Config()

        # Specified value
        assert config.max_file_size_mb == 250

        # Default values for unspecified sections
        assert config.batch_size == 4
        assert config.face_confidence_threshold == 0.5


class TestGetModelOptions:
    """Tests for get_model_options helper function."""

    def test_returns_dict(self, temp_config, reset_config):
        """Should return a dictionary."""
        from utils.config_manager import get_model_options

        result = get_model_options()
        assert isinstance(result, dict)

    def test_model_options_structure(self, reset_config):
        """Should return correctly structured model options."""
        from utils.config_manager import get_model_options

        # Test that get_model_options returns properly structured data
        result = get_model_options()

        # Should have at least one model
        assert len(result) > 0

        # Check structure of first model
        first_key = list(result.keys())[0]
        model_data = result[first_key]

        assert 'model_name' in model_data
        assert 'supports_denoise' in model_data
        assert 'description' in model_data
        assert isinstance(model_data['supports_denoise'], bool)

    def test_fallback_models_when_empty(self, reset_config, tmp_path):
        """Should return fallback models when config has none."""
        from utils.config_manager import Config, get_model_options

        # Empty config
        Config._config_path = tmp_path / "empty.yaml"
        Config._instance = None

        result = get_model_options()

        # Should have fallback models
        assert len(result) > 0
        assert any("RealESRGAN" in key for key in result.keys())


class TestConfigEdgeCases:
    """Tests for edge cases and error handling."""

    def test_null_values_in_yaml(self, reset_config, tmp_path):
        """Should handle null values in YAML."""
        from utils.config_manager import Config

        config_content = """
limits:
  max_file_size_mb: null
gpu:
  batch_size: 8
"""
        config_path = tmp_path / "null.yaml"
        config_path.write_text(config_content)

        Config._config_path = config_path
        config = Config()

        # Should use default for null value
        assert config.max_file_size_mb == 500
        # Non-null should work
        assert config.batch_size == 8

    def test_wrong_type_values(self, reset_config, tmp_path):
        """Should handle wrong type values gracefully."""
        from utils.config_manager import Config

        config_content = """
limits:
  max_file_size_mb: "not a number"
gpu:
  batch_size: [1, 2, 3]
"""
        config_path = tmp_path / "wrong.yaml"
        config_path.write_text(config_content)

        Config._config_path = config_path
        config = Config()

        # Should return the value as-is (caller handles type checking)
        result = config.get('limits', 'max_file_size_mb')
        assert result == "not a number"

    def test_empty_yaml_file(self, reset_config, tmp_path):
        """Should handle empty YAML file."""
        from utils.config_manager import Config

        empty_config = tmp_path / "empty.yaml"
        empty_config.write_text("")

        Config._config_path = empty_config
        config = Config()

        # Should use all defaults
        assert config.max_file_size_mb == 500
        assert config.batch_size == 4
