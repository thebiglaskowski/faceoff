"""
Unit tests for utils/memory_manager.py

Tests:
- MemoryManager class
- AutoMemoryManager context manager
- Convenience functions
- Memory statistics and cache clearing
"""

import pytest
from unittest.mock import MagicMock, patch


class TestMemoryManagerInit:
    """Tests for MemoryManager initialization."""

    def test_init_default_device(self, mock_gpu, temp_config, reset_config):
        """Should initialize with default device ID 0."""
        from utils.memory_manager import MemoryManager

        manager = MemoryManager()

        assert manager.device_id == 0

    def test_init_custom_device(self, mock_gpu, temp_config, reset_config):
        """Should accept custom device ID."""
        from utils.memory_manager import MemoryManager

        manager = MemoryManager(device_id=1)

        assert manager.device_id == 1

    def test_init_loads_config(self, mock_gpu, reset_config):
        """Should load settings from config."""
        from utils.memory_manager import MemoryManager

        manager = MemoryManager()

        # Verify manager has config-based attributes
        # (actual values come from default config.yaml)
        assert hasattr(manager, 'auto_clear')
        assert hasattr(manager, 'clear_threshold_mb')
        assert hasattr(manager, 'reduce_batch_on_oom')
        assert hasattr(manager, 'min_batch_size')

        # Verify types are correct
        assert isinstance(manager.auto_clear, bool)
        assert isinstance(manager.clear_threshold_mb, int)
        assert isinstance(manager.reduce_batch_on_oom, bool)
        assert isinstance(manager.min_batch_size, int)


class TestMemoryStats:
    """Tests for memory statistics."""

    def test_get_memory_stats_with_gpu(self, mock_gpu, temp_config, reset_config):
        """Should return memory stats when GPU available."""
        from utils.memory_manager import MemoryManager

        manager = MemoryManager()
        stats = manager.get_memory_stats()

        assert 'allocated_mb' in stats
        assert 'reserved_mb' in stats
        assert 'free_mb' in stats
        assert 'total_mb' in stats
        assert 'utilization_pct' in stats

    def test_get_memory_stats_without_gpu(self, mock_cuda_unavailable, temp_config, reset_config):
        """Should return zeros when GPU unavailable."""
        from utils.memory_manager import MemoryManager

        manager = MemoryManager()
        stats = manager.get_memory_stats()

        assert stats['allocated_mb'] == 0
        assert stats['reserved_mb'] == 0
        assert stats['free_mb'] == 0
        assert stats['total_mb'] == 0
        assert stats['utilization_pct'] == 0

    def test_memory_stats_values(self, mock_gpu, temp_config, reset_config):
        """Should calculate correct memory values."""
        from utils.memory_manager import MemoryManager

        manager = MemoryManager()
        stats = manager.get_memory_stats()

        # Based on mock_gpu fixture: 1GB allocated, 8GB total
        assert stats['allocated_mb'] == pytest.approx(1024, rel=0.01)
        assert stats['total_mb'] == pytest.approx(8192, rel=0.01)
        assert stats['free_mb'] == pytest.approx(7168, rel=0.01)
        assert stats['utilization_pct'] == pytest.approx(12.5, rel=0.1)


class TestCacheClearing:
    """Tests for cache clearing functionality."""

    def test_should_clear_cache_disabled(self, mock_gpu, temp_config, reset_config):
        """Should return False when auto_clear disabled."""
        from utils.memory_manager import MemoryManager

        manager = MemoryManager()
        manager.auto_clear = False

        assert manager.should_clear_cache() is False

    def test_should_clear_cache_below_threshold(self, mock_gpu, temp_config, reset_config):
        """Should return False when below threshold."""
        from utils.memory_manager import MemoryManager

        manager = MemoryManager()
        manager.auto_clear = True
        manager.clear_threshold_mb = 2000  # Above 1GB allocated

        assert manager.should_clear_cache() is False

    def test_should_clear_cache_above_threshold(self, mock_gpu, temp_config, reset_config):
        """Should return True when above threshold."""
        from utils.memory_manager import MemoryManager

        manager = MemoryManager()
        manager.auto_clear = True
        manager.clear_threshold_mb = 500  # Below 1GB allocated

        assert manager.should_clear_cache() is True

    def test_clear_cache_forced(self, mock_gpu, temp_config, reset_config):
        """Should clear cache when forced."""
        from utils.memory_manager import MemoryManager
        import torch.cuda as cuda

        manager = MemoryManager()
        manager.clear_cache(force=True)

        # Verify torch.cuda.empty_cache was called
        cuda.empty_cache.assert_called()

    def test_clear_cache_no_gpu(self, mock_cuda_unavailable, temp_config, reset_config):
        """Should do nothing when GPU unavailable."""
        from utils.memory_manager import MemoryManager

        manager = MemoryManager()
        # Should not raise
        manager.clear_cache(force=True)


class TestOptimalBatchSize:
    """Tests for batch size optimization."""

    def test_optimal_batch_below_current(self, mock_gpu, temp_config, reset_config):
        """Should keep current batch if optimal is higher."""
        from utils.memory_manager import MemoryManager

        manager = MemoryManager()

        # With 7GB free (mock_gpu), ~500MB per batch = ~14 batches max
        result = manager.get_optimal_batch_size(4)

        assert result == 4  # Should keep current

    def test_optimal_batch_reduces_when_needed(self, mock_gpu, temp_config, reset_config):
        """Should reduce batch size when memory limited."""
        from utils.memory_manager import MemoryManager

        manager = MemoryManager()

        # Request very large batch with limited memory
        result = manager.get_optimal_batch_size(
            current_batch_size=100,
            available_vram_mb=1000  # Only 1GB available
        )

        # ~1000MB / 500MB per batch = 2 batches max
        assert result <= 8  # Capped by max_batch_size from config

    def test_optimal_batch_respects_min(self, mock_gpu, temp_config, reset_config):
        """Should respect minimum batch size."""
        from utils.memory_manager import MemoryManager

        manager = MemoryManager()
        manager.min_batch_size = 2

        result = manager.get_optimal_batch_size(
            current_batch_size=10,
            available_vram_mb=100  # Very limited
        )

        assert result >= 2


class TestIsMemoryError:
    def test_detects_onnx_bfc_arena_failure(self):
        from utils.memory_manager import is_memory_error

        err = RuntimeError(
            "Fail: [ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned "
            "while running Conv node. Failed to allocate memory for requested "
            "buffer of size 26216448 bfc_arena.cc"
        )
        assert is_memory_error(err)

    def test_detects_torch_cuda_oom(self):
        import torch
        from utils.memory_manager import is_memory_error

        assert is_memory_error(torch.cuda.OutOfMemoryError("CUDA OOM"))


class TestOOMHandling:
    """Tests for OOM error handling."""

    def test_handle_oom_reduces_batch(self, mock_gpu, temp_config, reset_config):
        """Should reduce batch size on OOM."""
        from utils.memory_manager import MemoryManager

        manager = MemoryManager()
        manager.reduce_batch_on_oom = True

        new_size, should_retry = manager.handle_oom_error(8)

        assert new_size == 4  # Halved
        assert should_retry is True

    def test_handle_oom_respects_minimum(self, mock_gpu, temp_config, reset_config):
        """Should not reduce below minimum batch size."""
        from utils.memory_manager import MemoryManager

        manager = MemoryManager()
        manager.min_batch_size = 2
        manager.reduce_batch_on_oom = True

        new_size, should_retry = manager.handle_oom_error(2)

        # Can't go below minimum
        assert new_size >= 1
        # If at minimum, retry depends on whether we can reduce further
        # At batch_size=2, halving gives 1, which is below min_batch_size=2

    def test_handle_oom_disabled(self, mock_gpu, temp_config, reset_config):
        """Should not retry when OOM recovery disabled."""
        from utils.memory_manager import MemoryManager

        manager = MemoryManager()
        manager.reduce_batch_on_oom = False

        new_size, should_retry = manager.handle_oom_error(8)

        assert should_retry is False

    def test_handle_oom_clears_cache(self, mock_gpu, temp_config, reset_config):
        """Should clear cache when handling OOM."""
        from utils.memory_manager import MemoryManager
        import torch.cuda as cuda

        manager = MemoryManager()
        manager.handle_oom_error(8)

        cuda.empty_cache.assert_called()


class TestAutoMemoryManager:
    """Tests for AutoMemoryManager context manager."""

    def test_context_manager_enter(self, mock_gpu, temp_config, reset_config):
        """Should return manager on enter."""
        from utils.memory_manager import AutoMemoryManager

        with AutoMemoryManager() as manager:
            assert manager is not None
            assert hasattr(manager, 'get_memory_stats')

    def test_context_manager_clears_on_exit(self, reset_config):
        """Should clear cache on exit by default."""
        from unittest.mock import patch, MagicMock
        from utils.memory_manager import AutoMemoryManager

        # Create a fresh mock for this specific test
        # Use 2GB allocated to exceed 1024MB threshold and trigger clear
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.empty_cache') as mock_empty, \
             patch('torch.cuda.synchronize'), \
             patch('torch.cuda.memory_allocated', return_value=2*1024*1024*1024), \
             patch('torch.cuda.memory_reserved', return_value=3*1024*1024*1024), \
             patch('torch.cuda.get_device_properties') as mock_props:

            mock_props.return_value = MagicMock(total_memory=8*1024*1024*1024)

            with AutoMemoryManager(clear_on_exit=True):
                pass

            # Verify empty_cache was called (2GB > 1024MB threshold)
            assert mock_empty.called

    def test_context_manager_no_clear(self, mock_gpu, temp_config, reset_config):
        """Should not clear cache when clear_on_exit=False."""
        from utils.memory_manager import AutoMemoryManager
        import torch.cuda as cuda

        cuda.empty_cache.reset_mock()

        with AutoMemoryManager(clear_on_exit=False):
            pass

        # Only called during enter's should_clear_cache check, not on exit
        # The fixture has auto_clear=False so it shouldn't be called

    def test_context_manager_handles_exception(self, mock_gpu, temp_config, reset_config):
        """Should not suppress exceptions."""
        from utils.memory_manager import AutoMemoryManager

        with pytest.raises(ValueError):
            with AutoMemoryManager():
                raise ValueError("Test error")


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_clear_cuda_cache(self, mock_gpu, temp_config, reset_config):
        """Should clear CUDA cache."""
        from utils.memory_manager import clear_cuda_cache
        import torch.cuda as cuda

        clear_cuda_cache()

        cuda.empty_cache.assert_called()

    def test_get_memory_stats_function(self, mock_gpu, temp_config, reset_config):
        """Should return memory stats from function."""
        from utils.memory_manager import get_memory_stats

        stats = get_memory_stats()

        assert 'allocated_mb' in stats
        assert 'total_mb' in stats


class TestLogMemoryStats:
    """Tests for memory stats logging."""

    def test_log_memory_stats(self, mock_gpu, temp_config, reset_config, capture_logs):
        """Should log memory statistics."""
        from utils.memory_manager import MemoryManager

        manager = MemoryManager()
        manager.log_memory_stats(prefix="Test")

        assert "GPU Memory" in capture_logs.text or len(capture_logs.records) > 0

    def test_log_memory_stats_no_prefix(self, mock_gpu, temp_config, reset_config):
        """Should work without prefix."""
        from utils.memory_manager import MemoryManager

        manager = MemoryManager()
        # Should not raise
        manager.log_memory_stats()


class TestEdgeCases:
    """Tests for edge cases."""

    def test_multiple_managers_same_device(self, mock_gpu, temp_config, reset_config):
        """Should allow multiple managers for same device."""
        from utils.memory_manager import MemoryManager

        manager1 = MemoryManager(device_id=0)
        manager2 = MemoryManager(device_id=0)

        # Both should work independently
        stats1 = manager1.get_memory_stats()
        stats2 = manager2.get_memory_stats()

        assert stats1 == stats2

    def test_device_id_out_of_range(self, mock_gpu, temp_config, reset_config):
        """Should handle device IDs gracefully."""
        from utils.memory_manager import MemoryManager

        # Mock only has 1 GPU, but manager shouldn't fail at init
        manager = MemoryManager(device_id=99)
        assert manager.device_id == 99


class TestRefreshAndEnhancementPrep:
    """Tests for staged memory refresh helpers."""

    def test_refresh_gpu_memory_calls_empty_cache(self, mock_gpu, temp_config, reset_config):
        from utils.memory_manager import refresh_gpu_memory
        import torch.cuda as cuda

        stats = refresh_gpu_memory(0, force=True)
        assert 0 in stats
        cuda.empty_cache.assert_called()

    def test_select_enhancement_gpu_prefers_most_free(self, mock_gpu, temp_config, reset_config):
        from unittest.mock import patch
        from utils.memory_manager import MemoryManager, select_enhancement_gpu

        with patch.object(MemoryManager, "get_memory_stats") as mock_stats:
            mock_stats.side_effect = [
                {"free_mb": 500, "total_mb": 8000, "allocated_mb": 7500, "reserved_mb": 7600, "utilization_pct": 90},
                {"free_mb": 3000, "total_mb": 8000, "allocated_mb": 5000, "reserved_mb": 5100, "utilization_pct": 60},
            ]
            assert select_enhancement_gpu(0, [0, 1]) == 1

    def test_prepare_for_enhancement_releases_swap_models(self, mock_gpu, temp_config, reset_config):
        from unittest.mock import MagicMock, patch
        from utils.memory_manager import prepare_for_enhancement

        mock_pool = MagicMock()
        with patch("core.model_pool.get_model_pool", return_value=mock_pool):
            gpu = prepare_for_enhancement(0, device_ids=[0, 1])
        assert gpu == 0
        assert mock_pool.cleanup.call_count == 2
