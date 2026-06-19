"""Unit tests for startup model preloading."""

import numpy as np
from unittest.mock import MagicMock, patch


def _mock_config(**overrides):
    cfg = MagicMock()
    cfg.preload_on_startup = overrides.get("preload_on_startup", True)
    cfg.tensorrt_enabled = overrides.get("tensorrt_enabled", True)
    cfg.face_analysis_det_size = overrides.get("face_analysis_det_size", [640, 640])
    cfg.inswapper_model_path = overrides.get("inswapper_model_path", "swap.onnx")
    return cfg


def test_preload_skipped_when_disabled():
    with patch("processing.model_preloader.config", _mock_config(preload_on_startup=False)), \
         patch("processing.model_preloader.get_model_pool") as mock_pool:
        from processing.model_preloader import preload_models

        preload_models()
        mock_pool.assert_not_called()


def test_preload_warms_all_gpus_sequentially():
    instances = {0: MagicMock(), 1: MagicMock()}
    mock_pool = MagicMock()
    mock_pool.get_instance.side_effect = lambda device_id, **kwargs: instances[device_id]

    with patch("processing.model_preloader.config", _mock_config()), \
         patch("processing.model_preloader.is_tensorrt_runtime_available", return_value=True), \
         patch("processing.model_preloader.get_model_pool", return_value=mock_pool), \
         patch("processing.model_preloader.GPUManager.is_available", return_value=True), \
         patch("processing.model_preloader.GPUManager.get_device_count", return_value=2):
        from processing.model_preloader import preload_models

        preload_models()

    assert mock_pool.get_instance.call_count == 2
    mock_pool.get_instance.assert_any_call(0, use_tensorrt=True)
    mock_pool.get_instance.assert_any_call(1, use_tensorrt=False)
    instances[0].get_faces.assert_not_called()
    instances[1].get_faces.assert_called_once()


def test_preload_continues_after_single_gpu_failure():
    good_instance = MagicMock()
    mock_pool = MagicMock()
    mock_pool.get_instance.side_effect = [RuntimeError("gpu0 oom"), good_instance]

    with patch(
        "processing.model_preloader.config",
        _mock_config(tensorrt_enabled=False),
    ), patch("processing.model_preloader.get_model_pool", return_value=mock_pool):
        from processing.model_preloader import preload_models

        preload_models(device_ids=[0, 1])

    assert mock_pool.get_instance.call_count == 2
    good_instance.get_faces.assert_called_once()


def test_trt_warmup_images_cover_common_shapes():
    from core.model_pool import _trt_warmup_images

    images = _trt_warmup_images()
    assert len(images) == 3
    shapes = {(img.shape[0], img.shape[1]) for img in images}
    assert (640, 640) in shapes
    assert (1920, 1080) in shapes
    assert (1080, 1920) in shapes
    assert all(img.dtype == np.uint8 for img in images)


def test_trt_warmup_images_use_nonzero_pixels():
    from core.model_pool import _trt_warmup_images

    for image in _trt_warmup_images():
        assert image.max() > 0