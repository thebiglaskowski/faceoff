"""Error handler GPU runtime classification tests."""


def test_epfail_maps_to_gpu_runtime_error():
    from utils.error_handler import ErrorHandler

    class EPFail(Exception):
        pass

    err = EPFail(
        "[ONNXRuntimeError] : 11 : EP_FAIL : CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH"
    )
    friendly = ErrorHandler.handle_error(err, {})
    assert friendly.title == "GPU Runtime Error"
    assert "cuDNN" in friendly.message or "CUDA" in friendly.message