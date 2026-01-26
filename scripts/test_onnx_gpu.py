import onnxruntime as ort

print("Available execution providers:", ort.get_available_providers())
try:
    sess = ort.InferenceSession("models/buffalo_l/1k3d68.onnx", providers=["CUDAExecutionProvider"])
    print("CUDAExecutionProvider successfully initialized.")
except Exception as e:
    print("CUDAExecutionProvider FAILED:", e)
