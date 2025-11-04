import logging
from realesrgan import RealESRGANer
from pathlib import Path
import torch
import os
from config import Config  # Import the Config class

def test_realesrgan_model():
    model_path = "models/realesrgan/weights/RealESRGAN_x4plus.pth"
    config = Config()
    model_path = config.get("real_esrgan_weight_path", model_path)  # Use centralized path

    # Check if the model path exists
    if not Path(model_path).exists():
        logging.error(f"Model path does not exist: {model_path}")
        return

    # Check if the model weights file is not empty
    if Path(model_path).stat().st_size == 0:
        logging.error(f"Model weights file is empty: {model_path}")
        return

    # Check if CUDA is available, otherwise fall back to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logging.warning("CUDA is not available. Falling back to CPU.")

    try:
        logging.info(f"Testing Real-ESRGAN model loading from: {model_path}")

        # Initialize the Real-ESRGAN model with additional options
        try:
            model = RealESRGANer(
                scale=4,
                model_path=model_path,
                dni_weight=None,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=True,
                device=device,
            )
            if model is None:
                logging.error("Failed to initialize RealESRGANer.")
            else:
                logging.info("RealESRGANer initialized successfully.")
        except FileNotFoundError as fnf_error:
            logging.error(f"File not found error: {fnf_error}")
        except torch.cuda.CudaError as cuda_error:
            logging.error(f"CUDA error: {cuda_error}")
        except Exception as e:
            logging.error(f"Failed to load Real-ESRGAN model: {e}")
    except Exception as e:
        logging.error(f"An error occurred in the Real-ESRGAN testing: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_realesrgan_model()
