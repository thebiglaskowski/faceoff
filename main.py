"""
FaceOff - AI Face Swapper
Main entry point for the application.
"""
import logging
from ui.app import create_app
from utils.config_manager import config
from utils.model_cache import preload_models, get_cache_info

logger = logging.getLogger("FaceOff")

if __name__ == "__main__":
    # Display cache info at startup
    cache_info = get_cache_info()
    logger.info("Model cache: %d engine(s) cached (%.2f MB total)", 
               cache_info['num_files'], cache_info['total_size_mb'])
    
    # Optional: Preload models in background to reduce first-run delay
    if config.preload_on_startup:
        logger.info("Model preloading enabled - compiling TensorRT engines...")
        preload_models(device_id=0)
    
    # Launch Gradio app
    demo = create_app()
    demo.launch(
        server_name=config.server_name,
        server_port=config.server_port,
        share=config.share
    )
