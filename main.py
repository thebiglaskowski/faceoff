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
    
    # Try the configured port first, then auto-find if busy
    try:
        demo.launch(
            server_name=config.server_name,
            server_port=config.server_port,
            share=config.share
        )
    except OSError as e:
        if "Cannot find empty port" in str(e):
            logger.info(f"Port {config.server_port} is busy, trying alternative ports...")
            # Try ports 7861-7870
            for port in range(7861, 7871):
                try:
                    demo.launch(
                        server_name=config.server_name,
                        server_port=port,
                        share=config.share
                    )
                    logger.info(f"✅ Successfully started on port {port}")
                    break
                except OSError:
                    continue
            else:
                logger.error("Could not find any available port in range 7861-7870")
                raise
        else:
            raise
