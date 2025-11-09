"""
GPU utilities for UI.
Helper functions for GPU selection and status display.
"""
import logging
from typing import List

from core.gpu_manager import GPUManager

logger = logging.getLogger("FaceOff")


def get_gpu_options() -> List[str]:
    """
    Get GPU dropdown options for UI.
    
    Returns:
        List of GPU options for dropdown
    """
    return GPUManager.get_available_gpus()


def get_gpu_status() -> List[str]:
    """
    Get GPU memory status for all GPUs.
    
    Returns:
        List of GPU status strings
    """
    return GPUManager.get_memory_info()


def refresh_gpu_info() -> List[str]:
    """
    Refresh GPU information display.
    
    Returns:
        List of GPU info strings (at least one entry)
    """
    gpu_info_list = get_gpu_status()
    
    # Ensure we have at least one entry
    if not gpu_info_list:
        gpu_info_list = ["No GPU info available"]
    
    return gpu_info_list
