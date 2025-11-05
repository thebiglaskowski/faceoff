"""
GPU management and monitoring utilities.
"""
import logging
import subprocess
import torch
from typing import List, Optional

logger = logging.getLogger("FaceOff")


class GPUManager:
    """Manages GPU detection, selection, and monitoring."""
    
    @staticmethod
    def is_available() -> bool:
        """Check if CUDA/GPU is available."""
        return torch.cuda.is_available()
    
    @staticmethod
    def get_device_count() -> int:
        """Get number of available GPUs."""
        return torch.cuda.device_count() if GPUManager.is_available() else 0
    
    @staticmethod
    def get_memory_info() -> List[str]:
        """
        Get GPU memory usage information for all available GPUs.
        Returns list of formatted strings, one per GPU.
        """
        if not GPUManager.is_available():
            return ["🖥️ GPU: Not available (using CPU)"]
        
        try:
            gpu_count = GPUManager.get_device_count()
            gpu_info_list = []
            
            # Try to get info for all GPUs using nvidia-smi
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free', 
                     '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        gpu_idx, name, total_mb, used_mb, free_mb = line.split(',')
                        gpu_idx = gpu_idx.strip()
                        name = name.strip()
                        total_gb = float(total_mb) / 1024
                        used_gb = float(used_mb) / 1024
                        free_gb = float(free_mb) / 1024
                        usage_pct = (used_gb / total_gb * 100) if total_gb > 0 else 0
                        
                        # Get PyTorch allocation info
                        allocated_gb = torch.cuda.memory_allocated(int(gpu_idx)) / (1024**3)
                        
                        gpu_info_list.append(
                            f"🎮 GPU {gpu_idx}: {name}\n"
                            f"📊 VRAM: {used_gb:.1f}GB / {total_gb:.1f}GB ({usage_pct:.1f}%)\n"
                            f"💚 Free: {free_gb:.1f}GB\n"
                            f"🔧 PyTorch: {allocated_gb:.2f}GB"
                        )
                    return gpu_info_list
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                pass
            
            # Fallback to PyTorch info only
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                allocated_gb = torch.cuda.memory_allocated(i) / (1024**3)
                reserved_gb = torch.cuda.memory_reserved(i) / (1024**3)
                gpu_info_list.append(
                    f"🎮 GPU {i}: {gpu_name}\n"
                    f"🔧 PyTorch: {allocated_gb:.2f}GB allocated\n"
                    f"📦 Reserved: {reserved_gb:.2f}GB"
                )
            
            return gpu_info_list
            
        except Exception as e:
            logger.error("Failed to get GPU info: %s", e)
            return ["⚠️ GPU info unavailable"]
    
    @staticmethod
    def get_available_gpus() -> List[str]:
        """Get list of available GPU options for dropdown."""
        if not GPUManager.is_available():
            return ["CPU Only"]
        
        gpu_count = GPUManager.get_device_count()
        options = []
        
        # Add multi-GPU option if more than one GPU
        if gpu_count > 1:
            options.append(f"All GPUs ({gpu_count} GPUs)")
        
        # Add individual GPU options
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            options.append(f"GPU {i}: {gpu_name}")
        
        return options
    
    @staticmethod
    def parse_gpu_selection(selection: Optional[str]) -> List[int]:
        """
        Parse GPU selection string to list of device IDs.
        
        Args:
            selection: GPU selection string (e.g., "All GPUs (2 GPUs)", "GPU 0: RTX 3060")
            
        Returns:
            List of GPU device IDs to use
        """
        if not selection or selection == "CPU Only":
            return []
        
        if "All GPUs" in selection:
            return list(range(GPUManager.get_device_count()))
        
        # Extract GPU number from "GPU X: Name" format
        if "GPU" in selection:
            try:
                gpu_id = int(selection.split(":")[0].replace("GPU", "").strip())
                return [gpu_id]
            except (ValueError, IndexError):
                logger.warning("Failed to parse GPU selection: %s", selection)
                return [0]  # Default to GPU 0
        
        return [0]  # Default to GPU 0
