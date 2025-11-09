"""
Preset management for FaceOff processing settings.
Allows users to save, load, and manage processing configurations.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger("FaceOff")


class PresetManager:
    """Manages processing presets for FaceOff."""
    
    def __init__(self, presets_dir: str = "presets"):
        """
        Initialize preset manager.
        
        Args:
            presets_dir: Directory to store preset files
        """
        self.presets_dir = Path(presets_dir)
        self.presets_dir.mkdir(exist_ok=True)
        logger.info("PresetManager initialized: %s", self.presets_dir)
    
    def save_preset(self, name: str, settings: Dict[str, Any], description: str = "") -> bool:
        """
        Save processing settings as a preset.
        
        Args:
            name: Preset name (will be sanitized for filename)
            settings: Dictionary of processing settings
            description: Optional description of the preset
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Sanitize preset name for filename
            safe_name = self._sanitize_filename(name)
            if not safe_name:
                logger.error("Invalid preset name: %s", name)
                return False
            
            preset_path = self.presets_dir / f"{safe_name}.json"
            
            # Build preset data
            preset_data = {
                "name": name,
                "description": description,
                "created": datetime.now().isoformat(),
                "settings": settings
            }
            
            # Save to file
            with open(preset_path, 'w', encoding='utf-8') as f:
                json.dump(preset_data, f, indent=2)
            
            logger.info("Preset saved: %s → %s", name, preset_path)
            return True
            
        except Exception as e:
            logger.error("Failed to save preset '%s': %s", name, e)
            return False
    
    def load_preset(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Load a preset by name.
        
        Args:
            name: Preset name
            
        Returns:
            Preset settings dict, or None if not found
        """
        try:
            safe_name = self._sanitize_filename(name)
            preset_path = self.presets_dir / f"{safe_name}.json"
            
            if not preset_path.exists():
                logger.warning("Preset not found: %s", name)
                return None
            
            with open(preset_path, 'r', encoding='utf-8') as f:
                preset_data = json.load(f)
            
            logger.info("Preset loaded: %s", name)
            return preset_data.get("settings", {})
            
        except Exception as e:
            logger.error("Failed to load preset '%s': %s", name, e)
            return None
    
    def list_presets(self) -> List[Dict[str, str]]:
        """
        List all available presets.
        
        Returns:
            List of preset info dicts with 'name', 'description', 'created'
        """
        presets = []
        
        try:
            for preset_file in sorted(self.presets_dir.glob("*.json")):
                try:
                    with open(preset_file, 'r', encoding='utf-8') as f:
                        preset_data = json.load(f)
                    
                    presets.append({
                        "name": preset_data.get("name", preset_file.stem),
                        "description": preset_data.get("description", ""),
                        "created": preset_data.get("created", "")
                    })
                except Exception as e:
                    logger.warning("Failed to read preset file %s: %s", preset_file, e)
                    
        except Exception as e:
            logger.error("Failed to list presets: %s", e)
        
        return presets
    
    def delete_preset(self, name: str) -> bool:
        """
        Delete a preset.
        
        Args:
            name: Preset name
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            safe_name = self._sanitize_filename(name)
            preset_path = self.presets_dir / f"{safe_name}.json"
            
            if not preset_path.exists():
                logger.warning("Preset not found for deletion: %s", name)
                return False
            
            preset_path.unlink()
            logger.info("Preset deleted: %s", name)
            return True
            
        except Exception as e:
            logger.error("Failed to delete preset '%s': %s", name, e)
            return False
    
    def preset_exists(self, name: str) -> bool:
        """
        Check if a preset exists.
        
        Args:
            name: Preset name
            
        Returns:
            True if preset exists
        """
        safe_name = self._sanitize_filename(name)
        preset_path = self.presets_dir / f"{safe_name}.json"
        return preset_path.exists()
    
    def get_preset_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get full preset information including settings.
        
        Args:
            name: Preset name
            
        Returns:
            Full preset data dict, or None if not found
        """
        try:
            safe_name = self._sanitize_filename(name)
            preset_path = self.presets_dir / f"{safe_name}.json"
            
            if not preset_path.exists():
                return None
            
            with open(preset_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error("Failed to get preset info '%s': %s", name, e)
            return None
    
    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """
        Sanitize preset name for use as filename.
        
        Args:
            name: Preset name
            
        Returns:
            Safe filename string
        """
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        safe_name = name.strip()
        
        for char in invalid_chars:
            safe_name = safe_name.replace(char, '_')
        
        # Collapse multiple underscores
        while '__' in safe_name:
            safe_name = safe_name.replace('__', '_')
        
        # Remove leading/trailing underscores
        safe_name = safe_name.strip('_')
        
        return safe_name


# Default presets
DEFAULT_PRESETS = {
    "High Quality": {
        "description": "Maximum quality, slow processing. Best for final outputs.",
        "settings": {
            "enhance": True,
            "tile_size": 512,
            "outscale": 4,
            "model_name": "RealESRGAN_x4plus",
            "denoise_strength": 0.3,
            "use_fp32": True,
            "pre_pad": 10,
            "restore_faces": True,
            "restoration_weight": 0.5,
            "face_confidence": 0.5,
            "adaptive_detection": True,
            "detection_scale": 1.5
        }
    },
    "Balanced": {
        "description": "Good quality with reasonable speed. Recommended default.",
        "settings": {
            "enhance": True,
            "tile_size": 256,
            "outscale": 4,
            "model_name": "RealESRGAN_x4plus",
            "denoise_strength": 0.5,
            "use_fp32": False,
            "pre_pad": 10,
            "restore_faces": True,
            "restoration_weight": 0.5,
            "face_confidence": 0.5,
            "adaptive_detection": True,
            "detection_scale": 1.0
        }
    },
    "Fast Preview": {
        "description": "Quick processing for testing. Lower quality.",
        "settings": {
            "enhance": True,
            "tile_size": 128,
            "outscale": 2,
            "model_name": "RealESRGAN_x4plus",
            "denoise_strength": 0.5,
            "use_fp32": False,
            "pre_pad": 0,
            "restore_faces": False,
            "restoration_weight": 0.0,
            "face_confidence": 0.4,
            "adaptive_detection": False,
            "detection_scale": 1.0
        }
    },
    "Anime Style": {
        "description": "Optimized for anime and cartoon content.",
        "settings": {
            "enhance": True,
            "tile_size": 256,
            "outscale": 4,
            "model_name": "RealESRGAN_x4plus_anime_6B",
            "denoise_strength": 0.5,
            "use_fp32": False,
            "pre_pad": 10,
            "restore_faces": False,
            "restoration_weight": 0.0,
            "face_confidence": 0.6,
            "adaptive_detection": True,
            "detection_scale": 1.0
        }
    }
}


def initialize_default_presets(preset_manager: PresetManager) -> None:
    """
    Initialize default presets if they don't exist.
    
    Args:
        preset_manager: PresetManager instance
    """
    for name, data in DEFAULT_PRESETS.items():
        if not preset_manager.preset_exists(name):
            preset_manager.save_preset(
                name=name,
                settings=data["settings"],
                description=data["description"]
            )
            logger.info("Initialized default preset: %s", name)
