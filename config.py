import json
import os

class Config:
    """
    Configuration loader for the FaceOff project.
    Loads configuration from a JSON file.
    """
    def __init__(self, config_path: str = "config.json") -> None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.validate_config()

    def validate_config(self) -> None:
        """
        Validate the configuration file to ensure required keys are present.
        """
        # No required keys currently - config is optional
        pass

    def get(self, key: str, default=None):
        """
        Retrieve a configuration value with a default fallback.
        """
        return self.config.get(key, default)
