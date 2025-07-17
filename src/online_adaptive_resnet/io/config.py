"""Configuration loading utilities."""

import json
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, "r") as f:
        return json.load(f)