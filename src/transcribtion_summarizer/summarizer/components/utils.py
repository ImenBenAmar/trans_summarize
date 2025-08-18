import logging
import os
import yaml
from pathlib import Path

def setup_logger(name: str, log_file: str = "app.log") -> logging.Logger:
    """Set up a logger with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    return logger

def load_config(config_path: str = "../../config.yaml") -> dict:
    """Load configuration from a YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config or {}
    except FileNotFoundError:
        logger = setup_logger(__name__)
        logger.warning(f"Config file {config_path} not found. Using default settings.")
        return {}

def ensure_dir(directory: Path) -> None:
    """Ensure a directory exists, creating it if necessary."""
    directory.mkdir(parents=True, exist_ok=True)