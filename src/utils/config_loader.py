from pathlib import Path
import os
from typing import Optional
import yaml

def _project_root() -> Path:
    """Find the project root by looking for common project markers."""
    current = Path(__file__).resolve().parent

    # Go up until we find a marker of the project root
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent

    # Fallback to going up 3 levels from this file (src/utils/config_loader.py -> project root)
    return Path(__file__).resolve().parents[2]

def load_config(config_file: Optional[str] = None) -> dict:
    env_path = os.getenv("CONFIG_PATH")
    # If caller didn't provide a path, prefer environment variable, then
    # fall back to the package `config/config.yaml` file.
    if config_file is None:
        config_file = env_path or str(_project_root()/ "src" / "config" / "logging.yaml")

    path = Path(config_file)
    if not path.is_absolute():
        path = _project_root() / path
        
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file) or {}