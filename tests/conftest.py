"""Pytest configuration for Cerebrus AI."""
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file)

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Ensure the src directory exists
if not src_path.exists():
    raise RuntimeError(f"Source directory not found: {src_path}")