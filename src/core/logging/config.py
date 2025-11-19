"""
Logger configuration management.

This module handles all configuration-related logic for the logging system.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

class LoggingConfig:
    """
    Configuration class for logging settings.
    """
    def __init__(
        self,
        logs_dir: Optional[Union[Path, str]] = None,
        log_file_name: Optional[str] = None,
        log_level: int = logging.INFO,
        file_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        console_format: str = '%(levelname)s - %(message)s'
        ) -> None:
        """
        Initialize the logging configuration.
        """
        self._logs_dir = self._resolve_logs_dir(logs_dir)
        self._log_file_name = log_file_name or self._generate_log_filename()
        self._log_level = log_level
        self._file_format = file_format
        self._console_format = console_format
        
    @property
    def logs_dir(self) -> Path:
        """Get the directory where logs are stored."""
        return self._logs_dir
    
    @property
    def log_file_name(self) -> str:
        """Get the log file name."""
        return str(self._logs_dir / self._log_file_name)
    
    @property
    def log_level(self) -> int:
        """Get the logging level."""
        return self._log_level
    
    @property
    def file_format(self) -> str:
        """Get the file log format."""
        return self._file_format
    
    @property
    def console_format(self) -> str:
        """Get the console log format."""
        return self._console_format
    
    @staticmethod
    def _resolve_logs_dir(logs_dir: Optional[Union[Path, str]]) -> Path:
        """Resolve the logs directory, defaulting to repository root 'logs/'."""
        if logs_dir is not None:
            return Path(logs_dir)
        
        default_root = Path(__file__).resolve().parents[3]
        return default_root / "logs"
    
    @staticmethod
    def _generate_log_filename() -> str:
        """Generate a default log file name based on the current timestamp."""
        timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
        return f"app_{timestamp}.log"
    
    def ensure_logs_dir_exists(self) -> None:
        """Ensure that the logs directory exists."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)