"""
Logger configuration management using YAML configuration with backwards compatibility.

This module handles all configuration-related logic for the logging system.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
from src.utils.config_loader import load_config

class LoggingConfig:
    """
    Configuration class for logging settings - YAML-based with backwards compatibility.
    """
    def __init__(
        self,
        config_file: Optional[str] = None,
        # Backwards compatibility parameters
        logs_dir: Optional[Union[Path, str]] = None,
        log_file_name: Optional[str] = None,
        log_level: Optional[int] = None,
        file_format: Optional[str] = None,
        console_format: Optional[str] = None
        ) -> None:
        """
        Initialize the logging configuration.

        Args:
            config_file: Path to YAML config file (new approach)
            logs_dir, log_file_name, etc.: Direct parameters (backwards compatibility)
        """
        # If any legacy parameters are provided, use them instead of YAML
        if any([logs_dir, log_file_name, log_level, file_format, console_format]):
            # Backwards compatibility mode
            self._logs_dir = self._resolve_logs_dir(logs_dir)
            self._log_file_name = log_file_name or self._generate_log_filename()
            self._log_level = log_level if log_level is not None else logging.INFO
            self._file_format = file_format or '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
            self._console_format = console_format or '%(levelname)s - %(message)s'
        else:
            # YAML configuration mode
            self._config = load_config(config_file)
            self._logs_dir = self._resolve_logs_dir()
            self._log_file_name = self._generate_log_filename()
            self._log_level = self._parse_log_level()
            self._file_format = self._config.get('format', {}).get('file', '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n')
            self._console_format = self._config.get('format', {}).get('console', '%(levelname)s - %(message)s')

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

    def _resolve_logs_dir(self, logs_dir: Optional[Union[Path, str]] = None) -> Path:
        """Resolve the logs directory from parameter or config."""
        if logs_dir is not None:
            # Backwards compatibility: direct parameter provided
            return Path(logs_dir)

        # YAML configuration mode
        if hasattr(self, '_config'):
            log_dir_config = self._config.get('log_directory', 'logs')
            # If it's an absolute path, use it directly
            if Path(log_dir_config).is_absolute():
                return Path(log_dir_config)
            # Otherwise, resolve relative to project root
            project_root = Path(__file__).resolve().parents[3]
            return project_root / log_dir_config

        # Default fallback
        default_root = Path(__file__).resolve().parents[3]
        return default_root / "logs"

    def _generate_log_filename(self) -> str:
        """Generate a log file name based on config or default pattern."""
        if hasattr(self, '_config'):
            # YAML configuration mode
            pattern = self._config.get('filename_pattern', 'app_{timestamp}.log')
            timestamp_format = self._config.get('timestamp_format', '%d%m%Y_%H%M%S')
            timestamp = datetime.now().strftime(timestamp_format)
            return pattern.format(timestamp=timestamp)
        else:
            # Default pattern
            timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
            return f"app_{timestamp}.log"

    def _parse_log_level(self) -> int:
        """Parse log level string to logging constant."""
        if not hasattr(self, '_config'):
            return logging.INFO

        level_str = self._config.get('level', 'INFO').upper()
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        return level_map.get(level_str, logging.INFO)

    def ensure_logs_dir_exists(self) -> None:
        """Ensure that the logs directory exists."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def get_config_value(self, key: str, default=None):
        """Get a configuration value by key (YAML mode only)."""
        if hasattr(self, '_config'):
            return self._config.get(key, default)
        return default