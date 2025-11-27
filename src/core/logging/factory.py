"""
Logger factory for creating configured logger instances.
This module handles the creation and configuration of Python loggers.
"""
import logging
from .config import LoggingConfig

class LoggerFactory:
    """
    Factory for creating configured loggers instances.
    Handles logger creation with proper handler setup and formatting.
    """
    def __init__(self, config: LoggingConfig) -> None:
        """
        Initialize factory with configuration.
        """
        self._config = config
        self._config.ensure_logs_dir_exists()
        
    def create_logger(self, name: str) -> logging.Logger:
        """
        Create and configure a logger instance.
        """
        logger = logging.getLogger(name)
        
        if logger.handlers:
            return logger  # Logger already configured
        logger.setLevel(self._config.log_level)
        logger.addHandler(self._create_file_handler())
        logger.addHandler(self._create_console_handler())
        return logger
    
    def _create_file_handler(self) -> logging.Handler:
        """
        Create configured file handler.
        """
        import os
        class FlushingFileHandler(logging.FileHandler):
            def emit(self, record):
                super().emit(record)
                self.flush()
                if hasattr(self.stream, 'fileno'):
                    try:
                        os.fsync(self.stream.fileno())
                    except (OSError, AttributeError):
                        pass  # Ignore if not supported
        
        handler = FlushingFileHandler(
            self._config.log_file_name,
            encoding="utf-8"
        )
        handler.setLevel(self._config.log_level)
        formatter = logging.Formatter(self._config.file_format)
        handler.setFormatter(formatter)
        return handler
    
    def _create_console_handler(self) -> logging.Handler:
        """
        Create configured console handler.
        """
        handler = logging.StreamHandler()
        handler.setLevel(self._config.log_level)
        formatter = logging.Formatter(self._config.console_format)
        handler.setFormatter(formatter)
        return handler