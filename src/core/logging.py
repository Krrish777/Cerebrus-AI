import os
import logging
from datetime import datetime
from pathlib import Path

class CustomLogger:
    def __init__(self, logs_dir: str | Path | None = None, *, log_file_name: str | None = None):
        '''
        Initialize the CustomLogger.

        By default logs are written to the repository root `logs/` directory. You can
        override this by passing `logs_dir` (str or Path) or set an explicit
        `log_file_name`.
        '''
        # Default to repository root logs folder (two parents up from this file)
        default_root = Path(__file__).resolve().parents[2]
        self.logs_dir = Path(logs_dir) if logs_dir is not None else (default_root / "logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        if log_file_name is None:
            log_file_name = f"app_{datetime.now().strftime('%d%m%Y_%H%M%S')}.log"

        self.log_path = str(self.logs_dir / log_file_name)

        # Ensure configuration happens only once per instance
        self._configured = False
        self._logger = None
        
    def get_logger(self, name=__file__):
        '''Get a logger instance with the specified name.'''
        logger_name = os.path.basename(name).replace(".py", "")

        # Configure logging once per instance
        if not self._configured:
            # Create logger
            self._logger = logging.getLogger(logger_name)
            self._logger.setLevel(logging.INFO)
            
            # Clear any existing handlers
            self._logger.handlers = []
            
            # File handler
            file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            
            # Console handler  
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            
            # Add handlers
            self._logger.addHandler(file_handler)
            self._logger.addHandler(console_handler)

            self._configured = True

        return self._logger

    # --- Backwards-compatible class-level convenience methods ---
    # Some modules import the class as `from ...custom_logger import CustomLogger as log`
    # and then call `log.info(...)`. Provide class-level forwarding methods so
    # those call sites continue to work without changing imports.
    _default_logger = None
    _default_instance = None

    @classmethod
    def _ensure_default(cls):
        if cls._default_logger is None:
            inst = cls()
            cls._default_instance = inst
            # Use module name as logger name for the default
            cls._default_logger = inst.get_logger(__file__)
        return cls._default_logger

    @classmethod
    def info(cls, *args, **kwargs):
        logger = cls._ensure_default()
        if logger is not None:
            return logger.info(*args, **kwargs)

    @classmethod
    def error(cls, *args, **kwargs):
        logger = cls._ensure_default()
        if logger is not None:
            return logger.error(*args, **kwargs)

    @classmethod
    def warning(cls, *args, **kwargs):
        logger = cls._ensure_default()
        if logger is not None:
            return logger.warning(*args, **kwargs)

    @classmethod
    def debug(cls, *args, **kwargs):
        logger = cls._ensure_default()
        if logger is not None:
            return logger.debug(*args, **kwargs)

    @classmethod
    def critical(cls, *args, **kwargs):
        logger = cls._ensure_default()
        if logger is not None:
            return logger.critical(*args, **kwargs)

    @classmethod
    def exception(cls, *args, **kwargs):
        logger = cls._ensure_default()
        if logger is not None:
            return logger.exception(*args, **kwargs)