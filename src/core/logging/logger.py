"""
Main logger interface for application use.
This module provides the CustomLogger class and a utility function to get logger instances.
"""
import logging
from typing import Optional
from .config import LoggingConfig
from .factory import LoggerFactory

class CustomLogger:
    """
    Main logger interface for application use.
    Provides a clean API for obtaining configured loggers.
    """
    def __init__(self, config: Optional[LoggingConfig] = None) -> None:
        """
        Initialize CustomLogger with optional configuration.
        """
        self._config = config if config is not None else LoggingConfig()
        self._factory = LoggerFactory(self._config)
        
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """
        Get a configured logger instance.
        """
        if name is None:
            name = self._get_caller_module_name()
            
        return self._factory.create_logger(name)
    
    @staticmethod
    def _get_caller_module_name() -> str:
        """
        Retrieve the module name of the caller.
        """
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            module = frame.f_back.f_globals.get("__name__", "__main__")
            return module
        return "__main__"
    
_default_logger_instance: Optional[CustomLogger] = None

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance using the default configuration.
    """
    global _default_logger_instance
    if _default_logger_instance is None:
        _default_logger_instance = CustomLogger()
    return _default_logger_instance.get_logger(name)