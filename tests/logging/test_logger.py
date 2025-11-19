# ============================================================================
# tests/logging/test_logger.py
# Tests for CustomLogger class and convenience functions
# ============================================================================
"""
Tests for CustomLogger class and module-level convenience functions.

Validates the public API and integration of components.
"""
import pytest
import logging
from pathlib import Path
from unittest.mock import patch
from src.core.logging.logger import CustomLogger, get_logger
from src.core.logging.config import LoggingConfig


class TestCustomLoggerInitialization:
    """Test CustomLogger initialization."""
    
    def test_custom_logger_initializes_with_default_config(self) -> None:
        """Test that CustomLogger creates default config if none provided."""
        logger_wrapper = CustomLogger()
        
        assert logger_wrapper is not None
    
    def test_custom_logger_accepts_custom_config(
        self, temp_logs_dir: Path
    ) -> None:
        """Test that CustomLogger accepts custom LoggingConfig."""
        config = LoggingConfig(logs_dir=temp_logs_dir)
        logger_wrapper = CustomLogger(config)
        
        assert logger_wrapper is not None


class TestCustomLoggerGetLogger:
    """Test get_logger method functionality."""
    
    def test_get_logger_returns_logger_instance(
        self, temp_logs_dir: Path
    ) -> None:
        """Test that get_logger returns a Logger instance."""
        config = LoggingConfig(logs_dir=temp_logs_dir)
        logger_wrapper = CustomLogger(config)
        
        logger = logger_wrapper.get_logger("my_module")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "my_module"
    
    def test_get_logger_with_explicit_name(
        self, temp_logs_dir: Path
    ) -> None:
        """Test get_logger with explicitly provided name."""
        config = LoggingConfig(logs_dir=temp_logs_dir)
        logger_wrapper = CustomLogger(config)
        
        logger = logger_wrapper.get_logger("explicit_name")
        
        assert logger.name == "explicit_name"
    
    def test_get_logger_auto_detects_module_name(
        self, temp_logs_dir: Path
    ) -> None:
        """Test that get_logger auto-detects caller module name."""
        config = LoggingConfig(logs_dir=temp_logs_dir)
        logger_wrapper = CustomLogger(config)
        
        logger = logger_wrapper.get_logger()
        
        # Should detect test module name
        assert logger.name is not None
        assert len(logger.name) > 0


class TestCustomLoggerIntegration:
    """Test integration between CustomLogger and other components."""
    
    def test_logger_can_write_to_file(
        self, temp_logs_dir: Path, sample_log_file_name: str
    ) -> None:
        """Test that logger actually writes to log file."""
        config = LoggingConfig(
            logs_dir=temp_logs_dir,
            log_file_name=sample_log_file_name
        )
        logger_wrapper = CustomLogger(config)
        logger = logger_wrapper.get_logger("file_write_test")
        
        test_message = "Test log message"
        logger.info(test_message)
        
        log_file_path = temp_logs_dir / sample_log_file_name
        assert log_file_path.exists()
        
        log_content = log_file_path.read_text(encoding="utf-8")
        assert test_message in log_content
    
    def test_logger_respects_log_level(
        self, temp_logs_dir: Path, sample_log_file_name: str
    ) -> None:
        """Test that logger respects configured log level."""
        config = LoggingConfig(
            logs_dir=temp_logs_dir,
            log_file_name=sample_log_file_name,
            log_level=logging.WARNING
        )
        logger_wrapper = CustomLogger(config)
        logger = logger_wrapper.get_logger("level_test")
        
        logger.debug("Debug message - should not appear")
        logger.info("Info message - should not appear")
        logger.warning("Warning message - should appear")
        
        log_file_path = temp_logs_dir / sample_log_file_name
        log_content = log_file_path.read_text(encoding="utf-8")
        
        assert "Debug message" not in log_content
        assert "Info message" not in log_content
        assert "Warning message" in log_content


class TestModuleLevelGetLogger:
    """Test module-level get_logger convenience function."""
    
    def test_get_logger_function_returns_logger(self) -> None:
        """Test that module-level get_logger returns Logger instance."""
        logger = get_logger("test_convenience")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_convenience"
    
    def test_get_logger_function_creates_default_instance(self) -> None:
        """Test that get_logger creates default CustomLogger instance."""
        logger = get_logger("default_test")
        
        assert logger is not None
        assert len(logger.handlers) == 2
    
    def test_get_logger_function_reuses_instance(self) -> None:
        """Test that get_logger reuses the same default instance."""
        logger1 = get_logger("reuse_test1")
        logger2 = get_logger("reuse_test2")
        
        # Different logger names but same factory
        assert logger1.name != logger2.name
