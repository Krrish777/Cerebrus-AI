# ============================================================================
# tests/logging/test_factory.py
# Tests for LoggerFactory class
# ============================================================================
"""
Tests for LoggerFactory class.

Validates logger creation and handler configuration.
"""
import pytest
import logging
from pathlib import Path
from src.core.logging.config import LoggingConfig
from src.core.logging.factory import LoggerFactory


class TestLoggerFactoryInitialization:
    """Test LoggerFactory initialization."""
    
    def test_factory_requires_config(self, temp_logs_dir: Path) -> None:
        """Test that factory requires LoggingConfig instance."""
        config = LoggingConfig(logs_dir=temp_logs_dir)
        factory = LoggerFactory(config)
        
        assert factory is not None
    
    def test_factory_creates_logs_directory_on_init(
        self, tmp_path: Path
    ) -> None:
        """Test that factory ensures logs directory exists on initialization."""
        logs_dir = tmp_path / "factory_logs"
        config = LoggingConfig(logs_dir=logs_dir)
        
        assert not logs_dir.exists()
        
        factory = LoggerFactory(config)
        
        assert logs_dir.exists()


class TestLoggerFactoryLoggerCreation:
    """Test logger creation functionality."""
    
    def test_create_logger_returns_logger_instance(
        self, temp_logs_dir: Path
    ) -> None:
        """Test that create_logger returns a Logger instance."""
        config = LoggingConfig(logs_dir=temp_logs_dir)
        factory = LoggerFactory(config)
        
        logger = factory.create_logger("test_module")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"
    
    def test_create_logger_sets_correct_log_level(
        self, temp_logs_dir: Path
    ) -> None:
        """Test that logger has correct log level."""
        config = LoggingConfig(logs_dir=temp_logs_dir, log_level=logging.DEBUG)
        factory = LoggerFactory(config)
        
        logger = factory.create_logger("debug_logger")
        
        assert logger.level == logging.DEBUG
    
    def test_create_logger_adds_two_handlers(
        self, temp_logs_dir: Path
    ) -> None:
        """Test that logger has file and console handlers."""
        config = LoggingConfig(logs_dir=temp_logs_dir)
        factory = LoggerFactory(config)
        
        logger = factory.create_logger("test_logger")
        
        assert len(logger.handlers) == 2
    
    def test_create_logger_adds_file_handler(
        self, temp_logs_dir: Path
    ) -> None:
        """Test that a FileHandler is added to the logger."""
        config = LoggingConfig(logs_dir=temp_logs_dir)
        factory = LoggerFactory(config)
        
        logger = factory.create_logger("file_test")
        
        file_handlers = [
            h for h in logger.handlers 
            if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
    
    def test_create_logger_adds_console_handler(
        self, temp_logs_dir: Path
    ) -> None:
        """Test that a StreamHandler is added to the logger."""
        config = LoggingConfig(logs_dir=temp_logs_dir)
        factory = LoggerFactory(config)
        
        logger = factory.create_logger("console_test")
        
        stream_handlers = [
            h for h in logger.handlers 
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 1
    
    def test_create_logger_prevents_duplicate_handlers(
        self, temp_logs_dir: Path
    ) -> None:
        """Test that calling create_logger twice doesn't duplicate handlers."""
        config = LoggingConfig(logs_dir=temp_logs_dir)
        factory = LoggerFactory(config)
        
        logger1 = factory.create_logger("duplicate_test")
        logger2 = factory.create_logger("duplicate_test")
        
        assert logger1 is logger2  # Same logger instance
        assert len(logger1.handlers) == 2  # Still only 2 handlers


class TestLoggerFactoryHandlerConfiguration:
    """Test handler configuration details."""
    
    def test_file_handler_uses_correct_path(
        self, temp_logs_dir: Path, sample_log_file_name: str
    ) -> None:
        """Test that file handler writes to correct path."""
        config = LoggingConfig(
            logs_dir=temp_logs_dir,
            log_file_name=sample_log_file_name
        )
        factory = LoggerFactory(config)
        
        logger = factory.create_logger("path_test")
        
        file_handler = next(
            h for h in logger.handlers 
            if isinstance(h, logging.FileHandler)
        )
        expected_path = str(temp_logs_dir / sample_log_file_name)
        assert file_handler.baseFilename == expected_path
    
    def test_handlers_have_correct_formatters(
        self, temp_logs_dir: Path
    ) -> None:
        """Test that handlers have formatters attached."""
        config = LoggingConfig(logs_dir=temp_logs_dir)
        factory = LoggerFactory(config)
        
        logger = factory.create_logger("format_test")
        
        for handler in logger.handlers:
            assert handler.formatter is not None
