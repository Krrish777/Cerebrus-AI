# ============================================================================
# tests/logging/test_integration.py
# Integration tests for the entire logging package
# ============================================================================
"""
Integration tests for the logging package.

Tests the complete workflow from configuration to logging.
"""
import os
import pytest
import logging
from pathlib import Path
from src.core.logging import CustomLogger, LoggingConfig, get_logger


class TestEndToEndLogging:
    """Test complete logging workflows."""
    
    def test_complete_logging_workflow_with_custom_config(
        self, temp_logs_dir: Path
    ) -> None:
        """Test complete workflow: config -> logger -> write -> verify."""
        # Step 1: Create configuration
        config = LoggingConfig(
            logs_dir=temp_logs_dir,
            log_file_name="integration_test.log",
            log_level=logging.DEBUG
        )
        
        # Step 2: Create logger instance
        logger_wrapper = CustomLogger(config)
        logger = logger_wrapper.get_logger("integration_test")
        
        # Step 3: Write various log levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Step 4: Verify log file contents
        log_file = temp_logs_dir / "integration_test.log"
        assert log_file.exists()
        
        content = log_file.read_text(encoding="utf-8")
        assert "Debug message" in content
        assert "Info message" in content
        assert "Warning message" in content
        assert "Error message" in content
        assert "integration_test" in content  # Logger name in format
    
    def test_simple_workflow_with_convenience_function(self) -> None:
        """Test simple workflow using convenience function."""
        logger = get_logger(__name__)
        
        # Should not raise exception
        logger.info("Simple test message")
        logger.warning("Simple warning")
        
        assert len(logger.handlers) == 2
    
    @pytest.mark.parametrize("log_level,messages", [
        (logging.DEBUG, ["debug", "info", "warning", "error"]),
        (logging.INFO, ["info", "warning", "error"]),
        (logging.WARNING, ["warning", "error"]),
        (logging.ERROR, ["error"]),
    ])
    def test_different_log_levels_filter_correctly(
        self,
        temp_logs_dir: Path,
        log_level: int,
        messages: list
    ) -> None:
        """Test that different log levels filter messages correctly."""
        config = LoggingConfig(
            logs_dir=temp_logs_dir,
            log_file_name=f"level_{log_level}.log",
            log_level=log_level
        )
        logger_wrapper = CustomLogger(config)
        logger = logger_wrapper.get_logger(f"level_filter_test_{log_level}")
        
        # Write all levels
        logger.debug("debug")
        logger.info("info")
        logger.warning("warning")
        logger.error("error")
        
        # Check what appears in log
        log_file = temp_logs_dir / f"level_{log_level}.log"
        content = log_file.read_text(encoding="utf-8")
        
        for expected_message in messages:
            assert expected_message in content.lower()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.skipif(os.name == 'nt', reason="Readonly directory test not reliable on Windows")
    def test_logging_to_readonly_directory_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Test that logging to read-only directory raises appropriate error."""
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only
        
        config = LoggingConfig(logs_dir=readonly_dir)
        
        with pytest.raises(PermissionError):
            logger_wrapper = CustomLogger(config)
            logger = logger_wrapper.get_logger("readonly_test")
            logger.info("This should fail")
    
    def test_logger_handles_unicode_messages(
        self, temp_logs_dir: Path
    ) -> None:
        """Test that logger correctly handles Unicode characters."""
        config = LoggingConfig(logs_dir=temp_logs_dir, log_file_name="unicode.log")
        logger_wrapper = CustomLogger(config)
        logger = logger_wrapper.get_logger("unicode_test")
        
        unicode_message = "Testing Unicode: 你好世界 🌍 café"
        logger.info(unicode_message)
        
        log_file = temp_logs_dir / "unicode.log"
        content = log_file.read_text(encoding="utf-8")
        assert unicode_message in content
