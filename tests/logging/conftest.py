# ============================================================================
# tests/logging/conftest.py
# Shared fixtures for logging tests
# ============================================================================
"""
Shared pytest fixtures for logging package tests.

Provides common setup and teardown for test isolation.
"""
import pytest
import logging
from pathlib import Path
from typing import Generator


@pytest.fixture
def temp_logs_dir(tmp_path: Path) -> Path:
    """
    Provide a temporary directory for log files.
    
    :param tmp_path: pytest's built-in temporary directory fixture
    :return: Path to temporary logs directory
    """
    logs_dir = tmp_path / "test_logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


@pytest.fixture
def sample_log_file_name() -> str:
    """Provide a consistent log file name for tests."""
    return "test_app.log"


@pytest.fixture(autouse=True)
def cleanup_logging_handlers() -> Generator[None, None, None]:
    """
    Clean up logging handlers after each test.
    
    Prevents handler accumulation and test interference.
    """
    yield
    # Clean up all loggers after test
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)