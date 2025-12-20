"""
Pytest configuration and fixtures for all tests

This file is automatically loaded by pytest and provides:
- Logging initialization for all tests
- Common fixtures
- Test configuration
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LOG_DIR, ENABLE_FILE_LOGGING, LOG_VERBOSITY
from utils.logger_utils import initialize_logging
import pytest


def pytest_configure(config):
    """
    Called after command line options have been parsed and all plugins loaded.
    Initialize logging for all test sessions.
    """
    # Check if running in verbose mode
    verbosity = 'verbose' if config.option.verbose else LOG_VERBOSITY

    # Initialize logging system
    initialize_logging(
        enable_file_logging=ENABLE_FILE_LOGGING,
        log_dir=LOG_DIR,
        verbosity_mode=verbosity
    )


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """
    Session-wide fixture to ensure logging is initialized.
    This runs automatically for all tests.
    """
    # Logging already initialized in pytest_configure
    # This fixture exists to ensure proper teardown if needed
    yield
    # Teardown code can go here if needed
