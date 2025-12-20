#!/usr/bin/env python3
"""
Production Server Script

Runs the FastAPI server with production settings.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LOG_DIR, ENABLE_FILE_LOGGING, LOG_VERBOSITY
from utils.logger_utils import initialize_logging


def main():
    parser = argparse.ArgumentParser(description="Run FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    # Initialize logging
    initialize_logging(
        enable_file_logging=ENABLE_FILE_LOGGING,
        log_dir=LOG_DIR,
        verbosity_mode=LOG_VERBOSITY
    )

    import uvicorn

    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
