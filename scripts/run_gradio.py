#!/usr/bin/env python3
"""
Gradio Development Server Script

Runs the Gradio interface for development/demo purposes.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Run Gradio interface")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    from ui.gradio_app import launch_app

    launch_app(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
