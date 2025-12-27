"""
Development Launcher - Legal RAG System

Starts both API server and Unified UI for local development.
Includes auto-reload and debug logging.

File: launch_dev.py

Usage:
    python launch_dev.py           # Local only
    python launch_dev.py --share   # With public share link
"""

import os
import sys
import time
import signal
import subprocess
import threading
import argparse
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_api_server(host: str = "127.0.0.1", port: int = 8000):
    """Start API server with reload"""
    print(f"🚀 Starting API server on {host}:{port}...")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "api.server:app",
        "--host", host,
        "--port", str(port),
        "--reload",
        "--log-level", "info"
    ]
    
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )


def run_gradio_ui(host: str = "127.0.0.1", port: int = 7860, share: bool = False):
    """Start Gradio UI"""
    mode = "with PUBLIC SHARE LINK" if share else f"on {host}:{port}"
    print(f"🎨 Starting Unified UI {mode}...")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["GRADIO_SERVER_NAME"] = host
    env["GRADIO_SERVER_PORT"] = str(port)
    env["GRADIO_SHARE"] = "true" if share else "false"
    
    cmd = [
        sys.executable, "-m", "ui.unified_app_api"
    ]
    
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )


def stream_output(process, prefix: str):
    """Stream process output to console"""
    try:
        for line in process.stdout:
            print(f"{prefix} {line.rstrip()}")
    except:
        pass


def wait_for_api_ready(url: str = "http://127.0.0.1:8000/api/v1/ready", timeout: int = 300):
    """Wait for API to be ready"""
    import requests
    
    print("⏳ Waiting for API to be ready...")
    start = time.time()
    
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200 and r.json().get('ready'):
                print("✅ API is ready!")
                return True
            else:
                msg = r.json().get('message', 'Loading...')
                print(f"   {msg}")
        except:
            pass
        time.sleep(5)
    
    return False


def main(share: bool = False):
    """Main entry point"""
    print("\n" + "=" * 60)
    print("🔧 LEGAL RAG - DEVELOPMENT MODE")
    if share:
        print("📡 SHARE MODE ENABLED - Will generate public link")
    print("=" * 60)
    print("Starting services...")
    print("=" * 60 + "\n")
    
    processes = []
    
    try:
        # Start API server
        api_process = run_api_server()
        processes.append(api_process)
        
        # Stream API output in background
        api_thread = threading.Thread(
            target=stream_output, 
            args=(api_process, "[API]"),
            daemon=True
        )
        api_thread.start()
        
        # Wait for API to load (model loading takes time)
        time.sleep(10)
        
        # Start UI with share option
        ui_process = run_gradio_ui(share=share)
        processes.append(ui_process)
        
        # Stream UI output in background
        ui_thread = threading.Thread(
            target=stream_output,
            args=(ui_process, "[UI]"),
            daemon=True
        )
        ui_thread.start()
        
        print("\n" + "=" * 60)
        print("✅ Services started!")
        print("   API:  http://127.0.0.1:8000")
        print("   UI:   http://127.0.0.1:7860")
        if share:
            print("   📡 Share link will appear in [UI] output above")
        print("   Docs: http://127.0.0.1:8000/docs")
        print("=" * 60)
        print("Press Ctrl+C to stop...\n")
        
        # Wait for processes
        while True:
            for p in processes:
                if p.poll() is not None:
                    print(f"⚠️ A process exited with code {p.returncode}")
                    raise KeyboardInterrupt
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        for p in processes:
            try:
                p.terminate()
                p.wait(timeout=5)
            except:
                p.kill()
        print("✅ All services stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legal RAG Development Launcher")
    parser.add_argument("--share", action="store_true", help="Generate public share link")
    args = parser.parse_args()
    
    main(share=args.share)
