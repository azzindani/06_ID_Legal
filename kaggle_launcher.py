"""
Kaggle Launcher - Legal RAG System

This script is specifically designed for running in Kaggle notebooks.
It handles the unique requirements of the Kaggle environment:
- Starts API server in background thread (not subprocess)
- Launches Gradio UI with proper share settings
- Manages cleanup on shutdown

Usage in Kaggle:
    %run kaggle_launcher.py

File: kaggle_launcher.py
"""

import os
import sys
import time
import gc
import threading
from pathlib import Path

# =============================================================================
# PATH SETUP
# =============================================================================

# Detect Kaggle environment
IS_KAGGLE = os.path.exists('/kaggle')

if IS_KAGGLE:
    PROJECT_ROOT = Path('/kaggle/working/06_ID_Legal')
else:
    PROJECT_ROOT = Path(__file__).parent

sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

print(f"[KAGGLE] Project root: {PROJECT_ROOT}")
print(f"[KAGGLE] Is Kaggle: {IS_KAGGLE}")


# =============================================================================
# MEMORY MANAGEMENT
# =============================================================================

def clear_gpu_memory():
    """Clear GPU memory before starting"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[KAGGLE] GPU: {allocated:.2f}GB / {total:.2f}GB used")
    except Exception as e:
        print(f"[KAGGLE] GPU check failed: {e}")


# =============================================================================
# API SERVER (runs in background thread)
# =============================================================================

api_server_thread = None
api_app = None

def start_api_server_thread():
    """Start the API server in a background thread"""
    global api_app
    
    print("[KAGGLE] Initializing API server...")
    
    from api.server import create_app
    import uvicorn
    
    # Create the app (this loads the pipeline)
    api_app = create_app()
    
    # Configure uvicorn
    config = uvicorn.Config(
        app=api_app,
        host="127.0.0.1",
        port=8000,
        log_level="warning",
        access_log=False,
        loop="asyncio"
    )
    
    server = uvicorn.Server(config)
    
    print("[KAGGLE] Starting API server on http://127.0.0.1:8000")
    server.run()


def start_api_in_background():
    """Start API server in background thread"""
    global api_server_thread
    
    api_server_thread = threading.Thread(
        target=start_api_server_thread,
        daemon=True,
        name="APIServer"
    )
    api_server_thread.start()
    
    # Wait for server to be ready
    import requests
    
    print("[KAGGLE] Waiting for API to be ready...")
    max_wait = 600  # 10 minutes for model loading
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            resp = requests.get("http://127.0.0.1:8000/api/v1/health", timeout=5)
            if resp.status_code == 200:
                print("[KAGGLE] ✅ API server is ready!")
                return True
        except:
            pass
        
        elapsed = int(time.time() - start_time)
        if elapsed % 30 == 0 and elapsed > 0:
            print(f"[KAGGLE] Still loading... ({elapsed}s elapsed)")
        
        time.sleep(5)
    
    print("[KAGGLE] ❌ API server failed to start within timeout")
    return False


# =============================================================================
# GRADIO UI
# =============================================================================

def launch_gradio():
    """Launch Gradio UI"""
    print("[KAGGLE] Starting Gradio UI...")
    
    from ui.unified_app_api import launch_app
    
    # In Kaggle, we need share=True to get a public URL
    share = IS_KAGGLE
    
    launch_app(
        share=share,
        server_port=7860,
        server_name="0.0.0.0"
    )


# =============================================================================
# MAIN LAUNCHER
# =============================================================================

def main():
    """Main Kaggle launcher"""
    print("=" * 60)
    print("🏛️ LEGAL RAG INDONESIA - KAGGLE LAUNCHER")
    print("=" * 60)
    
    # Step 1: Clear GPU memory
    print("\n[Step 1/3] Clearing GPU memory...")
    clear_gpu_memory()
    
    # Step 2: Start API server in background
    print("\n[Step 2/3] Starting API server...")
    if not start_api_in_background():
        print("[KAGGLE] Failed to start API server. Exiting.")
        return
    
    # Small delay to ensure API is fully ready
    time.sleep(2)
    
    # Step 3: Launch Gradio UI
    print("\n[Step 3/3] Launching Gradio UI...")
    print("=" * 60)
    
    launch_gradio()


if __name__ == "__main__":
    main()
