"""
Production Launcher - Legal RAG System

Production-ready launcher with health monitoring and auto-restart.
For Docker/Cloud deployment.

File: launch_production.py

Usage:
    python launch_production.py
    
Environment Variables:
    API_HOST: API server host (default: 0.0.0.0)
    API_PORT: API server port (default: 8000)
    UI_HOST: UI server host (default: 0.0.0.0)
    UI_PORT: UI server port (default: 7860)
    LEGAL_API_KEY: API authentication key
"""

import os
import sys
import time
import signal
import logging
import subprocess
import threading
from pathlib import Path
from datetime import datetime

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Configuration from environment
CONFIG = {
    'api_host': os.environ.get('API_HOST', '0.0.0.0'),
    'api_port': int(os.environ.get('API_PORT', 8000)),
    'ui_host': os.environ.get('UI_HOST', '0.0.0.0'),
    'ui_port': int(os.environ.get('UI_PORT', 7860)),
    'health_check_interval': 30,  # seconds
    'max_restart_attempts': 3,
    'restart_cooldown': 60  # seconds
}


class ServiceManager:
    """Manages API and UI services with health monitoring"""
    
    def __init__(self):
        self.processes = {}
        self.restart_counts = {'api': 0, 'ui': 0}
        self.last_restart = {'api': 0, 'ui': 0}
        self.running = True
        
    def start_api(self):
        """Start API server"""
        logger.info(f"Starting API server on {CONFIG['api_host']}:{CONFIG['api_port']}...")
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        
        cmd = [
            sys.executable, "-m", "uvicorn",
            "api.server:app",
            "--host", CONFIG['api_host'],
            "--port", str(CONFIG['api_port']),
            "--log-level", "warning",
            "--workers", "1"  # Single worker for GPU
        ]
        
        self.processes['api'] = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Stream output
        threading.Thread(
            target=self._stream_output,
            args=('api', self.processes['api']),
            daemon=True
        ).start()
        
    def start_ui(self):
        """Start Gradio UI"""
        logger.info(f"Starting UI on {CONFIG['ui_host']}:{CONFIG['ui_port']}...")
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        env["GRADIO_SERVER_NAME"] = CONFIG['ui_host']
        env["GRADIO_SERVER_PORT"] = str(CONFIG['ui_port'])
        
        cmd = [sys.executable, "-m", "ui.unified_app_api"]
        
        self.processes['ui'] = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Stream output
        threading.Thread(
            target=self._stream_output,
            args=('ui', self.processes['ui']),
            daemon=True
        ).start()
    
    def _stream_output(self, name: str, process):
        """Stream process output to logger"""
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    logger.info(f"[{name.upper()}] {line.rstrip()}")
        except:
            pass
    
    def check_api_health(self) -> bool:
        """Check if API is healthy"""
        try:
            import requests
            url = f"http://127.0.0.1:{CONFIG['api_port']}/api/v1/health"
            r = requests.get(url, timeout=5)
            return r.status_code == 200
        except:
            return False
    
    def check_process_alive(self, name: str) -> bool:
        """Check if process is running"""
        if name not in self.processes:
            return False
        return self.processes[name].poll() is None
    
    def restart_service(self, name: str):
        """Restart a service with cooldown"""
        now = time.time()
        
        # Check cooldown
        if now - self.last_restart[name] < CONFIG['restart_cooldown']:
            logger.warning(f"Skipping {name} restart (cooldown)")
            return
        
        # Check max attempts
        if self.restart_counts[name] >= CONFIG['max_restart_attempts']:
            logger.error(f"Max restart attempts reached for {name}")
            return
        
        logger.warning(f"Restarting {name}...")
        
        # Kill existing
        if name in self.processes and self.processes[name]:
            try:
                self.processes[name].terminate()
                self.processes[name].wait(timeout=10)
            except:
                self.processes[name].kill()
        
        # Restart
        if name == 'api':
            self.start_api()
        else:
            self.start_ui()
        
        self.restart_counts[name] += 1
        self.last_restart[name] = now
    
    def run_health_monitor(self):
        """Background health monitoring"""
        while self.running:
            time.sleep(CONFIG['health_check_interval'])
            
            # Check API
            if not self.check_process_alive('api'):
                logger.error("API process died!")
                self.restart_service('api')
            elif not self.check_api_health():
                logger.warning("API health check failed")
            
            # Check UI
            if not self.check_process_alive('ui'):
                logger.error("UI process died!")
                self.restart_service('ui')
    
    def wait_for_api_ready(self, timeout: int = 600):
        """Wait for API to be ready"""
        import requests
        
        logger.info("Waiting for API to be ready...")
        start = time.time()
        url = f"http://127.0.0.1:{CONFIG['api_port']}/api/v1/ready"
        
        while time.time() - start < timeout:
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    if data.get('ready'):
                        logger.info("API is ready!")
                        return True
                    else:
                        logger.info(f"Loading: {data.get('message', '...')}")
            except:
                pass
            time.sleep(10)
        
        logger.error("API failed to become ready")
        return False
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down services...")
        self.running = False
        
        for name, process in self.processes.items():
            if process:
                try:
                    process.terminate()
                    process.wait(timeout=10)
                    logger.info(f"{name} stopped")
                except:
                    process.kill()
                    logger.warning(f"{name} killed")
        
        logger.info("All services stopped")
    
    def run(self):
        """Main run loop"""
        logger.info("=" * 60)
        logger.info("LEGAL RAG - PRODUCTION MODE")
        logger.info("=" * 60)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, lambda s, f: self.shutdown())
        signal.signal(signal.SIGTERM, lambda s, f: self.shutdown())
        
        # Start services
        self.start_api()
        
        # Wait for API to load models
        if not self.wait_for_api_ready():
            logger.error("Failed to start API")
            self.shutdown()
            return
        
        # Start UI after API is ready
        self.start_ui()
        
        logger.info("=" * 60)
        logger.info(f"API:  http://{CONFIG['api_host']}:{CONFIG['api_port']}")
        logger.info(f"UI:   http://{CONFIG['ui_host']}:{CONFIG['ui_port']}")
        logger.info(f"Docs: http://{CONFIG['api_host']}:{CONFIG['api_port']}/docs")
        logger.info("=" * 60)
        
        # Start health monitoring
        monitor_thread = threading.Thread(target=self.run_health_monitor, daemon=True)
        monitor_thread.start()
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.shutdown()


def main():
    """Entry point"""
    manager = ServiceManager()
    manager.run()


if __name__ == "__main__":
    main()
