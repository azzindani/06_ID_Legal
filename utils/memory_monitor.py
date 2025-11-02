# utils/memory_monitor.py
"""
Memory usage monitoring and optimization utilities.
"""
import gc
import torch
import psutil
import os
from typing import Dict, Any, Optional
from utils.logging_config import get_logger

logger = get_logger(__name__)

class MemoryMonitor:
    """
    Monitor and manage memory usage.
    """
    
    def __init__(self, warning_threshold_percent: float = 85.0):
        """
        Initialize memory monitor.
        
        Args:
            warning_threshold_percent: Log warning when memory exceeds this
        """
        self.warning_threshold = warning_threshold_percent
        self.process = psutil.Process(os.getpid())
        
        logger.info(f"MemoryMonitor initialized (warning at {warning_threshold_percent}%)")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        try:
            # System memory
            system_memory = psutil.virtual_memory()
            
            # Process memory
            process_memory = self.process.memory_info()
            
            # GPU memory (if available)
            gpu_info = {}
            if torch.cuda.is_available():
                gpu_info = {
                    'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                    'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                    'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3
                }
            
            return {
                'system': {
                    'total_gb': system_memory.total / 1024**3,
                    'available_gb': system_memory.available / 1024**3,
                    'used_gb': system_memory.used / 1024**3,
                    'percent': system_memory.percent
                },
                'process': {
                    'rss_gb': process_memory.rss / 1024**3,  # Resident Set Size
                    'vms_gb': process_memory.vms / 1024**3   # Virtual Memory Size
                },
                'gpu': gpu_info
            }
            
        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return {}
    
    def check_memory(self, log_level: str = 'info') -> bool:
        """
        Check memory usage and log if needed.
        
        Args:
            log_level: Logging level ('debug', 'info', 'warning')
            
        Returns:
            True if memory is OK, False if above threshold
        """
        info = self.get_memory_info()
        
        if not info:
            return True
        
        system_percent = info['system']['percent']
        
        # Format message
        msg = (
            f"Memory: {system_percent:.1f}% "
            f"(Process: {info['process']['rss_gb']:.2f}GB)"
        )
        
        if info['gpu']:
            msg += f" | GPU: {info['gpu']['allocated_gb']:.2f}GB"
        
        # Log based on threshold
        if system_percent > self.warning_threshold:
            logger.warning(f"⚠️ HIGH {msg}")
            return False
        else:
            if log_level == 'debug':
                logger.debug(msg)
            elif log_level == 'info':
                logger.info(msg)
        
        return True
    
    def optimize_memory(self):
        """Run memory optimization procedures."""
        logger.info("Running memory optimization...")
        
        # Python garbage collection
        collected = gc.collect()
        logger.debug(f"Garbage collected: {collected} objects")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
        
        # Get updated memory
        info = self.get_memory_info()
        logger.info(
            f"Memory after optimization: "
            f"{info['system']['percent']:.1f}% "
            f"({info['process']['rss_gb']:.2f}GB)"
        )
    
    def get_recommendations(self) -> list[str]:
        """Get memory optimization recommendations."""
        recommendations = []
        info = self.get_memory_info()
        
        if not info:
            return recommendations
        
        system_percent = info['system']['percent']
        
        if system_percent > 90:
            recommendations.append("⚠️ CRITICAL: System memory >90%. Consider reducing batch sizes.")
        elif system_percent > 85:
            recommendations.append("⚠️ HIGH: System memory >85%. Monitor closely.")
        
        if info['gpu'] and info['gpu']['allocated_gb'] > 10:
            recommendations.append("ℹ️ GPU memory usage high. Consider using CPU for some operations.")
        
        if info['process']['rss_gb'] > 16:
            recommendations.append("ℹ️ Process using >16GB. Consider enabling caching to disk.")
        
        return recommendations