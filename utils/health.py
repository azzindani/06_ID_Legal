"""
System Health Monitoring Utilities

This module provides functions for checking and reporting system health,
including memory, GPU, and component status.

File: utils/health.py
"""

import torch
from typing import Dict, Any, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def system_health_check(
    pipeline: Optional[Any] = None,
    manager: Optional[Any] = None,
    initialization_complete: bool = False
) -> Dict[str, Any]:
    """
    Run comprehensive system health check

    Checks component initialization, memory usage, and GPU status.

    Args:
        pipeline: RAG pipeline instance (optional)
        manager: Conversation manager instance (optional)
        initialization_complete: Whether system initialization is complete

    Returns:
        Dictionary with health status:
        {
            'status': 'healthy' | 'warning' | 'critical',
            'components': {...},
            'memory': {...},
            'gpu': {...},
            'issues': [...]
        }
    """
    health = {
        'status': 'healthy',
        'components': {},
        'memory': {},
        'gpu': {},
        'issues': []
    }

    # Check initialization
    health['components']['pipeline'] = pipeline is not None
    health['components']['manager'] = manager is not None
    health['components']['initialization'] = initialization_complete

    # Memory check
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        health['memory']['used_gb'] = mem.used / 1024**3
        health['memory']['total_gb'] = mem.total / 1024**3
        health['memory']['percent'] = mem.percent

        if mem.percent > 90:
            health['issues'].append("Critical: Memory usage above 90%")
            health['status'] = 'critical'
        elif mem.percent > 80:
            health['issues'].append("Warning: Memory usage above 80%")
            if health['status'] == 'healthy':
                health['status'] = 'warning'
    else:
        health['memory']['available'] = False
        health['issues'].append("Warning: psutil not available, cannot check memory")

    # GPU check
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_used = torch.cuda.memory_allocated(i) / 1024**3
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            health['gpu'][f'gpu_{i}'] = {
                'used_gb': mem_used,
                'total_gb': mem_total,
                'percent': (mem_used / mem_total) * 100 if mem_total > 0 else 0
            }

            # Check GPU memory usage
            gpu_percent = (mem_used / mem_total) * 100 if mem_total > 0 else 0
            if gpu_percent > 90:
                health['issues'].append(f"Critical: GPU {i} memory usage above 90%")
                health['status'] = 'critical'
    else:
        health['gpu']['available'] = False

    return health


def format_health_report(health: Dict[str, Any]) -> str:
    """
    Format health check results for human-readable display

    Args:
        health: Health dictionary from system_health_check()

    Returns:
        Formatted markdown string with health report
    """
    status_emoji = {
        'healthy': '✅',
        'warning': '⚠️',
        'critical': '❌'
    }

    report = f"## {status_emoji.get(health['status'], '❓')} System Health: {health['status'].upper()}\n\n"

    # Components
    report += "### Components\n"
    for comp, status in health['components'].items():
        emoji = '✅' if status else '❌'
        report += f"- {comp}: {emoji}\n"

    # Memory
    if 'used_gb' in health['memory']:
        report += f"\n### Memory\n"
        report += f"- Used: {health['memory']['used_gb']:.1f} / {health['memory']['total_gb']:.1f} GB ({health['memory']['percent']:.1f}%)\n"
    elif 'available' in health['memory'] and not health['memory']['available']:
        report += f"\n### Memory\n"
        report += "- Memory monitoring not available (psutil not installed)\n"

    # GPU
    if health['gpu']:
        report += f"\n### GPU\n"
        if 'available' in health['gpu'] and not health['gpu']['available']:
            report += "- No GPU available\n"
        else:
            for gpu_id, gpu_info in health['gpu'].items():
                if isinstance(gpu_info, dict):
                    report += f"- {gpu_id}: {gpu_info['used_gb']:.1f} / {gpu_info['total_gb']:.1f} GB ({gpu_info['percent']:.1f}%)\n"

    # Issues
    if health['issues']:
        report += f"\n### Issues\n"
        for issue in health['issues']:
            report += f"- {issue}\n"

    return report
