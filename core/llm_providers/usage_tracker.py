"""
Usage Tracker - Token and Cost Tracking

Tracks LLM token usage and estimated costs per session
and aggregated over time.

File: core/llm_providers/usage_tracker.py
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path

# Import logger
try:
    from utils.logger_utils import get_logger
    logger = get_logger("UsageTracker")
except ImportError:
    import logging
    logger = logging.getLogger("UsageTracker")


@dataclass
class UsageRecord:
    """Single usage record"""
    timestamp: float
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float = 0.0
    generation_time: float = 0.0
    query_preview: str = ""  # First 50 chars of query
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UsageRecord':
        return cls(**data)


class UsageTracker:
    """
    Track LLM token usage and costs.
    
    Features:
    - Per-session tracking
    - Persistent storage (optional)
    - Aggregated statistics
    - CSV export
    - Cost estimation
    
    Usage:
        tracker = UsageTracker()
        
        # Record usage
        tracker.record(UsageRecord(
            timestamp=time.time(),
            provider="openrouter",
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        ))
        
        # Get stats
        stats = tracker.get_session_stats()
    """
    
    STORAGE_DIR = Path.home() / ".legal_rag"
    USAGE_FILE = "usage_history.json"
    
    def __init__(
        self,
        persist: bool = True,
        max_history: int = 1000
    ):
        """
        Initialize usage tracker.
        
        Args:
            persist: Whether to persist usage to disk
            max_history: Maximum records to keep in memory
        """
        self.persist = persist
        self.max_history = max_history
        
        # Session records (current session only)
        self.session_records: List[UsageRecord] = []
        self.session_start = time.time()
        
        # Historical records (from file)
        self.history: List[UsageRecord] = []
        
        if persist:
            self._load_history()
        
        logger.info("Usage tracker initialized")
    
    def _load_history(self):
        """Load historical usage from file"""
        history_path = self.STORAGE_DIR / self.USAGE_FILE
        
        if not history_path.exists():
            return
        
        try:
            with open(history_path, 'r') as f:
                data = json.load(f)
            
            self.history = [
                UsageRecord.from_dict(r) 
                for r in data.get('records', [])
            ]
            
            # Trim to max history
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]
            
            logger.debug(f"Loaded {len(self.history)} historical records")
            
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
    
    def _save_history(self):
        """Save usage history to file"""
        if not self.persist:
            return
        
        self.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        history_path = self.STORAGE_DIR / self.USAGE_FILE
        
        try:
            # Combine history and session
            all_records = self.history + self.session_records
            
            # Trim to max
            if len(all_records) > self.max_history:
                all_records = all_records[-self.max_history:]
            
            data = {
                'records': [r.to_dict() for r in all_records],
                'last_updated': time.time()
            }
            
            with open(history_path, 'w') as f:
                json.dump(data, f)
                
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    
    def record(
        self,
        usage: UsageRecord = None,
        provider: str = "",
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cost_usd: float = 0.0,
        generation_time: float = 0.0,
        query: str = ""
    ):
        """
        Record a usage event.
        
        Can pass UsageRecord directly or individual parameters.
        """
        if usage is None:
            usage = UsageRecord(
                timestamp=time.time(),
                provider=provider,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens or (prompt_tokens + completion_tokens),
                cost_usd=cost_usd,
                generation_time=generation_time,
                query_preview=query[:50] if query else ""
            )
        
        self.session_records.append(usage)
        
        # Auto-save periodically
        if len(self.session_records) % 10 == 0:
            self._save_history()
        
        logger.debug(f"Recorded usage", {
            "provider": usage.provider,
            "tokens": usage.total_tokens
        })
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for current session"""
        if not self.session_records:
            return {
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "request_count": 0,
                "session_duration_minutes": 0,
                "by_provider": {},
                "by_model": {}
            }
        
        total_tokens = sum(r.total_tokens for r in self.session_records)
        total_cost = sum(r.cost_usd for r in self.session_records)
        
        # By provider
        by_provider: Dict[str, Dict[str, Any]] = {}
        for r in self.session_records:
            if r.provider not in by_provider:
                by_provider[r.provider] = {"tokens": 0, "cost": 0.0, "count": 0}
            by_provider[r.provider]["tokens"] += r.total_tokens
            by_provider[r.provider]["cost"] += r.cost_usd
            by_provider[r.provider]["count"] += 1
        
        # By model
        by_model: Dict[str, Dict[str, Any]] = {}
        for r in self.session_records:
            if r.model not in by_model:
                by_model[r.model] = {"tokens": 0, "cost": 0.0, "count": 0}
            by_model[r.model]["tokens"] += r.total_tokens
            by_model[r.model]["cost"] += r.cost_usd
            by_model[r.model]["count"] += 1
        
        session_duration = (time.time() - self.session_start) / 60
        
        return {
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "request_count": len(self.session_records),
            "session_duration_minutes": round(session_duration, 1),
            "by_provider": by_provider,
            "by_model": by_model,
            "avg_tokens_per_request": total_tokens // len(self.session_records) if self.session_records else 0
        }
    
    def get_daily_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get daily statistics for last N days"""
        all_records = self.history + self.session_records
        
        if not all_records:
            return {"days": []}
        
        cutoff = time.time() - (days * 24 * 3600)
        recent = [r for r in all_records if r.timestamp > cutoff]
        
        # Group by day
        daily: Dict[str, Dict[str, Any]] = {}
        for r in recent:
            day = datetime.fromtimestamp(r.timestamp).strftime("%Y-%m-%d")
            if day not in daily:
                daily[day] = {"tokens": 0, "cost": 0.0, "count": 0}
            daily[day]["tokens"] += r.total_tokens
            daily[day]["cost"] += r.cost_usd
            daily[day]["count"] += 1
        
        return {
            "days": [
                {"date": day, **stats}
                for day, stats in sorted(daily.items())
            ],
            "total_tokens": sum(d["tokens"] for d in daily.values()),
            "total_cost": sum(d["cost"] for d in daily.values()),
            "total_requests": sum(d["count"] for d in daily.values())
        }
    
    def export_csv(self, path: str = None) -> str:
        """
        Export usage to CSV.
        
        Args:
            path: Output path (default: usage_export.csv in current dir)
            
        Returns:
            Path to exported file
        """
        import csv
        
        path = path or f"usage_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        all_records = self.history + self.session_records
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'timestamp', 'datetime', 'provider', 'model',
                'prompt_tokens', 'completion_tokens', 'total_tokens',
                'cost_usd', 'generation_time', 'query_preview'
            ])
            
            # Data
            for r in all_records:
                dt = datetime.fromtimestamp(r.timestamp).isoformat()
                writer.writerow([
                    r.timestamp, dt, r.provider, r.model,
                    r.prompt_tokens, r.completion_tokens, r.total_tokens,
                    r.cost_usd, r.generation_time, r.query_preview
                ])
        
        logger.info(f"Exported {len(all_records)} records to {path}")
        return path
    
    def clear_session(self):
        """Clear current session records"""
        self.session_records = []
        self.session_start = time.time()
        logger.info("Session cleared")
    
    def save_and_close(self):
        """Save history before shutdown"""
        self._save_history()
        logger.info("Usage history saved")


# Global instance
_usage_tracker: Optional[UsageTracker] = None


def get_usage_tracker() -> UsageTracker:
    """Get global usage tracker instance"""
    global _usage_tracker
    if _usage_tracker is None:
        _usage_tracker = UsageTracker()
    return _usage_tracker
