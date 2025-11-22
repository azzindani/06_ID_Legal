"""
Analytics Dashboard

Tracks and displays usage metrics, query statistics, and performance data.

File: core/analytics.py
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json
import os
from logger_utils import get_logger


class AnalyticsDashboard:
    """
    Tracks and reports system analytics
    """

    def __init__(self, data_dir: str = ".analytics"):
        self.logger = get_logger("Analytics")
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # In-memory metrics
        self._session_start = datetime.now()
        self._queries: List[Dict[str, Any]] = []
        self._errors: List[Dict[str, Any]] = []
        self._performance_samples: List[Dict[str, Any]] = []

        # Aggregates
        self._query_types = defaultdict(int)
        self._provider_usage = defaultdict(int)
        self._response_times: List[float] = []

    def track_query(
        self,
        query: str,
        query_type: str,
        response_time: float,
        success: bool,
        provider: str = "local",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track a query execution"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],  # Truncate for storage
            "query_type": query_type,
            "response_time": response_time,
            "success": success,
            "provider": provider,
            "metadata": metadata or {}
        }

        self._queries.append(record)
        self._query_types[query_type] += 1
        self._provider_usage[provider] += 1
        self._response_times.append(response_time)

        self.logger.debug("Query tracked", {
            "query_type": query_type,
            "response_time": f"{response_time:.2f}s"
        })

    def track_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Track an error occurrence"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "message": error_message,
            "context": context or {}
        }

        self._errors.append(record)
        self.logger.debug("Error tracked", {"type": error_type})

    def track_performance(
        self,
        component: str,
        operation: str,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track component performance"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "operation": operation,
            "duration": duration,
            "metadata": metadata or {}
        }

        self._performance_samples.append(record)

    def get_summary(self) -> Dict[str, Any]:
        """Get analytics summary"""
        session_duration = (datetime.now() - self._session_start).total_seconds()

        # Calculate response time stats
        if self._response_times:
            avg_response = sum(self._response_times) / len(self._response_times)
            min_response = min(self._response_times)
            max_response = max(self._response_times)
        else:
            avg_response = min_response = max_response = 0

        return {
            "session": {
                "start_time": self._session_start.isoformat(),
                "duration_seconds": session_duration,
                "duration_formatted": str(timedelta(seconds=int(session_duration)))
            },
            "queries": {
                "total": len(self._queries),
                "successful": sum(1 for q in self._queries if q["success"]),
                "failed": sum(1 for q in self._queries if not q["success"]),
                "by_type": dict(self._query_types),
                "queries_per_minute": len(self._queries) / (session_duration / 60) if session_duration > 0 else 0
            },
            "performance": {
                "avg_response_time": round(avg_response, 3),
                "min_response_time": round(min_response, 3),
                "max_response_time": round(max_response, 3),
                "total_samples": len(self._performance_samples)
            },
            "providers": dict(self._provider_usage),
            "errors": {
                "total": len(self._errors),
                "by_type": self._get_error_counts()
            }
        }

    def get_query_history(
        self,
        limit: int = 50,
        query_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent query history"""
        queries = self._queries

        if query_type:
            queries = [q for q in queries if q["query_type"] == query_type]

        return queries[-limit:]

    def get_performance_report(
        self,
        component: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get detailed performance report"""
        samples = self._performance_samples

        if component:
            samples = [s for s in samples if s["component"] == component]

        if not samples:
            return {"message": "No performance data available"}

        # Group by component and operation
        by_component = defaultdict(lambda: defaultdict(list))
        for sample in samples:
            by_component[sample["component"]][sample["operation"]].append(
                sample["duration"]
            )

        # Calculate stats
        report = {}
        for comp, operations in by_component.items():
            report[comp] = {}
            for op, durations in operations.items():
                report[comp][op] = {
                    "count": len(durations),
                    "avg": round(sum(durations) / len(durations), 3),
                    "min": round(min(durations), 3),
                    "max": round(max(durations), 3)
                }

        return report

    def get_error_report(self) -> Dict[str, Any]:
        """Get error report"""
        if not self._errors:
            return {"message": "No errors recorded"}

        return {
            "total_errors": len(self._errors),
            "by_type": self._get_error_counts(),
            "recent": self._errors[-10:]
        }

    def _get_error_counts(self) -> Dict[str, int]:
        """Count errors by type"""
        counts = defaultdict(int)
        for error in self._errors:
            counts[error["error_type"]] += 1
        return dict(counts)

    def export_data(self, filepath: Optional[str] = None) -> str:
        """Export analytics data to JSON"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.data_dir, f"analytics_{timestamp}.json")

        data = {
            "exported_at": datetime.now().isoformat(),
            "summary": self.get_summary(),
            "queries": self._queries,
            "errors": self._errors,
            "performance": self._performance_samples
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Analytics exported to {filepath}")
        return filepath

    def reset(self):
        """Reset all analytics data"""
        self._session_start = datetime.now()
        self._queries = []
        self._errors = []
        self._performance_samples = []
        self._query_types = defaultdict(int)
        self._provider_usage = defaultdict(int)
        self._response_times = []

        self.logger.info("Analytics data reset")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get formatted data for dashboard display"""
        summary = self.get_summary()

        # Format for display
        return {
            "overview": {
                "session_duration": summary["session"]["duration_formatted"],
                "total_queries": summary["queries"]["total"],
                "success_rate": f"{(summary['queries']['successful'] / summary['queries']['total'] * 100):.1f}%" if summary['queries']['total'] > 0 else "N/A",
                "avg_response": f"{summary['performance']['avg_response_time']:.2f}s",
                "error_count": summary["errors"]["total"]
            },
            "charts": {
                "query_types": [
                    {"type": k, "count": v}
                    for k, v in summary["queries"]["by_type"].items()
                ],
                "providers": [
                    {"provider": k, "count": v}
                    for k, v in summary["providers"].items()
                ],
                "response_times": self._response_times[-50:],  # Last 50 samples
            },
            "recent_queries": self.get_query_history(10),
            "recent_errors": self._errors[-5:] if self._errors else []
        }


# Singleton instance
_analytics = None


def get_analytics() -> AnalyticsDashboard:
    """Get or create analytics singleton"""
    global _analytics
    if _analytics is None:
        _analytics = AnalyticsDashboard()
    return _analytics
