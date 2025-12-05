"""
Rate Limiting Middleware - Simple implementation using in-memory storage

For production use, consider Redis-based rate limiting for distributed systems.
"""

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Tuple
import time
from collections import defaultdict
import threading


class SimpleRateLimiter(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiter using sliding window algorithm.

    For single-server deployments. For multi-server, use Redis-based limiter.
    """

    def __init__(self, app, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour

        # Storage: {client_ip: [(timestamp, window_type), ...]}
        self._request_history: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()

        # Cleanup old entries every 5 minutes
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes

    def _cleanup_old_entries(self):
        """Remove entries older than 1 hour to prevent memory growth"""
        current_time = time.time()

        # Only cleanup periodically
        if current_time - self._last_cleanup < self._cleanup_interval:
            return

        with self._lock:
            one_hour_ago = current_time - 3600

            # Clean up old entries for each client
            for client_ip in list(self._request_history.keys()):
                self._request_history[client_ip] = [
                    (ts, window) for ts, window in self._request_history[client_ip]
                    if ts > one_hour_ago
                ]

                # Remove client if no recent requests
                if not self._request_history[client_ip]:
                    del self._request_history[client_ip]

            self._last_cleanup = current_time

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check for proxy headers first
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to direct client
        return request.client.host if request.client else "unknown"

    def _is_rate_limited(self, client_ip: str) -> Tuple[bool, str]:
        """
        Check if client has exceeded rate limits.

        Returns:
            (is_limited, reason)
        """
        current_time = time.time()
        one_minute_ago = current_time - 60
        one_hour_ago = current_time - 3600

        with self._lock:
            history = self._request_history[client_ip]

            # Count requests in the last minute
            minute_requests = sum(1 for ts, _ in history if ts > one_minute_ago)
            if minute_requests >= self.requests_per_minute:
                return True, f"Rate limit exceeded: {self.requests_per_minute} requests per minute"

            # Count requests in the last hour
            hour_requests = sum(1 for ts, _ in history if ts > one_hour_ago)
            if hour_requests >= self.requests_per_hour:
                return True, f"Rate limit exceeded: {self.requests_per_hour} requests per hour"

            return False, ""

    def _record_request(self, client_ip: str):
        """Record a request for the client"""
        current_time = time.time()

        with self._lock:
            self._request_history[client_ip].append((current_time, 'request'))

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting"""

        # Skip rate limiting for health checks
        if request.url.path in ["/api/v1/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        # Periodic cleanup
        self._cleanup_old_entries()

        # Get client IP
        client_ip = self._get_client_ip(request)

        # Check rate limit
        is_limited, reason = self._is_rate_limited(client_ip)

        if is_limited:
            raise HTTPException(
                status_code=429,
                detail=reason,
                headers={"Retry-After": "60"}
            )

        # Record request
        self._record_request(client_ip)

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        with self._lock:
            history = self._request_history[client_ip]
            minute_requests = sum(1 for ts, _ in history if ts > time.time() - 60)
            hour_requests = sum(1 for ts, _ in history if ts > time.time() - 3600)

        response.headers["X-RateLimit-Limit-Minute"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining-Minute"] = str(max(0, self.requests_per_minute - minute_requests))
        response.headers["X-RateLimit-Limit-Hour"] = str(self.requests_per_hour)
        response.headers["X-RateLimit-Remaining-Hour"] = str(max(0, self.requests_per_hour - hour_requests))

        return response
