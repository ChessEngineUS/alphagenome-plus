"""Rate limiter for API request management."""

import asyncio
import time
from collections import deque
from typing import Optional


class RateLimiter:
    """Token bucket rate limiter for API requests.
    
    Implements a token bucket algorithm to ensure requests don't exceed
    the specified rate limit while allowing short bursts.
    
    Example:
        >>> limiter = RateLimiter(max_requests_per_minute=60)
        >>> await limiter.acquire()  # Wait if necessary
        >>> # Make API call
    """
    
    def __init__(
        self,
        max_requests_per_minute: int = 60,
        burst_size: Optional[int] = None,
    ):
        """Initialize rate limiter.
        
        Args:
            max_requests_per_minute: Maximum sustained request rate
            burst_size: Maximum burst size (defaults to max_requests_per_minute)
        """
        self.rate = max_requests_per_minute / 60.0  # Requests per second
        self.burst_size = burst_size or max_requests_per_minute
        self.tokens = float(self.burst_size)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens, waiting if necessary.
        
        Args:
            tokens: Number of tokens to acquire (default: 1)
        """
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.last_update
                
                # Add tokens based on elapsed time
                self.tokens = min(
                    self.burst_size,
                    self.tokens + elapsed * self.rate
                )
                self.last_update = now
                
                # Check if we have enough tokens
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                
                # Wait until we'll have enough tokens
                wait_time = (tokens - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
    
    def reset(self) -> None:
        """Reset the rate limiter to full burst capacity."""
        self.tokens = float(self.burst_size)
        self.last_update = time.monotonic()