"""
Performance profiling utilities for FaceOff.

This module provides decorators and context managers for
measuring execution time of functions and code blocks.
"""

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("FaceOff")


class ProfilingStats:
    """Collects and reports profiling statistics."""

    def __init__(self):
        """Initialize profiling stats collector."""
        self._timings: Dict[str, List[float]] = {}
        self._enabled = True

    def enable(self):
        """Enable profiling."""
        self._enabled = True

    def disable(self):
        """Disable profiling."""
        self._enabled = False

    def record(self, name: str, duration: float):
        """Record a timing measurement."""
        if not self._enabled:
            return

        if name not in self._timings:
            self._timings[name] = []
        self._timings[name].append(duration)

    def clear(self):
        """Clear all recorded timings."""
        self._timings.clear()

    def get_stats(self, name: str) -> Optional[Dict[str, float]]:
        """
        Get statistics for a named operation.

        Returns:
            Dict with 'count', 'total', 'mean', 'min', 'max', or None if no data
        """
        if name not in self._timings or not self._timings[name]:
            return None

        timings = self._timings[name]
        return {
            'count': len(timings),
            'total': sum(timings),
            'mean': sum(timings) / len(timings),
            'min': min(timings),
            'max': max(timings),
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all recorded operations."""
        return {name: self.get_stats(name) for name in self._timings}

    def report(self, threshold_ms: float = 0) -> str:
        """
        Generate a profiling report.

        Args:
            threshold_ms: Only include operations with mean > threshold

        Returns:
            Formatted report string
        """
        all_stats = self.get_all_stats()
        if not all_stats:
            return "No profiling data recorded."

        lines = ["Performance Profile:"]
        lines.append("-" * 70)
        lines.append(f"{'Operation':<35} {'Count':>8} {'Total':>10} {'Mean':>10} {'Max':>10}")
        lines.append("-" * 70)

        for name, stats in sorted(all_stats.items(), key=lambda x: x[1]['total'], reverse=True):
            if stats['mean'] * 1000 < threshold_ms:
                continue

            lines.append(
                f"{name:<35} {stats['count']:>8} {stats['total']:>9.2f}s "
                f"{stats['mean']*1000:>9.1f}ms {stats['max']*1000:>9.1f}ms"
            )

        lines.append("-" * 70)
        return "\n".join(lines)

    def log_report(self, threshold_ms: float = 0):
        """Log the profiling report."""
        logger.info("\n%s", self.report(threshold_ms))


# Global profiling stats instance
_global_stats = ProfilingStats()


def get_profiler() -> ProfilingStats:
    """Get the global profiler instance."""
    return _global_stats


def profile(name: Optional[str] = None):
    """
    Decorator to profile function execution time.

    Args:
        name: Optional name for the operation (uses function name if not provided)

    Usage:
        @profile()
        def my_function():
            ...

        @profile("custom_name")
        def other_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        operation_name = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                _global_stats.record(operation_name, duration)
                logger.debug("⏱ %s took %.3fs", operation_name, duration)

        return wrapper

    return decorator


@contextmanager
def profile_block(name: str):
    """
    Context manager to profile a code block.

    Usage:
        with profile_block("my_operation"):
            # code to profile
            ...
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        _global_stats.record(name, duration)
        logger.debug("⏱ %s took %.3fs", name, duration)


class Profiler:
    """
    Profiler context manager with more detailed control.

    Usage:
        profiler = Profiler("operation_name")
        with profiler:
            # code to profile
        print(f"Took {profiler.duration:.3f}s")
    """

    def __init__(self, name: str, log_on_exit: bool = True):
        """
        Initialize profiler.

        Args:
            name: Name for this profiling operation
            log_on_exit: Whether to log timing on exit
        """
        self.name = name
        self.log_on_exit = log_on_exit
        self.start_time: Optional[float] = None
        self.duration: Optional[float] = None

    def __enter__(self):
        """Start profiling."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling and record results."""
        self.duration = time.perf_counter() - self.start_time
        _global_stats.record(self.name, self.duration)

        if self.log_on_exit:
            logger.debug("⏱ %s took %.3fs", self.name, self.duration)

        return False

    def elapsed(self) -> float:
        """Get elapsed time since start (while still running)."""
        if self.start_time is None:
            return 0
        return time.perf_counter() - self.start_time


def reset_profiler():
    """Reset the global profiler stats."""
    _global_stats.clear()
