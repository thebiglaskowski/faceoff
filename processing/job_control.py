"""Cooperative cancellation for long swap jobs.

One process-wide flag (same pattern as the Models download cancel):
the UI Cancel button sets it, chunk loops and stage boundaries poll it
via raise_if_cancelled(). Setup errors are never swallowed — only a
user cancel raises JobCancelled.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger("FaceOff")


class JobCancelled(Exception):
    """Raised when the user cancels a running swap job."""


_cancel_job = threading.Event()


def request_job_cancel() -> None:
    """Signal the running swap job (if any) to stop at the next checkpoint."""
    _cancel_job.set()


def clear_job_cancel() -> None:
    """Clear a stale cancel signal at job start."""
    _cancel_job.clear()


def job_cancel_requested() -> bool:
    return _cancel_job.is_set()


def raise_if_cancelled() -> None:
    """Raise JobCancelled when the user asked to stop. Cheap: call per chunk."""
    if _cancel_job.is_set():
        logger.info("Swap job cancelled by user")
        raise JobCancelled("Swap cancelled by user.")
