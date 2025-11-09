"""
Terminal progress tracking for FaceOff processing.
Provides rich progress bars and stage indicators without interfering with Gradio UI.
"""
import logging
import sys
from typing import Optional, List
from contextlib import contextmanager

logger = logging.getLogger("FaceOff")

# Try to import tqdm, fallback gracefully if not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm not available - progress bars disabled")


class ProgressTracker:
    """
    Terminal-only progress tracking for long-running operations.
    Automatically detects if running in terminal or Gradio context.
    """
    
    def __init__(self, disable_in_ui: bool = True):
        """
        Initialize progress tracker.
        
        Args:
            disable_in_ui: If True, disables progress bars when not in terminal
        """
        self.disable_in_ui = disable_in_ui
        self._active_bars = []
        
    def _is_terminal(self) -> bool:
        """Check if running in an interactive terminal."""
        return sys.stdout.isatty() and sys.stderr.isatty()
    
    def _should_show(self) -> bool:
        """Determine if progress bars should be shown."""
        if not TQDM_AVAILABLE:
            return False
        if self.disable_in_ui and not self._is_terminal():
            return False
        return True
    
    @contextmanager
    def track(
        self,
        total: int,
        desc: str = "Processing",
        unit: str = "frame",
        stage: Optional[str] = None
    ):
        """
        Context manager for tracking progress.
        
        Args:
            total: Total number of items to process
            desc: Description of the operation
            unit: Unit name (e.g., "frame", "image", "file")
            stage: Current processing stage (e.g., "Detection", "Swapping", "Enhancement")
            
        Example:
            with progress.track(100, "Swapping faces", "frame", "Face Swapping") as pbar:
                for i in range(100):
                    # ... do work ...
                    pbar.update(1)
        """
        if self._should_show():
            # Format description with stage if provided
            full_desc = f"[{stage}] {desc}" if stage else desc
            
            pbar = tqdm(
                total=total,
                desc=full_desc,
                unit=unit,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                ncols=100,
                file=sys.stderr  # Use stderr to avoid interfering with stdout
            )
            self._active_bars.append(pbar)
            try:
                yield pbar
            finally:
                pbar.close()
                self._active_bars.remove(pbar)
        else:
            # Null progress bar that does nothing
            yield NullProgressBar()
    
    def set_stage(self, stage: str):
        """
        Log the current processing stage.
        
        Args:
            stage: Stage name (e.g., "Face Detection", "Face Swapping", "Enhancement")
        """
        if self._should_show():
            tqdm.write(f"\n{'='*60}", file=sys.stderr)
            tqdm.write(f"  {stage}", file=sys.stderr)
            tqdm.write(f"{'='*60}\n", file=sys.stderr)
        else:
            logger.info(f"Stage: {stage}")
    
    def log(self, message: str, level: str = "info"):
        """
        Log a message without disrupting progress bars.
        
        Args:
            message: Message to log
            level: Log level (info, warning, error)
        """
        if self._should_show():
            tqdm.write(message, file=sys.stderr)
        else:
            getattr(logger, level)(message)


class NullProgressBar:
    """Null object pattern for progress bar when disabled."""
    
    def update(self, n: int = 1):
        """No-op update."""
        pass
    
    def set_description(self, desc: str):
        """No-op set description."""
        pass
    
    def set_postfix(self, **kwargs):
        """No-op set postfix."""
        pass
    
    def close(self):
        """No-op close."""
        pass


# Global progress tracker instance
_progress_tracker = None


def get_progress_tracker() -> ProgressTracker:
    """Get the global ProgressTracker instance."""
    global _progress_tracker
    if _progress_tracker is None:
        _progress_tracker = ProgressTracker()
    return _progress_tracker


def create_stage_tracker(stages: List[str], total_items: int, desc: str = "Processing"):
    """
    Create a multi-stage progress tracker.
    
    Args:
        stages: List of stage names
        total_items: Total items across all stages
        desc: Overall description
        
    Returns:
        StageTracker instance
        
    Example:
        tracker = create_stage_tracker(
            ["Detection", "Swapping", "Enhancement"],
            total_items=100,
            desc="Video processing"
        )
        
        with tracker.stage("Detection"):
            for i in range(100):
                # ... detect faces ...
                tracker.update(1)
        
        with tracker.stage("Swapping"):
            for i in range(100):
                # ... swap faces ...
                tracker.update(1)
    """
    return StageTracker(stages, total_items, desc)


class StageTracker:
    """Multi-stage progress tracker with overall and per-stage progress."""
    
    def __init__(self, stages: List[str], total_items: int, desc: str = "Processing"):
        """
        Initialize stage tracker.
        
        Args:
            stages: List of stage names
            total_items: Total items to process across all stages
            desc: Overall description
        """
        self.stages = stages
        self.total_items = total_items
        self.desc = desc
        self.current_stage_idx = -1
        self.progress = get_progress_tracker()
        self._stage_pbar = None
        self._overall_pbar = None
        
    @contextmanager
    def stage(self, stage_name: str):
        """
        Context manager for a processing stage.
        
        Args:
            stage_name: Name of the stage
        """
        self.current_stage_idx += 1
        self.progress.set_stage(f"Stage {self.current_stage_idx + 1}/{len(self.stages)}: {stage_name}")
        
        if self.progress._should_show():
            self._stage_pbar = tqdm(
                total=self.total_items,
                desc=f"  {stage_name}",
                unit="item",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                ncols=100,
                file=sys.stderr,
                leave=True
            )
            try:
                yield self
            finally:
                if self._stage_pbar:
                    self._stage_pbar.close()
                    self._stage_pbar = None
        else:
            yield self
    
    def update(self, n: int = 1):
        """Update the current stage progress."""
        if self._stage_pbar:
            self._stage_pbar.update(n)
    
    def set_postfix(self, **kwargs):
        """Set postfix info on current stage progress bar."""
        if self._stage_pbar:
            self._stage_pbar.set_postfix(**kwargs)
