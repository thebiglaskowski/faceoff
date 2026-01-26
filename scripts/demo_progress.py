"""
Quick demo of terminal progress tracking.
Run this to see how progress bars will look during processing.
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import from utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from utils.progress import get_progress_tracker, create_stage_tracker

def demo_simple_progress():
    """Demo simple progress tracking."""
    print("\n" + "="*70)
    print("DEMO 1: Simple Progress Bar")
    print("="*70 + "\n")
    
    progress = get_progress_tracker()
    
    with progress.track(100, "Processing frames", "frame") as pbar:
        for i in range(100):
            time.sleep(0.02)  # Simulate work
            pbar.update(1)
            if i % 20 == 0:
                pbar.set_postfix(batch=f"{i//20 + 1}/5")


def demo_stage_tracking():
    """Demo multi-stage progress tracking."""
    print("\n" + "="*70)
    print("DEMO 2: Multi-Stage Progress (like video processing)")
    print("="*70 + "\n")
    
    tracker = create_stage_tracker(
        stages=["Face Detection", "Face Swapping", "Enhancement"],
        total_items=50,
        desc="Video processing"
    )
    
    # Stage 1: Detection
    with tracker.stage("Face Detection"):
        for i in range(50):
            time.sleep(0.03)
            tracker.update(1)
            tracker.set_postfix(faces=f"{i+1}")
    
    # Stage 2: Swapping
    with tracker.stage("Face Swapping"):
        for i in range(50):
            time.sleep(0.04)
            tracker.update(1)
            if i % 10 == 0:
                tracker.set_postfix(batch=f"{i//10 + 1}/5")
    
    # Stage 3: Enhancement
    with tracker.stage("Enhancement"):
        for i in range(50):
            time.sleep(0.02)
            tracker.update(1)


def demo_with_logging():
    """Demo progress with logging messages."""
    print("\n" + "="*70)
    print("DEMO 3: Progress with Logging (logs won't disrupt progress)")
    print("="*70 + "\n")
    
    progress = get_progress_tracker()
    progress.set_stage("Processing with Logs")
    
    with progress.track(30, "Processing items", "item") as pbar:
        for i in range(30):
            time.sleep(0.1)
            pbar.update(1)
            
            # Simulate occasional log messages
            if i == 10:
                progress.log(f"✓ Checkpoint reached at item {i}")
            elif i == 20:
                progress.log("⚠ Warning: High memory usage detected", "warning")


if __name__ == "__main__":
    print("\n🚀 FaceOff Terminal Progress Tracking Demo")
    print("This shows how progress will appear when processing media in terminal\n")
    
    demo_simple_progress()
    time.sleep(1)
    
    demo_stage_tracking()
    time.sleep(1)
    
    demo_with_logging()
    
    print("\n" + "="*70)
    print("✅ Demo Complete!")
    print("="*70)
    print("\nThese progress bars will automatically appear when running FaceOff")
    print("from a terminal, and stay hidden when using Gradio UI.\n")
