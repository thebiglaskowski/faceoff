# Terminal Progress Tracking

FaceOff now includes rich terminal progress bars that automatically appear when running from the command line, providing real-time feedback on processing status.

## Features

### 📊 Real-Time Progress Bars
- **Frame-by-frame progress**: See exact frame number and total frames
- **Time estimates**: Remaining time and processing speed (frames/sec)
- **Batch indicators**: Current batch out of total batches
- **Stage tracking**: Clear indicators for Detection → Swapping → Enhancement

### 🎯 Smart Detection
- **Auto-disables in Gradio**: Progress bars only show in terminal, never interfere with web UI
- **TTY detection**: Automatically detects terminal vs non-terminal environments
- **Graceful fallback**: Falls back to regular logging if tqdm unavailable

### 📈 Multi-Stage Tracking
For video and GIF processing, you'll see progress through multiple stages:

```
============================================================
  Stage 1/3: Face Detection
============================================================

  Face Detection: 100%|████████████| 240/240 [00:15<00:00, 15.8frame/s]

============================================================
  Stage 2/3: Face Swapping
============================================================

  Processing frames: 100%|█████████| 240/240 [01:30<00:00, 2.7frame/s] batch: 60/60

============================================================
  Stage 3/3: Enhancement
============================================================

  Enhancement: 100%|███████████████| 240/240 [03:45<00:00, 1.1frame/s]
```

## Example Output

### Single-GPU Video Processing
```bash
$ python main.py

============================================================
  Face Swapping
============================================================

Processing video frames: 100%|████| 500/500 [02:30<00:00, 3.3frame/s] batch: 125/125
```

### GIF Processing with Enhancement
```bash
Processing GIF frames: 100%|██████| 120/120 [00:45<00:00, 2.7frame/s] batch: 30/30

Enhancing frames: 100%|████████████| 120/120 [01:15<00:00, 1.6frame/s]
```

## Testing

Run the demo to see how progress bars will look:

```bash
cd G:\My Drive\scripts\faceoff
python test_progress.py
```

This will show:
1. Simple progress bar with frame counting
2. Multi-stage processing simulation
3. Progress with log messages

## Technical Details

### Implementation
- Uses `tqdm` library for progress bars (already in dependencies)
- Progress tracker in `utils/progress.py`
- Integrated into `processing/video_processing.py` and `processing/gif_processing.py`

### Components
- **`ProgressTracker`**: Main progress tracking class
- **`StageTracker`**: Multi-stage progress with overall tracking
- **`NullProgressBar`**: No-op implementation when progress disabled

### Configuration
Progress tracking is automatic and requires no configuration. It:
- ✅ Shows in terminal/console
- ❌ Hidden in Gradio web UI
- ❌ Hidden when redirecting output to files
- ❌ Hidden if tqdm not available (graceful fallback)

## Usage in Code

### Simple Progress
```python
from utils.progress import get_progress_tracker

progress = get_progress_tracker()

with progress.track(total_frames, "Processing", "frame", "Face Swapping") as pbar:
    for frame in frames:
        # Process frame...
        pbar.update(1)
        pbar.set_postfix(batch=f"{current_batch}/{total_batches}")
```

### Multi-Stage Progress
```python
from utils.progress import create_stage_tracker

tracker = create_stage_tracker(
    ["Detection", "Swapping", "Enhancement"],
    total_items=num_frames,
    desc="Video processing"
)

with tracker.stage("Face Detection"):
    for frame in frames:
        # Detect faces...
        tracker.update(1)

with tracker.stage("Face Swapping"):
    for frame in frames:
        # Swap faces...
        tracker.update(1)
```

### Logging Without Disruption
```python
progress = get_progress_tracker()

with progress.track(100, "Processing") as pbar:
    for i in range(100):
        # Work...
        pbar.update(1)
        
        # Log message without breaking progress bar
        if i == 50:
            progress.log("Halfway done!")
```

## Benefits

1. **Better UX**: Users know exactly what's happening and how long to wait
2. **No UI Pollution**: Automatically hidden in Gradio interface
3. **Performance Insights**: See processing speed in real-time
4. **Debugging**: Identify slow stages easily
5. **Professional**: Makes terminal output look polished and modern

---

**Note**: This feature is completely transparent. Existing code works without modification, and progress bars appear automatically when appropriate.
