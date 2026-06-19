"""
Video and GIF I/O utilities using direct FFmpeg subprocess calls.

Replaces moviepy for video/GIF read, frame extraction, and video writing.
All operations are subprocess-based; no Python video codec dependencies.
"""
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger("FaceOff")


# =============================================================================
# Video/GIF probe (metadata only -- no decoding)
# =============================================================================

def probe_video(video_path: str) -> dict:
    """
    Probe video metadata using ffprobe --show_entries.

    Returns dict with:
        fps, duration, width, height, has_audio, audio_codec,
        video_codec, video_idx, audio_idx
    """
    video_path = str(video_path)
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v",
            "-show_entries",
            "stream=width,height,codec_name,r_frame_rate,avg_frame_rate,index",
            "-show_entries",
            "format=duration",
            "-of", "json",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.error("ffprobe failed: %s", result.stderr.strip())
            return {}

        raw = result.stdout.strip()
        # ffprobe JSON output -- parse manually for reliability
        # Extract duration
        dur_match = re.search(r'"duration"\s*:\s*"([^"]+)"', raw)
        duration = float(dur_match.group(1)) if dur_match else 0.0

        # Extract stream info
        # width/hight/codec_name/r_frame_rate/avg_frame_rate/index
        streams = re.findall(r'\{[^}]+\}', raw)
        video_stream = streams[0] if streams else "{}"

        def _json_val(obj, key):
            m = re.search(rf'"{key}"\s*:\s*"([^"]+)"', obj)
            if m:
                return m.group(1)
            m = re.search(rf'"{key}"\s*:\s*([0-9.]+)', obj)
            if m:
                return float(m.group(1))
            return None

        width = int(_json_val(video_stream, "width") or 0)
        height = int(_json_val(video_stream, "height") or 0)
        video_codec = _json_val(video_stream, "codec_name") or ""
        r_frame = _json_val(video_stream, "r_frame_rate") or ""
        video_idx = int(_json_val(video_stream, "index") or -1)

        # Calculate fps from fraction string "numerator/denominator"
        fps = 0.0
        if r_frame and "/" in r_frame:
            num, den = r_frame.split("/")
            num, den = int(num), int(den)
            if den > 0:
                fps = num / den
        else:
            try:
                fps = float(r_frame)
            except (ValueError, TypeError):
                pass

        # Check for audio stream
        has_audio = False
        audio_codec = ""
        try:
            cmd2 = [
                "ffprobe", "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=codec_name",
                "-of", "json", video_path,
            ]
            res2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=15)
            if res2.returncode == 0 and "codec_name" in res2.stdout:
                has_audio = True
                m = re.search(r'"codec_name"\s*:\s*"([^"]+)"', res2.stdout)
                if m:
                    audio_codec = m.group(1)
        except Exception:
            pass

        return {
            "fps": fps,
            "duration": duration,
            "width": width,
            "height": height,
            "has_audio": has_audio,
            "audio_codec": audio_codec,
            "video_codec": video_codec,
            "video_idx": video_idx,
        }

    except FileNotFoundError:
        logger.error("ffprobe not found on PATH")
        return {}
    except Exception as e:
        logger.error("ffprobe probe error: %s", e)
        return {}


def probe_gif(gif_path: str) -> dict:
    """
    Same interface as probe_video but works for GIF files too.
    """
    return probe_video(gif_path)


# =============================================================================
# Frame extraction (video -> frames)
# =============================================================================

@dataclass
class FrameResult:
    path: Path
    frame_index: int
    pts_sec: float


def extract_video_frames(
    video_path: str,
    output_dir: Union[str, Path],
    start_time: float = 0.0,
    duration: Optional[float] = None,
    fps: Optional[float] = None,
) -> List[FrameResult]:
    """
    Extract frames from video using FFmpeg.

    Writes one PNG per frame to output_dir. Returns sorted list of
    FrameResult objects with path, frame_index, and pts_sec.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = probe_video(video_path)
    if fps is None:
        fps = float(meta.get("fps") or 30.0)

    if fps <= 0:
        fps = 30.0

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
        ]
        cmd.extend(["-ss", str(start_time)])
        if duration:
            cmd.extend(["-t", str(duration)])
        cmd.extend([
            "-vf", f"fps={fps}",
            "-q:v", "3",
            str(output_dir / "frame_%06d.png"),
        ])

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            logger.error(
                "FFmpeg extract frames failed (rc=%d): %s",
                result.returncode, result.stderr.strip()[:500],
            )
            return []

        frames = []
        for i, f in enumerate(sorted(output_dir.glob("frame_*.png"))):
            frames.append(FrameResult(
                path=f, frame_index=i, pts_sec=i / fps
            ))

        logger.info(
            "Extracted %d frames from video (%.2f fps)",
            len(frames), fps,
        )
        return frames

    except FileNotFoundError:
        logger.error("ffmpeg not found on PATH")
        return []
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg extract frames timed out")
        return []
    except Exception as e:
        logger.error("FFmpeg extract frames error: %s", e)
        return []


def extract_video_frames_raw(
    video_path: str,
    output_dir: Union[str, Path],
    start_time: float = 0.0,
    duration: Optional[float] = None,
    fmt: str = "yuv420p",
) -> Path:
    """
    Extract raw YUV frame sequence into a single binary file.

    Much faster than PNG per-frame when all you need is raw pixel data.
    Callers must parse the binary blob.

    Returns path to the raw file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "frames.raw"

    meta = probe_video(video_path)
    height = int(meta.get("height") or 480)
    width = int(meta.get("width") or 640)
    fps = float(meta.get("fps") or 30.0)
    stride = ((width + 31) // 32) * 32  # align to 32px

    try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", video_path,
        ]
        if duration:
            cmd.extend(["-t", str(duration)])
        cmd.extend([
            "-vf", f"fps={fps}",
            "-pix_fmt", fmt,
            "-s", f"{width}x{height}",
            str(raw_path),
        ])

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            logger.error(
                "FFmpeg raw extract failed: %s", result.stderr.strip()[:500]
            )
            return Path()

        frame_count = 0
        if raw_path.exists() and raw_path.stat().st_size > 0:
            # Estimate frame count from file size
            bytes_per_frame = stride * height + stride * height // 4 + stride * height // 4
            if bytes_per_frame > 0:
                frame_count = int(raw_path.stat().st_size / bytes_per_frame)

        logger.info(
            "Extracted %d raw frames (%d bytes) from video",
            frame_count, raw_path.stat().st_size if raw_path.exists() else 0,
        )
        return raw_path

    except FileNotFoundError:
        logger.error("ffmpeg not found on PATH")
    except Exception as e:
        logger.error("FFmpeg raw extract error: %s", e)
    return Path()


# =============================================================================
# Video writing (frames -> video)
# =============================================================================

def write_video_from_pil_frames(
    frames,
    output_path: Union[str, Path],
    fps: float,
    audio_path: Optional[Union[str, Path]] = None,
    codec: str = "libx264",
    preset: str = "medium",
    crf: int = 18,
    pix_fmt: str = "yuv420p",
) -> bool:
    """
    Write video from PIL Image frames to a temp dir, then run FFmpeg.

    Args:
        frames: List of PIL Image objects (RGB or RGBA).
        output_path: Destination .mp4 path.
        fps: Frames per second.
        audio_path: Optional audio file to mux.
        codec: Video codec.
        preset: FFmpeg preset.
        crf: Quality (18-28).
        pix_fmt: Output pixel format.

    Returns:
        True on success, False on failure.
    """
    from PIL import Image

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        logger.error("No frames to write video")
        return False

    # Write frames to temp dir as PNGs
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        frame_paths = []
        for i, frame in enumerate(frames):
            fpath = tmpdir / f"frame_{i:06d}.png"
            # Ensure RGB for PNG
            if isinstance(frame, np.ndarray):
                # If it's a numpy array, it's already in RGB format
                pil_frame = Image.fromarray(
                    np.clip(frame, 0, 255).astype(np.uint8)
                )
            else:
                pil_frame = Image.fromarray(np.array(frame))
            pil_frame = pil_frame.convert("RGB")
            pil_frame.save(fpath)
            frame_paths.append(fpath)

        logger.info("Wrote %d frames to temp dir", len(frame_paths))

        # Write video with FFmpeg
        try:
            cmd = ["ffmpeg", "-y", "-framerate", str(fps)]
            cmd.extend(["-i", str(tmpdir / "frame_%06d.png")])
            cmd.extend([
                "-c:v", codec,
                "-preset", preset,
                "-crf", str(crf),
                "-pix_fmt", pix_fmt,
            ])

            if audio_path:
                cmd.extend(["-i", str(audio_path), "-c:a", "aac", "-b:a", "192k", "-map", "0:v:0", "-map", "1:a:0", "-shortest"])

            cmd.append(str(output_path))

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                logger.error(
                    "FFmpeg write video failed: %s", result.stderr.strip()[:500]
                )
                return False

            logger.info("Wrote video: %d frames, %s", len(frame_paths), output_path)
            return True

        except FileNotFoundError:
            logger.error("ffmpeg not found on PATH")
            return False
        except Exception as e:
            logger.error("FFmpeg write video error: %s", e)
            return False


def write_video_from_frames(
    frames,
    output_path: Union[str, Path],
    fps: float,
    audio_path:Optional[Union[str, Path]] = None,
    codec: str = "libx264",
    preset: str = "medium",
    crf: int = 18,
    pix_fmt: str = "yuv420p",
) -> bool:
    """
    Write video from numpy frames (RGB arrays) or PIL Images.

    Same interface as write_video_from_pil_frames but accepts numpy arrays too.
    """
    return write_video_from_pil_frames(
        frames=frames,
        output_path=output_path,
        fps=fps,
        audio_path=audio_path,
        codec=codec,
        preset=preset,
        crf=crf,
        pix_fmt=pix_fmt,
    )


# =============================================================================
# Audio extraction
# =============================================================================

def extract_audio(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    codec: str = "aac",
    bitrate: str = "192k",
) -> Optional[Path]:
    """
    Extract audio track from video as AAC.

    Returns path to the extracted .aac file, or None if no audio / failed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / "audio_extracted.aac"

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-map", "0:a?",
            "-c:a", codec,
            "-b:a", bitrate,
            str(audio_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0 or not audio_path.exists():
            logger.warning("Audio extraction failed: %s", result.stderr.strip()[:200])
            return None

        logger.info("Extracted audio to %s", audio_path)
        return audio_path

    except FileNotFoundError:
        logger.error("ffmpeg not found on PATH")
        return None
    except Exception as e:
        logger.error("Audio extract error: %s", e)
        return None


# =============================================================================
# GIF frame duration extraction
# =============================================================================

def extract_gif_frame_durations(gif_path: str) -> List[int]:
    """
    Extract frame durations from a GIF (in ms) using ffprobe.

    For GIFs, frame time is stored in the 'time_subscript' field.
    Returns list of durations in milliseconds.
    """
    import PIL.Image as PILImage
    durations = []

    try:
        gif = PILImage.open(gif_path)
        if hasattr(gif, 'n_frames'):
            for i in range(gif.n_frames):
                gif.seek(i)
                duration = gif.info.get('duration', 100)
                durations.append(int(duration))
            gif.close()
            return durations
    except Exception as e:
        logger.warning("GIF duration extraction failed: %s", e)

    # Fallback: return default 100ms per frame based on fps
    meta = probe_gif(gif_path)
    fps = meta.get("fps", 10.0) or 10.0
    if fps <= 0:
        fps = 10.0
    default_ms = int(1000 / fps)
    # Estimate frame count from GIF size
    return [default_ms] * 10  # Placeholder


# =============================================================================
# GIF writing
# =============================================================================

def write_gif(
    frames,
    output_path: Union[str, Path],
    durations: Optional[List[int]] = None,
    loop: int = 0,
    fps: Optional[float] = None,
) -> bool:
    """
    Write a GIF from PIL Images or numpy arrays, preserving per-frame durations.

    If durations are provided, they must match len(frames).
    Otherwise, frame time is approximated from fps.

    Returns True on success.
    """
    from PIL import Image

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        logger.error("No frames to write GIF")
        return False

    # Convert all frames to PIL Images
    pil_frames = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            pil_frame = Image.fromarray(
                np.clip(frame, 0, 255).astype(np.uint8), mode='RGBA' if frame.shape[2] == 4 else 'RGB'
            )
        else:
            pil_frame = Image.fromarray(np.array(frame), mode='RGBA' if frame.mode == 'RGBA' else 'RGB')

        if pil_frame.mode != 'RGBA':
            pil_frame = pil_frame.convert('RGBA')

        pil_frames.append(pil_frame)

    try:
        if durations and len(durations) == len(pil_frames):
            pil_frames[0].save(
                str(output_path),
                save_all=True,
                append_images=pil_frames[1:],
                loop=loop,
                duration=durations,
                optimize=True,
            )
        else:
            # Use approximate frame time
            if fps and fps > 0:
                frame_time = int(1000 / fps)
            else:
                frame_time = 100

            pil_frames[0].save(
                str(output_path),
                save_all=True,
                append_images=pil_frames[1:],
                loop=loop,
                duration=frame_time,
                optimize=True,
            )

        logger.info("Wrote GIF: %s (%d frames)", output_path, len(pil_frames))
        return True

    except Exception as e:
        logger.error("Failed to write GIF: %s", e)
        return False


# =============================================================================
# Helpers
# =============================================================================

def get_video_info(video_path: str) -> dict:
    """Convenience function. Returns same dict as probe_video."""
    return probe_video(video_path)
