"""
Video and GIF I/O utilities using direct FFmpeg subprocess calls.

Replaces moviepy for video/GIF read, frame extraction, and video writing.
All operations are subprocess-based; no Python video codec dependencies.
"""
import logging
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger("FaceOff")


class _FfmpegStderrDrain:
    """
    Drain FFmpeg stderr in the background.

    If stderr is PIPE and nobody reads it, the ~64KB kernel buffer fills,
    FFmpeg blocks, and the Python side deadlocks on stdin/stdout I/O.
    """

    def __init__(
        self,
        proc: subprocess.Popen,
        *,
        label: str = "ffmpeg",
        max_bytes: int = 262144,
    ):
        self._buf = bytearray()
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        if proc.stderr is not None:
            self._thread = threading.Thread(
                target=self._run,
                args=(proc.stderr,),
                name=f"{label}-stderr-drain",
                daemon=True,
            )
            self._thread.start()

    def _run(self, stderr) -> None:
        try:
            while True:
                chunk = stderr.read(4096)
                if not chunk:
                    break
                with self._lock:
                    if len(self._buf) < self._max_bytes:
                        room = self._max_bytes - len(self._buf)
                        self._buf.extend(chunk[:room])
        finally:
            try:
                stderr.close()
            except OSError:
                pass

    def get_text(self) -> str:
        with self._lock:
            return bytes(self._buf).decode("utf-8", errors="replace")

    def join(self, timeout: float = 2.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)


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
# Streaming I/O (piped raw RGB — no PNG round-trips)
# =============================================================================

_NVENC_AVAILABLE: Optional[bool] = None


def nvenc_available() -> bool:
    """Return True if FFmpeg reports h264_nvenc encoder."""
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        _NVENC_AVAILABLE = result.returncode == 0 and "h264_nvenc" in result.stdout
    except Exception:
        _NVENC_AVAILABLE = False
    return _NVENC_AVAILABLE


_CUDA_HWACCEL_AVAILABLE: Optional[bool] = None


def cuda_hwaccel_available() -> bool:
    """Return True when FFmpeg reports CUDA hwaccel support."""
    global _CUDA_HWACCEL_AVAILABLE
    if _CUDA_HWACCEL_AVAILABLE is not None:
        return _CUDA_HWACCEL_AVAILABLE
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        _CUDA_HWACCEL_AVAILABLE = (
            result.returncode == 0 and "cuda" in result.stdout.lower()
        )
    except Exception:
        _CUDA_HWACCEL_AVAILABLE = False
    return _CUDA_HWACCEL_AVAILABLE


def _hwaccel_decode_args(enabled: bool, zero_copy: bool = False) -> List[str]:
    if not enabled or not cuda_hwaccel_available():
        return []
    if zero_copy:
        return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    return ["-hwaccel", "cuda"]


def _decode_video_filter(fps: float, zero_copy: bool) -> str:
    vf = f"fps={fps}"
    if zero_copy:
        vf = f"hwdownload,format=rgb24,{vf}"
    return vf


class StreamingFrameReader:
    """Decode video/GIF frames via FFmpeg raw RGB pipe in bounded chunks."""

    def __init__(
        self,
        media_path: str,
        fps: Optional[float] = None,
        hwaccel: bool = False,
        zero_copy: bool = False,
        pinned_pool_size: int = 0,
    ):
        self.media_path = str(media_path)
        meta = probe_video(self.media_path)
        self.width = int(meta.get("width") or 0)
        self.height = int(meta.get("height") or 0)
        self.fps = float(fps or meta.get("fps") or 30.0)
        if self.fps <= 0:
            self.fps = 30.0
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid media dimensions for {self.media_path}")

        self._frame_bytes = self.width * self.height * 3
        self._pinned_pool = None
        if pinned_pool_size > 0:
            from utils.pinned_pool import PinnedFramePool

            self._pinned_pool = PinnedFramePool(
                self.height, self.width, pinned_pool_size
            )

        use_zero_copy = bool(zero_copy and hwaccel)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        cmd.extend(_hwaccel_decode_args(hwaccel, zero_copy=use_zero_copy))
        cmd.extend(
            [
                "-i",
                self.media_path,
                "-vf",
                _decode_video_filter(self.fps, use_zero_copy),
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "-",
            ]
        )
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._stderr_drain = _FfmpegStderrDrain(self._proc, label="decode")
        self.frames_read = 0
        logger.info(
            "Streaming decode started: %s (%dx%d @ %.2f fps, cuda=%s, pinned=%s)",
            Path(self.media_path).name,
            self.width,
            self.height,
            self.fps,
            use_zero_copy,
            self._pinned_pool is not None,
        )

    def read_chunk(self, count: int) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        if self._proc.stdout is None:
            return frames
        for i in range(count):
            data = self._proc.stdout.read(self._frame_bytes)
            if len(data) < self._frame_bytes:
                break
            if self._pinned_pool is not None:
                frame = self._pinned_pool.borrow(i)
                np.copyto(frame, np.frombuffer(data, dtype=np.uint8).reshape(
                    self.height, self.width, 3
                ))
            else:
                frame = np.frombuffer(data, dtype=np.uint8).reshape(
                    self.height, self.width, 3
                ).copy()
            frames.append(frame)
            self.frames_read += 1
        return frames

    def close(self) -> None:
        if self._proc.stdout:
            self._proc.stdout.close()
        self._stderr_drain.join()
        stderr_text = self._stderr_drain.get_text()
        self._proc.wait(timeout=30)
        if self._proc.returncode not in (0, None) and self.frames_read == 0:
            logger.warning(
                "Streaming decode ended rc=%s: %s",
                self._proc.returncode,
                stderr_text[:300],
            )

    def __enter__(self) -> "StreamingFrameReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False


class StreamingVideoWriter:
    """Encode raw RGB frames to MP4 via FFmpeg stdin pipe."""

    def __init__(
        self,
        output_path: Union[str, Path],
        width: int,
        height: int,
        fps: float,
        audio_path: Optional[Union[str, Path]] = None,
        codec: str = "libx264",
        preset: str = "medium",
        crf: int = 18,
        use_nvenc: bool = False,
        pix_fmt: str = "yuv420p",
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.frames_written = 0
        self._finalized = False

        video_codec = "h264_nvenc" if use_nvenc and nvenc_available() else codec
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "pipe:0",
        ]
        if audio_path:
            cmd.extend(["-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0", "-shortest"])
        if video_codec == "h264_nvenc":
            cmd.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(crf)])
        else:
            cmd.extend(["-c:v", video_codec, "-preset", preset, "-crf", str(crf)])
        if audio_path:
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        cmd.extend(["-pix_fmt", pix_fmt, "-movflags", "+faststart", str(self.output_path)])

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._stderr_drain = _FfmpegStderrDrain(self._proc, label="encode")
        logger.info(
            "Streaming encode started: %s (%dx%d @ %.2f fps, codec=%s)",
            self.output_path.name,
            width,
            height,
            fps,
            video_codec,
        )

    def write_frames_pinned(self, contiguous: np.ndarray, frame_count: int) -> None:
        """Write a contiguous pinned HWC batch (N,H,W,3) with one syscall per chunk."""
        if self._proc.stdin is None:
            return
        expected = frame_count * self.height * self.width * 3
        view = memoryview(contiguous)
        if len(view) < expected:
            raise ValueError(
                f"Pinned batch too small: {len(view)} bytes, need {expected}"
            )
        offset = 0
        while offset < expected:
            written = self._proc.stdin.write(view[offset:expected])
            if written == 0:
                raise BrokenPipeError("FFmpeg encode stdin closed unexpectedly")
            offset += written
        self.frames_written += frame_count
        self._proc.stdin.flush()

    def write_frames(self, frames: List[np.ndarray]) -> None:
        if self._proc.stdin is None:
            return
        for frame in frames:
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.shape[:2] != (self.height, self.width):
                raise ValueError(
                    f"Frame shape {arr.shape} does not match writer {self.height}x{self.width}"
                )
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            data = arr.tobytes()
            view = memoryview(data)
            offset = 0
            while offset < len(view):
                written = self._proc.stdin.write(view[offset:])
                if written == 0:
                    raise BrokenPipeError("FFmpeg encode stdin closed unexpectedly")
                offset += written
            self.frames_written += 1
        if self._proc.stdin:
            self._proc.stdin.flush()

    def finalize(self) -> bool:
        if self._finalized:
            return self.output_path.exists()
        self._finalized = True
        if self._proc.stdin:
            self._proc.stdin.close()
        self._stderr_drain.join(timeout=5.0)
        stderr = self._stderr_drain.get_text()
        self._proc.wait(timeout=600)
        ok = self._proc.returncode == 0 and self.output_path.exists()
        if ok:
            logger.info("Streaming encode complete: %d frames → %s", self.frames_written, self.output_path)
        else:
            logger.error("Streaming encode failed (rc=%s): %s", self._proc.returncode, stderr[:500])
        return ok

    def __enter__(self) -> "StreamingVideoWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self.finalize()
        elif self._proc.stdin:
            self._proc.stdin.close()
            self._proc.kill()
        return False


class StreamingGifWriter:
    """Write GIF frames via temp disk files — bounded RAM, variable durations."""

    def __init__(
        self,
        output_path: Union[str, Path],
        temp_dir: Union[str, Path],
        durations: Optional[List[int]] = None,
    ):
        self.output_path = Path(output_path)
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.durations = durations or []
        self.frames_written = 0
        self._finalized = False

    def write_frames(self, frames: List[np.ndarray]) -> None:
        import cv2

        for frame in frames:
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            fpath = self.temp_dir / f"frame_{self.frames_written:06d}.png"
            cv2.imwrite(str(fpath), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            self.frames_written += 1

    def finalize(self) -> bool:
        if self._finalized:
            return self.output_path.exists()
        self._finalized = True
        if self.frames_written == 0:
            logger.error("No frames to write GIF")
            return False

        paths = sorted(self.temp_dir.glob("frame_*.png"))
        from PIL import Image

        pil_frames = []
        for p in paths:
            pil_frames.append(Image.open(p).convert("RGBA"))

        durations = self.durations
        if not durations or len(durations) != len(pil_frames):
            default_ms = int(1000 / 10.0)
            durations = [default_ms] * len(pil_frames)
        elif len(durations) > len(pil_frames):
            durations = durations[: len(pil_frames)]
        elif len(durations) < len(pil_frames):
            durations = (durations * ((len(pil_frames) // len(durations)) + 1))[
                : len(pil_frames)
            ]

        try:
            pil_frames[0].save(
                str(self.output_path),
                save_all=True,
                append_images=pil_frames[1:],
                loop=0,
                duration=durations,
                optimize=True,
            )
            logger.info(
                "Streaming GIF encode complete: %d frames → %s",
                self.frames_written,
                self.output_path,
            )
            return True
        except Exception as exc:
            logger.error("Streaming GIF encode failed: %s", exc)
            return False

    def cleanup(self) -> None:
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def __enter__(self) -> "StreamingGifWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            if exc_type is None:
                self.finalize()
        finally:
            self.cleanup()
        return False


# =============================================================================
# Frame extraction (video -> frames) — legacy / tests only
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


# =============================================================================
# Video writing (frames -> video) — legacy fallback
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
                "-movflags", "+faststart",
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
