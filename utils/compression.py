"""
Compression utilities for optimizing output file sizes while maintaining quality.
"""
import logging
from pathlib import Path
from PIL import Image
import subprocess
import shutil
from typing import Tuple, Optional
import os

logger = logging.getLogger("FaceOff")

# Get the project root directory (where this script is located)
PROJECT_ROOT = Path(__file__).parent.parent
EXTERNAL_GIFSICLE = PROJECT_ROOT / "external" / "gifsicle" / "gifsicle.exe"


def _find_gifsicle() -> Optional[str]:
    """Find gifsicle executable in local external folder or system PATH."""
    # Check local external folder first
    if EXTERNAL_GIFSICLE.exists():
        return str(EXTERNAL_GIFSICLE)
    # Fall back to system PATH
    return shutil.which('gifsicle')


def compress_image(input_path: str, output_path: Optional[str] = None, quality: int = 95) -> Tuple[bool, str, dict]:
    """
    Compress image with minimal quality loss.
    
    Args:
        input_path: Path to input image
        output_path: Path for output (if None, overwrites input)
        quality: JPEG quality (85-100), PNG uses optimize=True
        
    Returns:
        Tuple of (success, message, stats_dict)
    """
    try:
        input_file = Path(input_path)
        if not input_file.exists():
            return False, f"Input file not found: {input_path}", {}
        
        output_file = Path(output_path) if output_path else input_file
        original_size = input_file.stat().st_size
        
        # Load image
        img = Image.open(input_file)
        
        # Convert RGBA to RGB if saving as JPEG
        if output_file.suffix.lower() in ['.jpg', '.jpeg'] and img.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
            img = background
        
        # Save with optimization
        if output_file.suffix.lower() == '.png':
            # PNG: Lossless compression with optimize
            img.save(output_file, 'PNG', optimize=True, compress_level=9)
        elif output_file.suffix.lower() in ['.jpg', '.jpeg']:
            # JPEG: High quality with optimization
            img.save(output_file, 'JPEG', quality=quality, optimize=True)
        elif output_file.suffix.lower() == '.webp':
            # WebP: Near-lossless compression
            img.save(output_file, 'WEBP', quality=quality, method=6)
        else:
            img.save(output_file, optimize=True)
        
        # Calculate compression stats
        new_size = output_file.stat().st_size
        savings = original_size - new_size
        savings_percent = (savings / original_size) * 100 if original_size > 0 else 0
        
        stats = {
            'original_size': original_size,
            'compressed_size': new_size,
            'savings_bytes': savings,
            'savings_percent': savings_percent
        }
        
        logger.info(f"Compressed image: {input_file.name} | {original_size:,} -> {new_size:,} bytes ({savings_percent:.1f}% reduction)")
        
        return True, f"✅ Compressed: {savings_percent:.1f}% smaller ({_format_bytes(original_size)} -> {_format_bytes(new_size)})", stats
        
    except Exception as e:
        logger.error(f"Error compressing image {input_path}: {e}")
        return False, f"❌ Compression failed: {str(e)}", {}


def compress_gif(input_path: str, output_path: Optional[str] = None, lossy: int = 30) -> Tuple[bool, str, dict]:
    """
    Compress GIF using gifsicle, ImageMagick, or PIL optimization.
    
    Args:
        input_path: Path to input GIF
        output_path: Path for output (if None, overwrites input)
        lossy: Lossy compression level 0-200 (0=lossless, 30=good balance, 80=very lossy)
        
    Returns:
        Tuple of (success, message, stats_dict)
    """
    try:
        input_file = Path(input_path)
        if not input_file.exists():
            return False, f"Input file not found: {input_path}", {}
        
        output_file = Path(output_path) if output_path else input_file
        original_size = input_file.stat().st_size
        
        # Method 1: Try gifsicle first (best quality/size ratio)
        gifsicle_path = _find_gifsicle()
        if gifsicle_path:
            try:
                temp_output = output_file.with_suffix('.gif.tmp')
                
                cmd = [
                    gifsicle_path,
                    '--optimize=3',  # Maximum optimization
                    '--colors=256',  # Full color palette
                ]
                
                if lossy > 0:
                    cmd.append(f'--lossy={lossy}')
                
                cmd.extend(['-o', str(temp_output), str(input_file)])
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0 and temp_output.exists():
                    # Replace original with compressed
                    temp_output.replace(output_file)
                    new_size = output_file.stat().st_size
                    
                    savings = original_size - new_size
                    savings_percent = (savings / original_size) * 100 if original_size > 0 else 0
                    
                    stats = {
                        'original_size': original_size,
                        'compressed_size': new_size,
                        'savings_bytes': savings,
                        'savings_percent': savings_percent,
                        'method': 'gifsicle'
                    }
                    
                    logger.info(f"Compressed GIF (gifsicle): {input_file.name} | {original_size:,} -> {new_size:,} bytes ({savings_percent:.1f}% reduction)")
                    
                    return True, f"✅ Compressed: {savings_percent:.1f}% smaller ({_format_bytes(original_size)} -> {_format_bytes(new_size)})", stats
                else:
                    logger.warning(f"gifsicle failed: {result.stderr}")
                    # Fall through to PIL method
                    
            except subprocess.TimeoutExpired:
                logger.warning("gifsicle timeout, falling back to ImageMagick/PIL")
            except Exception as e:
                logger.warning(f"gifsicle error: {e}, falling back to ImageMagick/PIL")
        
        # Method 2: Try ImageMagick (better than PIL, widely available)
        if shutil.which('magick'):
            try:
                temp_output = output_file.with_suffix('.gif.tmp')
                
                # ImageMagick with optimization and quality settings
                cmd = [
                    'magick',
                    str(input_file),
                    '-coalesce',  # Expand frames to full size
                    '-layers', 'optimize',  # Optimize frame overlaps
                    '-fuzz', '5%',  # Allow slight color variations for better compression
                    str(temp_output)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0 and temp_output.exists():
                    # Replace original with compressed
                    temp_output.replace(output_file)
                    new_size = output_file.stat().st_size
                    
                    savings = original_size - new_size
                    savings_percent = (savings / original_size) * 100 if original_size > 0 else 0
                    
                    stats = {
                        'original_size': original_size,
                        'compressed_size': new_size,
                        'savings_bytes': savings,
                        'savings_percent': savings_percent,
                        'method': 'ImageMagick'
                    }
                    
                    logger.info(f"Compressed GIF (ImageMagick): {input_file.name} | {original_size:,} -> {new_size:,} bytes ({savings_percent:.1f}% reduction)")
                    
                    return True, f"✅ Compressed: {savings_percent:.1f}% smaller ({_format_bytes(original_size)} -> {_format_bytes(new_size)})", stats
                else:
                    logger.warning(f"ImageMagick failed: {result.stderr}")
                    # Fall through to PIL method
                    
            except subprocess.TimeoutExpired:
                logger.warning("ImageMagick timeout, falling back to PIL")
            except Exception as e:
                logger.warning(f"ImageMagick error: {e}, falling back to PIL")
        
        # Method 3: PIL optimization (basic but always available)
        img = Image.open(input_file)
        
        # Save with optimization
        frames = []
        try:
            while True:
                frames.append(img.copy())
                img.seek(img.tell() + 1)
        except EOFError:
            pass
        
        if frames:
            frames[0].save(
                output_file,
                save_all=True,
                append_images=frames[1:],
                optimize=True,
                duration=img.info.get('duration', 100),
                loop=img.info.get('loop', 0)
            )
            
            new_size = output_file.stat().st_size
            savings = original_size - new_size
            savings_percent = (savings / original_size) * 100 if original_size > 0 else 0
            
            stats = {
                'original_size': original_size,
                'compressed_size': new_size,
                'savings_bytes': savings,
                'savings_percent': savings_percent,
                'method': 'PIL'
            }
            
            logger.info(f"Compressed GIF (PIL): {input_file.name} | {original_size:,} -> {new_size:,} bytes ({savings_percent:.1f}% reduction)")
            
            return True, f"✅ Compressed: {savings_percent:.1f}% smaller ({_format_bytes(original_size)} -> {_format_bytes(new_size)})", stats
        
        return False, "❌ No frames found in GIF", {}
        
    except Exception as e:
        logger.error(f"Error compressing GIF {input_path}: {e}")
        return False, f"❌ Compression failed: {str(e)}", {}


def compress_video(input_path: str, output_path: Optional[str] = None, crf: int = 23) -> Tuple[bool, str, dict]:
    """
    Compress video using ffmpeg with H.265 (HEVC) or H.264.
    
    Args:
        input_path: Path to input video
        output_path: Path for output (if None, creates _compressed.mp4)
        crf: Constant Rate Factor (18=visually lossless, 23=good balance, 28=lower quality)
             Lower = better quality, larger file
        
    Returns:
        Tuple of (success, message, stats_dict)
    """
    try:
        input_file = Path(input_path)
        if not input_file.exists():
            return False, f"Input file not found: {input_path}", {}
        
        # Default output path
        if output_path is None:
            output_path = input_file.with_stem(input_file.stem + '_compressed')
        
        output_file = Path(output_path)
        original_size = input_file.stat().st_size
        
        # Check if ffmpeg is available
        if not shutil.which('ffmpeg'):
            return False, "❌ ffmpeg not found. Install ffmpeg to compress videos.", {}
        
        # Try H.265 first (better compression)
        cmd = [
            'ffmpeg',
            '-i', str(input_file),
            '-c:v', 'libx265',  # H.265 codec
            '-crf', str(crf),
            '-preset', 'medium',  # Balance between speed and compression
            '-c:a', 'aac',  # Audio codec
            '-b:a', '128k',  # Audio bitrate
            '-movflags', '+faststart',  # Enable streaming
            '-y',  # Overwrite output
            str(output_file)
        ]
        
        logger.info(f"Compressing video with H.265 (CRF={crf}): {input_file.name}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                # H.265 failed, try H.264
                logger.warning("H.265 compression failed, trying H.264")
                cmd[cmd.index('libx265')] = 'libx264'
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
            
            if result.returncode == 0 and output_file.exists():
                new_size = output_file.stat().st_size
                savings = original_size - new_size
                savings_percent = (savings / original_size) * 100 if original_size > 0 else 0
                
                stats = {
                    'original_size': original_size,
                    'compressed_size': new_size,
                    'savings_bytes': savings,
                    'savings_percent': savings_percent,
                    'crf': crf
                }
                
                logger.info(f"Compressed video: {input_file.name} | {original_size:,} -> {new_size:,} bytes ({savings_percent:.1f}% reduction)")
                
                return True, f"✅ Compressed: {savings_percent:.1f}% smaller ({_format_bytes(original_size)} -> {_format_bytes(new_size)})", stats
            else:
                logger.error(f"ffmpeg error: {result.stderr}")
                return False, f"❌ Compression failed: {result.stderr[:200]}", {}
                
        except subprocess.TimeoutExpired:
            return False, "❌ Compression timeout (>5 minutes)", {}
            
    except Exception as e:
        logger.error(f"Error compressing video {input_path}: {e}")
        return False, f"❌ Compression failed: {str(e)}", {}


def _format_bytes(bytes_val: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} TB"


def compress_media_file(file_path: str, media_type: str, **kwargs) -> Tuple[bool, str, dict]:
    """
    Compress media file based on type.
    
    Args:
        file_path: Path to file
        media_type: "image", "gif", or "video"
        **kwargs: Additional compression options
        
    Returns:
        Tuple of (success, message, stats_dict)
    """
    if media_type == "image":
        return compress_image(file_path, **kwargs)
    elif media_type == "gif":
        return compress_gif(file_path, **kwargs)
    elif media_type == "video":
        return compress_video(file_path, **kwargs)
    else:
        return False, f"Unknown media type: {media_type}", {}
