import os
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
from pathlib import Path
import shutil
from natsort import natsorted
import subprocess
import gc
from PIL import Image
from config import get_temp_frames_dir

def get_ffmpeg_path():
    """Get the path to the bundled ffmpeg binary or system ffmpeg."""
    # Try bundled ffmpeg first (in packaged app)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check multiple possible locations for the bundled binary
    possible_paths = [
        # Packaged app structure: backend/processing -> ../../binaries/ffmpeg
        os.path.join(current_dir, '..', '..', 'binaries', 'ffmpeg'),
        # Alternative: go up to Resources and then to binaries  
        os.path.join(current_dir, '..', '..', '..', 'binaries', 'ffmpeg'),
        # Try from the main app directory
        os.path.join(current_dir, '..', '..', '..', '..', 'Resources', 'binaries', 'ffmpeg'),
        # Additional paths for different packaged app structures
        os.path.join(current_dir, 'binaries', 'ffmpeg'),
        os.path.join(current_dir, '..', 'app.asar.unpacked', 'binaries', 'ffmpeg'),
    ]
    
    print(f"🔍 Looking for bundled ffmpeg from base directory: {current_dir}")
    
    for bundled_ffmpeg in possible_paths:
        print(f"🔍 Checking ffmpeg at: {bundled_ffmpeg}")
        if os.path.exists(bundled_ffmpeg):
            print(f"✅ Found bundled ffmpeg: {bundled_ffmpeg}")
            return bundled_ffmpeg
        else:
            print(f"❌ Not found at: {bundled_ffmpeg}")
    
    # Fallback to system ffmpeg
    print("⚠️ Bundled ffmpeg not found, using system ffmpeg")
    return 'ffmpeg'

def get_ffprobe_path():
    """Get the path to the bundled ffprobe binary or system ffprobe."""
    # Try bundled ffprobe first (in packaged app)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check multiple possible locations for the bundled binary
    possible_paths = [
        # Packaged app structure: backend/processing -> ../../binaries/ffprobe
        os.path.join(current_dir, '..', '..', 'binaries', 'ffprobe'),
        # Alternative: go up to Resources and then to binaries  
        os.path.join(current_dir, '..', '..', '..', 'binaries', 'ffprobe'),
        # Try from the main app directory
        os.path.join(current_dir, '..', '..', '..', '..', 'Resources', 'binaries', 'ffprobe'),
        # Additional paths for different packaged app structures
        os.path.join(current_dir, 'binaries', 'ffprobe'),
        os.path.join(current_dir, '..', 'app.asar.unpacked', 'binaries', 'ffprobe'),
    ]
    
    print(f"🔍 Looking for bundled ffprobe from base directory: {current_dir}")
    
    for bundled_ffprobe in possible_paths:
        print(f"🔍 Checking ffprobe at: {bundled_ffprobe}")
        if os.path.exists(bundled_ffprobe):
            print(f"✅ Found bundled ffprobe: {bundled_ffprobe}")
            return bundled_ffprobe
        else:
            print(f"❌ Not found at: {bundled_ffprobe}")
    
    # Fallback to system ffprobe
    print("⚠️ Bundled ffprobe not found, using system ffprobe")
    return 'ffprobe'

def get_length(filename):
    result = subprocess.run([get_ffprobe_path(), "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL)  # Suppress stderr
    return round(float(result.stdout), 0)

def calculate_keyframe_interval(video_length):
    """
    Calculate the interval between keyframes based on video length.
    Shorter videos get more frequent keyframes.
    """
    if video_length <= 1:  # Ultra-short videos (1 second or less)
        return 0.5  # Every 0.5 seconds
    elif video_length <= 5:  # Very short videos (5 seconds or less)
        return 1  # Every 1 second
    elif video_length <= 30:  # Short videos (30 seconds or less)
        return 2  # Every 2 seconds
    elif video_length <= 120:  # Medium videos (2 minutes or less)
        return 3  # Every 3 seconds
    elif video_length <= 300:  # Long videos (5 minutes or less)
        return 4  # Every 4 seconds
    elif video_length <= 600:  # Longer videos (10 minutes or less)
        return 5  # Every 5 seconds
    else:  # Very long videos
        return 6  # Every 6 seconds

def load_gray_image(path):
    """Load image and convert to grayscale for comparison"""
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def compress_image(image_path, max_pixels=1000, quality=85):
    """
    Compress an image by resizing to max_pixels on longest side.
    
    Args:
        image_path (str): Path to the image file
        max_pixels (int): Maximum pixels for longest side (default: 1000)
        quality (int): JPEG quality (default: 85)
        
    Returns:
        tuple: (original_size_kb, compressed_size_kb, compression_ratio)
    """
    try:
        # Get original file size
        original_size = os.path.getsize(image_path) / 1024  # KB
        
        # Open and process image
        with Image.open(image_path) as img:
            # Get current dimensions
            width, height = img.size
            
            # Calculate new dimensions if resizing needed
            if max(width, height) > max_pixels:
                # Calculate scale factor to fit within max_pixels
                scale_factor = max_pixels / max(width, height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                # Resize image
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save with compression
            img.save(image_path, "JPEG", quality=quality, optimize=True)
        
        # Get compressed file size
        compressed_size = os.path.getsize(image_path) / 1024  # KB
        compression_ratio = ((original_size - compressed_size) / original_size) * 100
        
        return original_size, compressed_size, compression_ratio
        
    except Exception as e:
        print(f"Error compressing {image_path}: {e}")
        return 0, 0, 0

def split_frames(vid_path, compress_frames=False, max_pixels=1000):
    """
    Extract frames from video with optional compression.
    Uses temporary directory for frame storage to avoid conflicts.
    
    Args:
        vid_path (str): Path to video file
        compress_frames (bool): Whether to compress extracted frames
        max_pixels (int): Maximum pixels for longest side if compressing
        
    Returns:
        tuple: (frames_dir, metadata) - Returns the temp directory path and frame metadata
    """
    print(f"🎬 Extracting frames{'(compressed)' if compress_frames else ''} to temp directory...")
    frame_interval = calculate_keyframe_interval(get_length(vid_path))  # seconds between frames

    # Use temporary directory instead of creating frames dir next to video
    frame_dir = get_temp_frames_dir(vid_path)
    print(f"📁 Using temp directory: {frame_dir}")

    # Extract frames using subprocess instead of ffmpeg library
    try:
        ffmpeg_path = get_ffmpeg_path()
        
        # Build ffmpeg command
        cmd = [
            ffmpeg_path,
            '-i', vid_path,
            '-vf', f'fps=1/{frame_interval}',
            '-y',  # Overwrite output files
            '-loglevel', 'error',  # Only show errors
            f'{frame_dir}/frame_%06d.jpg'
        ]
        
        print(f"🔧 Running ffmpeg: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"FFmpeg failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error during frame extraction: {e}")
        # Clean up empty temp directory
        try:
            shutil.rmtree(frame_dir)
        except:
            pass
        return None, {}

    frame_paths = natsorted([str(p) for p in Path(frame_dir).glob("*.jpg")])
    
    if not frame_paths:
        print(f"❌ No frames extracted from {vid_path}")
        try:
            shutil.rmtree(frame_dir)
        except:
            pass
        return None, {}
    
    # Compress frames if requested
    if compress_frames and frame_paths:
        print(f"🗜️  Compressing {len(frame_paths)} frames to max {max_pixels}px...")
        total_original = 0
        total_compressed = 0
        
        for frame_path in frame_paths:
            orig_size, comp_size, ratio = compress_image(frame_path, max_pixels)
            total_original += orig_size
            total_compressed += comp_size
        
        overall_ratio = ((total_original - total_compressed) / total_original) * 100 if total_original > 0 else 0
        print(f"✅ Compression complete: {total_original:.1f}KB → {total_compressed:.1f}KB ({overall_ratio:.1f}% reduction)")

    # Create metadata with timestamps
    metadata = {}
    for i, path in enumerate(frame_paths):
        timestamp = round(i * frame_interval, 2)
        metadata[path] = {"timestamp": timestamp}

    # Mark duplicates instead of removing them during iteration
    duplicates = set()
    i = 0
    while i < len(frame_paths)-1:
        current_path = frame_paths[i]
        next_path = frame_paths[i+1]
        
        try:
            reference_image = load_gray_image(current_path)
            comparison_image = load_gray_image(next_path)

            if ssim(reference_image, comparison_image) > 0.75:
                duplicates.add(next_path)
            
            # Clean up images
            del reference_image
            del comparison_image
            gc.collect()
        except Exception as e:
            print(f"Error comparing frames {current_path} and {next_path}: {e}")
            i += 1
            continue
            
        i += 1

    # Remove duplicates after iteration
    for dup_path in duplicates:
        if dup_path in frame_paths:
            frame_paths.remove(dup_path)
        if dup_path in metadata:
            del metadata[dup_path]
        try:
            os.remove(dup_path)
        except Exception as e:
            print(f"Error removing duplicate frame {dup_path}: {e}")

    print(f"✅ Extracted {len(frame_paths)} unique frames to {frame_dir}")
    return frame_dir, metadata

# vid_path = 'day7.mp4'
# split_frames(vid_path)

