import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from framesegmentation import split_frames
from config import cleanup_temp_frames
from framestagging import (
    analyze_keyframes_with_gpt4o_vision, 
    process_frames_with_moondream_api,
    calculate_keyframes_to_send,
    select_representative_keyframes
)
import audioanalyzer
from location_utils import process_location_from_metadata
from config import get_video_metadata_path, get_temp_frames_dir
from api_client import get_api_client
import api_client
from dotenv import load_dotenv
# NOTE: OpenAI import removed - packaged app uses Railway API for all AI calls
import json
import shutil
import subprocess
from datetime import datetime
import psutil
import base64
import math
from pathlib import Path
import easyocr
from PIL import Image
import numpy as np
import warnings
import gc
import threading
import concurrent.futures
from sentence_transformers import util
import time
import torch

# Suppress various warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*FP16 is not supported on CPU.*")

# Suppress DETR model warnings specifically
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint.*were not used")
warnings.filterwarnings("ignore", message=".*This IS expected if you are initializing.*")
warnings.filterwarnings("ignore", message=".*This IS NOT expected if you are initializing.*")
warnings.filterwarnings("ignore", message="Using a slow image processor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

# Set environment variables to suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress transformers logging for cleaner output
try:
    from transformers import logging as transformers_logging
    transformers_logging.set_verbosity_error()
except ImportError:
    pass

# Performance logging configuration
ENABLE_PERFORMANCE_LOGGING = os.getenv("VERBOSE_LOGGING", "true").lower() == "true"
ENABLE_DEBUG_LOGGING = os.getenv("DEBUG_LOGGING", "false").lower() == "true"

def perf_print(*args, **kwargs):
    """High-performance logging that can be toggled off"""
    if ENABLE_PERFORMANCE_LOGGING:
        print(*args, **kwargs)

def debug_print(*args, **kwargs):
    """Debug logging for detailed analysis"""
    if ENABLE_DEBUG_LOGGING:
        print(*args, **kwargs)

load_dotenv()

# Initialize API client for server-based processing
api_client.use_api_server()  # Uses environment variable CONTENTCACHE_API_URL or defaults to localhost

# Initialize OCR reader globally (expensive to initialize)
try:
    # Suppress EasyOCR warnings during initialization
    import logging
    logging.getLogger('easyocr').setLevel(logging.ERROR)
    easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)  # Set verbose=False
    print("✅ EasyOCR initialized for video text extraction")
except Exception as e:
    easyocr_reader = None
    print(f"⚠️ Could not initialize EasyOCR: {e}")

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def extract_video_metadata(vid_path):
    """Extract metadata from video file using ffprobe."""
    try:
        # Run ffprobe to get metadata
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', vid_path
        ], capture_output=True, text=True, check=True)
        
        metadata = json.loads(result.stdout)
        return metadata
    except subprocess.CalledProcessError as e:
        print(f"Error running ffprobe: {e}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing ffprobe output: {e}")
        return {}
    except FileNotFoundError:
        print("ffprobe not found. Please install ffmpeg.")
        return {}

def convert_coordinates_to_location(location_text):
    """
    DEPRECATED: Use location_utils.process_location_from_metadata() instead.
    This function is kept for compatibility but now delegates to Google Maps processing.
    """
    debug_print(f"⚠️ Using deprecated convert_coordinates_to_location() - consider updating to use location_utils directly")
    return process_location_from_metadata(location_text)

def extract_and_store_location_coordinates(video_metadata):
    """
    Extract location coordinates from video metadata and store them as raw coordinates.
    No longer converts to readable text - coordinates will be used for proximity search.
    
    Args:
        video_metadata (dict): Video metadata from ffprobe
        
    Returns:
        dict: Location data with coordinates or None if no location found
    """
    if not video_metadata:
        return None
    
    # Look for location data in various metadata fields
    location_sources = []
    
    # Check format tags
    format_tags = video_metadata.get('format', {}).get('tags', {})
    if format_tags:
        # Common location fields
        location_fields = [
            'location', 'Location', 'LOCATION',
            'gps_coordinates', 'GPS_COORDINATES', 'gps',
            'com.apple.quicktime.location.ISO6709',
            'com.apple.quicktime.location.name',
            'location-eng', 'location-und'
        ]
        
        for field in location_fields:
            if field in format_tags and format_tags[field]:
                location_sources.append(format_tags[field])
    
    # Check stream tags
    for stream in video_metadata.get('streams', []):
        stream_tags = stream.get('tags', {})
        if stream_tags:
            for field in ['location', 'Location', 'LOCATION']:
                if field in stream_tags and stream_tags[field]:
                    location_sources.append(stream_tags[field])
    
    # Process each potential location source to extract coordinates
    for location_data in location_sources:
        if not location_data or location_data.lower() in ['none', 'null', '']:
            continue
            
        location_str = str(location_data).strip()
        
        # Skip obviously invalid location strings
        if len(location_str) > 3 and not location_str.lower() in ['unknown', 'n/a']:
            try:
                # Try to parse coordinates from the location string
                # Look for patterns like "+37.7749-122.4194/" or "37.7749,-122.4194"
                import re
                coord_pattern = r'([+-]?\d+\.?\d*)[,\s]*([+-]?\d+\.?\d*)'
                match = re.search(coord_pattern, location_str)
                
                if match:
                    lat, lon = float(match.group(1)), float(match.group(2))
                    
                    # Validate coordinate ranges
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        # Return simple coordinate string for consistent storage
                        coordinate_string = f"{lat}, {lon}"
                        debug_print(f"📍 Video coordinates extracted: {coordinate_string}")
                        return coordinate_string
                    else:
                        debug_print(f"⚠️ Invalid coordinate ranges: lat={lat}, lon={lon}")
                        continue
                else:
                    # If no coordinates found, store location text directly
                    debug_print(f"📍 Video location (text): {location_str}")
                    return location_str
                    
            except Exception as e:
                debug_print(f"⚠️ Location processing error: {e}")
                continue
    
    return None

def trim_metadata_for_api(video_metadata):
    """
    Trim video metadata to only essential fields for API calls.
    Reduces token usage and focuses on relevant information.
    """
    if not video_metadata:
        return {}
    
    trimmed = {}
    
    # Extract filename from format section
    format_info = video_metadata.get('format', {})
    if 'filename' in format_info:
        trimmed['filename'] = os.path.basename(format_info['filename'])
    
    # Extract creation time from format tags
    format_tags = format_info.get('tags', {})
    if 'creation_time' in format_tags:
        trimmed['creation_time'] = format_tags['creation_time']
    
    # Extract device/encoder info from streams
    streams = video_metadata.get('streams', [])
    for stream in streams:
        stream_tags = stream.get('tags', {})
        if 'handler_name' in stream_tags:
            trimmed['device_info'] = stream_tags['handler_name']
            break
        if 'encoder' in stream_tags:
            trimmed['encoder'] = stream_tags['encoder']
            break
    
    # Extract basic video properties
    video_stream = next((s for s in streams if s.get('codec_type') == 'video'), {})
    if video_stream:
        trimmed['duration'] = video_stream.get('duration', format_info.get('duration'))
        trimmed['resolution'] = f"{video_stream.get('width', 'unknown')}x{video_stream.get('height', 'unknown')}"
    
    return trimmed

def call_gpt4o(frame_captions, audio_summary, video_metadata, vision_analysis=None, text_data=None, processed_location=None):
    # Use API server for video summary with built-in prompts
    try:
        # Trim metadata to essential fields only
        trimmed_metadata = trim_metadata_for_api(video_metadata)
        
        print(f"🌐 Calling Railway API for video summary...")
        print(f"📝 Frame captions: {len(frame_captions)} items")
        print(f"🔊 Audio summary available: {'Yes' if audio_summary else 'No'}")
        print(f"📍 Location data: {'Yes' if processed_location else 'No'}")
        
        response_data = api_client.get_api_client().openai_video_summary(
            frame_captions=frame_captions,
            audio_summary=audio_summary,
            video_metadata=trimmed_metadata,
            vision_analysis=vision_analysis,
            text_data=text_data,
            processed_location=processed_location
        )
        
        print(f"✅ Railway API call successful")
        
        # Extract the structured JSON output from API response
        if response_data and 'result' in response_data:
            return response_data['result']
        else:
            print(f"❌ Invalid API response structure: {response_data}")
            raise Exception("Invalid API response for video summary")
            
    except Exception as e:
        print(f"❌ Railway API call failed with error: {str(e)}")
        print(f"📊 Error type: {type(e).__name__}")
        
        # Check if it's a connection error specifically
        if "ConnectionError" in str(type(e)) or "Connection" in str(e):
            print("🔌 This appears to be a connection issue with Railway API")
        elif "timeout" in str(e).lower():
            print("⏱️ This appears to be a timeout issue with Railway API")
        elif "404" in str(e) or "Not Found" in str(e):
            print("🔍 API endpoint not found - check Railway API is deployed correctly")
        else:
            print(f"🔧 Other error details: {str(e)}")
        
        # Re-raise the exception instead of returning fallback to make failures obvious
        raise Exception(f"Railway API call failed: {str(e)}")
        
        # OLD CODE - commented out to avoid silent failures
        # return json.dumps({
        #     "video_summary": "Video analysis unavailable due to API server error.",
        #     "tags": {
        #         "mood": [],
        #         "locations": [],
        #         "context": [],
        #         "objects": [],
        #         "video_style": [],
        #         "actions": [],
        #         "people": []
        #     },
        #     "metadata": {
        #         "date_recorded": "None",
        #         "location": processed_location if processed_location else "None",
        #         "included_description": "None"
        #     }
        # })

def remove_frames_dir(frames_dir):
    """Remove frames directory after processing is complete"""
    try:
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
            print(f"🧹 Cleaned up frames directory: {frames_dir}")
        else:
            print(f"⚠️ Frames directory not found (may have been cleaned already): {frames_dir}")
    except Exception as e:
        print(f"⚠️ Failed to remove frames directory {frames_dir}: {e}")
        # For temp directories, this isn't critical - OS will clean up eventually

def tag_video(vid_path):
    """
    Main video tagging function - uses the latest smart conflict resolution implementation.
    """
    return tag_video_smart_conflict_resolution(vid_path, use_moondream_api=True)

def extract_prominent_text_from_frames(selected_frames):
    """
    Extract prominent text from selected video frames using OCR.
    This runs BEFORE compression to get better text detection accuracy.
    
    Args:
        selected_frames (list): Paths to selected keyframe images
        
    Returns:
        dict: Extracted prominent text information from all frames
    """
    print(f"📖 Extracting text from {len(selected_frames)} keyframes...")
    
    all_prominent_text = []
    frame_text_data = {}
    
    for i, frame_path in enumerate(selected_frames):
        try:
            # Load image at full resolution for better OCR accuracy
            pil_image = Image.open(frame_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            image_np = np.array(pil_image)
            frame_name = os.path.basename(frame_path)
            
            frame_text = {
                'prominent_text': [],
                'method': 'none'
            }
            
            # Try EasyOCR first (generally more accurate)
            if easyocr_reader:
                try:
                    results = easyocr_reader.readtext(image_np)
                    
                    frame_prominent = []
                    
                    for (bbox, text, confidence) in results:
                        if confidence > 0.6:  # Only high confidence text
                            text_clean = text.strip()
                            if len(text_clean) > 2:  # Only reasonably long text
                                    frame_prominent.append(text_clean)
                    
                    frame_text.update({
                        'prominent_text': frame_prominent,
                        'method': 'easyocr'
                    })
                    
                    # Add to overall collection
                    all_prominent_text.extend(frame_prominent)
                    
                    if frame_prominent:
                        debug_print(f"    📖 {frame_name}: Found {len(frame_prominent)} prominent text elements")
                    
                except Exception as e:
                    debug_print(f"    ⚠️ EasyOCR failed for {frame_name}: {e}")
            
            # If EasyOCR failed to extract any text, log it for debugging
            if not frame_text['prominent_text']:
                debug_print(f"    ℹ️ No prominent text found in {frame_name}")
            
            frame_text_data[frame_name] = frame_text
            
        except Exception as e:
            print(f"    ❌ Text extraction failed for {os.path.basename(frame_path)}: {e}")
    
    # Remove duplicates while preserving order
    unique_prominent = []
    
    for text in all_prominent_text:
        if text not in unique_prominent:
            unique_prominent.append(text)
    
    result = {
        'prominent_text': unique_prominent,
        'frame_details': frame_text_data,
        'total_frames_processed': len(selected_frames),
        'frames_with_text': len([f for f in frame_text_data.values() if f['prominent_text']])
    }
    
    print(f"✅ Text extraction complete: {len(unique_prominent)} prominent text elements")
    return result

def save_video_metadata(vid_path, metadata, output_file=None, processed_location=None):
    # Use config path by default, allow override for backward compatibility
    if output_file is None:
        output_file = get_video_metadata_path()
    
    # Load existing metadata (if any)
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            all_data = json.load(f)
    else:
        all_data = {}

    # Use the absolute path to ensure uniqueness
    abs_path = os.path.abspath(vid_path)

    # Prepare metadata, using processed location if available
    video_metadata = metadata.get("metadata", {})
    if processed_location:
        video_metadata["location"] = processed_location
        debug_print(f"📍 Using processed location in saved metadata: {processed_location}")
    elif not video_metadata.get("location") or video_metadata.get("location") == "None":
        video_metadata["location"] = "None"

    # Save/overwrite the entry
    all_data[abs_path] = {
        "video_summary": metadata.get("video_summary", ""),
        "tags": metadata.get("tags", {}),
        "metadata": video_metadata
    }

    # Write back to the JSON file
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"✅ Metadata saved for: {abs_path}")
    
    # Generate embeddings for search functionality
    try:
        from embedding_generator import generate_video_embeddings
        generate_video_embeddings(abs_path, all_data[abs_path])
        debug_print(f"✅ Generated embeddings for: {abs_path}")
    except Exception as e:
        print(f"⚠️ Failed to generate video embeddings: {e}")

# ============================================================================
# ENHANCED VISUAL ANALYSIS WITH GPT-4O MINI VISION API
# ============================================================================

def encode_image_to_base64(image_path, resize=True, max_dimension=512, quality=50):
    """
    Encode image to base64 for OpenAI API with compression to reduce token costs.
    
    Args:
        image_path (str): Path to the image file
        resize (bool): Whether to resize the image (default True for cost savings)
        max_dimension (int): Maximum size for the larger dimension (width or height)
        quality (int): JPEG quality (1-100, lower = more compression)
        show_preview (bool): Whether to display the image being sent to API
        
    Returns:
        str: Base64 encoded compressed image
    """
    try:
        from PIL import Image
        import io
        
        # Open and process the image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Store original dimensions for logging
            original_dimensions = img.size
            
            # Resize based on the larger dimension
            if resize:
                width, height = img.size
                if width > height:
                    # Width is larger, scale based on width
                    new_width = max_dimension
                    new_height = int((height * max_dimension) / width)
                else:
                    # Height is larger (or equal), scale based on height
                    new_height = max_dimension
                    new_width = int((width * max_dimension) / height)
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                resize_info = f" resized from {original_dimensions} to {img.size}"
            else:
                resize_info = f" kept at {img.size}"
            # Compress to JPEG in memory with optimization
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            buffer.seek(0)
            
            # Encode to base64
            compressed_data = base64.b64encode(buffer.read()).decode('utf-8')
            
            # Log compression info
            original_size = os.path.getsize(image_path)
            compressed_size = len(compressed_data) * 3 // 4  # Approximate size after base64 decoding
            compression_ratio = (1 - compressed_size / original_size) * 100
            
            print(f"    🗜️  Compressed {os.path.basename(image_path)}: {original_size//1024}KB → {compressed_size//1024}KB ({compression_ratio:.1f}% reduction){resize_info}")
            
            return compressed_data
            
    except Exception as e:
        print(f"❌ Error encoding/compressing image {image_path}: {e}")
        return None

# Functions moved to framestagging.py for better organization





def cleanup_cache_and_memory():
    """
    Light cleanup: Clear caches, temporary files, and free unused memory,
    but KEEP models loaded for continued processing.
    """
    print("🧹 Performing memory cache cleanup (keeping models loaded)...")
    memory_before = get_memory_usage()
    
    # Multiple rounds of garbage collection to clear variables
    for _ in range(3):
        gc.collect()
    
    # Clear ML framework caches but keep models loaded
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass
    
    # Clear TensorFlow memory (but keep models)
    try:
        import tensorflow as tf
        # Only clear session, don't reset memory growth
        tf.keras.backend.clear_session()
    except (ImportError, AttributeError):
        pass
    
    memory_after = get_memory_usage()
    memory_freed = memory_before - memory_after
    
    if memory_freed > 0:
        print(f"🧹 Cache cleanup freed: {memory_freed:.1f} MB (models still loaded)")
    else:
        print(f"🧹 Cache cleanup complete (Memory: {memory_after:.1f} MB, models still loaded)")
    
    return memory_freed

def cleanup_all_models():
    """
    Cleanup all loaded models to free memory.
    Should be called after processing to prevent memory leaks.
    """
    print("🧹 Cleaning up all models...")
    
    memory_before = get_memory_usage()
    
    models_cleaned = []
    
    # Cleanup framestagging models
    try:
        if hasattr(framestagging, 'sentence_model'):
            del framestagging.sentence_model
            models_cleaned.append("SentenceTransformer")
    except Exception as e:
        print(f"⚠️ Error cleaning up framestagging models: {e}")
    
    # Cleanup audio models
    try:
        if hasattr(audioprocessor, 'model'):
            del audioprocessor.model
            models_cleaned.append("Whisper model")
        if hasattr(audioprocessor, 'yamnet_model'):
            del audioprocessor.yamnet_model
            models_cleaned.append("YAMNet model")
        if hasattr(audioprocessor, 'music_classifier'):
            del audioprocessor.music_classifier
            models_cleaned.append("Music classifier")
    except Exception as e:
        print(f"⚠️ Error cleaning up audio models: {e}")
    
    # Cleanup image processing models
    try:
        # Check if ImageProcessor exists and has models to cleanup
        import imageprocessor
        import sys
        
        # Look for ImageProcessor instances in memory
        for obj_name in dir(sys.modules.get('imageprocessor', {})):
            obj = getattr(imageprocessor, obj_name, None)
            if hasattr(obj, 'object_detector'):
                del obj.object_detector
                models_cleaned.append("Object detector")
                
        # Clean up global EasyOCR reader
        if hasattr(imageprocessor, 'global_easyocr_reader'):
            del imageprocessor.global_easyocr_reader
            models_cleaned.append("EasyOCR reader")
            
    except Exception as e:
        print(f"⚠️ Error cleaning up image processing models: {e}")
    
    # Cleanup EasyOCR models specifically
    try:
        import easyocr
        # Clear EasyOCR model cache if possible
        if hasattr(easyocr, 'Reader'):
            # Force cleanup of any existing readers
            import gc
            for obj in gc.get_objects():
                if isinstance(obj, easyocr.Reader):
                    if hasattr(obj, 'detector'):
                        del obj.detector
                    if hasattr(obj, 'recognizer'):
                        del obj.recognizer
                    models_cleaned.append("EasyOCR detector/recognizer")
    except Exception as e:
        print(f"⚠️ Error cleaning up EasyOCR models: {e}")
    
        gc.collect()
    
    # Clear all ML framework caches
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if hasattr(torch, '_C') and hasattr(torch._C, '_cuda_clearCublasWorkspaces'):
            torch._C._cuda_clearCublasWorkspaces()
    except ImportError:
        pass
    
    # Clear TensorFlow memory
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
        tf.config.experimental.reset_memory_growth()
    except (ImportError, AttributeError):
        pass
    
    # Clear transformers cache
    try:
        import transformers
        transformers.utils.hub.cached_file.cache_clear()
    except (ImportError, AttributeError):
        pass
    
    memory_after = get_memory_usage()
    memory_freed = memory_before - memory_after
    
    if models_cleaned:
        print(f"✅ Models cleaned: {', '.join(models_cleaned)}")
    
    if memory_freed > 0:
        print(f"🧹 Memory freed: {memory_freed:.1f} MB ({memory_before:.1f} MB → {memory_after:.1f} MB)")
    else:
        print(f"🧹 Model cleanup complete (Memory: {memory_after:.1f} MB)")
    
    return len(models_cleaned) > 0

def detect_caption_conflicts(caption, context_embedding, quality_score, threshold=-2):
    """
    Detect conflicts between generated caption and overall video context.
    
    Args:
        caption (str): Generated caption
        context_embedding: Pre-computed embedding of overall video context
        quality_score (int): Quality score from rotation corrector
        threshold (int): Quality score threshold for conflicts
    
    Returns:
        tuple: (has_conflicts, conflict_list)
    """
    conflicts = []
    
    # Check quality score threshold
    if quality_score < threshold:
        conflicts.append({
            "type": "low_quality_score", 
            "severity": "high",
            "details": f"Quality score {quality_score} below threshold {threshold}"
        })
    
    # Check semantic conflict with overall video context
    if context_embedding is not None and caption:
        try:
            # Get embedding for caption only (context already embedded)
            caption_embedding = framestagging.sentence_model.encode([caption])
            
            # Calculate similarity
            similarity = util.pytorch_cos_sim(caption_embedding, context_embedding).item()
            
            # If similarity is very low, there might be a conflict
            if similarity < 0.3:  # Threshold for semantic conflict
                conflicts.append({
                    "type": "semantic_conflict",
                    "severity": "medium",
                    "details": f"Caption has low semantic similarity ({similarity:.2f}) with overall video context"
                })
        except Exception as e:
            # If semantic analysis fails, just skip this check
            pass
    
    # Check for very generic/low-quality captions
    if len(caption.split()) < 4:
        conflicts.append({
            "type": "too_generic",
            "severity": "medium", 
            "details": f"Caption too generic/short: '{caption}'"
        })
    
    return len(conflicts) > 0, conflicts


def get_overall_video_context(selected_frames):
    """
    Get overall video context using GPT-4o mini vision analysis.
    This runs in parallel with frame processing.
    """
    if not selected_frames:
        return None
    
    # Use framestagging function for vision analysis
    return framestagging.analyze_keyframes_with_gpt4o_vision(selected_frames)

def tag_video_smart_conflict_resolution(vid_path, use_moondream=False, use_moondream_api=True, compress_frames=False, max_pixels=1000, stop_flag=None):
    """
    Enhanced video tagging with concurrent processing of all major operations.
    
    Args:
        vid_path (str): Path to video file
        use_moondream (bool): Whether to use Moondream2 local model (deprecated)
        use_moondream_api (bool): Whether to use Moondream API for initial captioning (default: True)
        compress_frames (bool): Whether to compress frames to max_pixels during extraction
        max_pixels (int): Maximum pixels for longest side if compressing frames
        stop_flag (callable): Optional function that returns True when processing should stop
    
    Process:
    1. Extract frames
    2. Run concurrently: OCR, GPT-4o Vision, Moondream API, Audio processing
    3. Combine results for final analysis
    """
    
    # Set up API client with stop callback
    try:
        client = api_client.get_api_client()
        print("✅ API client configured")
    except Exception as e:
        print(f"⚠️ Could not configure API client: {e}")
    
    memory_start = get_memory_usage()
    
    print(f"🔍 Starting CONCURRENT video analysis for: {vid_path}")
    print(f"🧵 Using multithreading for: OCR, GPT-4o Vision, Moondream API, Audio")
    print(f"📊 Initial memory usage: {memory_start:.1f} MB")
    
    # Step 1: Extract frames
    print("Step 1: Extracting frames...")
    frames_dir, frame_metadata = split_frames(vid_path)
    
    # Validate frame extraction was successful
    if frames_dir is None or not os.path.exists(frames_dir):
        error_msg = "Frame extraction failed: frames directory not created"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
        
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
    if not frame_files:
        error_msg = f"Frame extraction failed: no frames found in {frames_dir}"
        print(f"❌ {error_msg}")
        # Clean up frames directory  
        try:
            shutil.rmtree(frames_dir)
        except:
            pass
        raise RuntimeError(error_msg)
    
    print(f"✓ Frames extracted to: {frames_dir} ({len(frame_files)} frames)")
    
    # Step 2: Select keyframes for processing
    print("Step 2: Selecting keyframes...")
    total_keyframes = len(frame_files)
    num_to_send = calculate_keyframes_to_send(total_keyframes)
    selected_frames = select_representative_keyframes(frames_dir, num_to_send)
    print(f"✓ Selected {len(selected_frames)} keyframes from {total_keyframes} total frames")
    
    # Step 3: Extract video metadata for analysis
    print("Step 3: Extracting video metadata...")
    video_metadata = extract_video_metadata(vid_path)
    trimmed_metadata = trim_metadata_for_api(video_metadata)
    
    # Extract location information
    processed_location = extract_and_store_location_coordinates(video_metadata)
    if processed_location:
        print(f"✓ Location processed: {processed_location}")
    else:
        print("✓ No location information found in metadata")
    
    print(f"✓ Video metadata extracted")
    
    # Step 4: Define concurrent processing functions
    def process_ocr():
        """Extract text from keyframes using local EasyOCR"""
        try:
            print("🔤 [OCR Thread] Starting text extraction (local EasyOCR)...")
            text_data = extract_prominent_text_from_frames(selected_frames)
            
            print(f"✅ [OCR Thread] Text extraction complete: {len(text_data.get('prominent_text', []))} prominent texts")
            return text_data
            
        except Exception as e:
            print(f"❌ [OCR Thread] Text extraction failed: {e}")
            return {'prominent_text': [], 'frame_details': {}}
    
    def process_moondream():
        """Process frames with Moondream API"""
        try:
            print("🌙 [Moondream Thread] Starting API processing...")
            
            # Use API server for Moondream batch processing
            api_response = api_client.get_api_client().moondream_batch_process(selected_frames)
            
            if api_response and 'results' in api_response:
                frame_captions = api_response['results']
                print(f"✅ [Moondream Thread] API processing complete: {len(frame_captions)} results")
                return frame_captions
            else:
                raise Exception("Invalid API response for Moondream processing")
                
        except Exception as e:
            print(f"⚠️ [Moondream Thread] API server failed, falling back to local processing: {e}")
            # Fallback to original function
            try:
                frame_captions = process_frames_with_moondream_api(selected_frames)
                print(f"✅ [Moondream Thread] Fallback processing complete")
                return frame_captions
            except Exception as fallback_error:
                print(f"❌ [Moondream Thread] Both API and fallback failed: {fallback_error}")
                return []
    
    def process_audio():
        """Extract and analyze audio using Whisper + GPT-4o"""
        try:
            print("🎵 [Audio Thread] Starting audio processing...")
            
            # Use audioanalyzer for Whisper transcription + OpenAI analysis
            audio_summary = analyze_audio_with_openai(vid_path)
            
            print(f"✅ [Audio Thread] Audio processing complete")
            return audio_summary
            
        except Exception as e:
            print(f"❌ [Audio Thread] Audio processing failed: {e}")
            return None
    
    # Step 5: Run all operations concurrently
    print("\nStep 4: Starting concurrent processing...")
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="VideoProcessor") as executor:
        # Submit all tasks
        futures = {
            'ocr': executor.submit(process_ocr),
            'moondream': executor.submit(process_moondream),
            'audio': executor.submit(process_audio)
        }
        
        print("🧵 All threads started, waiting for completion...")
        
        # Wait for all tasks to complete with progress reporting
        completed_tasks = []
        text_data = None
        frame_captions = None
        audio_summary = None
        
        # Use as_completed to show progress
        for future in concurrent.futures.as_completed(futures.values()):
            for task_name, task_future in futures.items():
                if future == task_future:
                    completed_tasks.append(task_name)
                    print(f"✅ [{task_name.upper()}] Completed ({len(completed_tasks)}/3)")
                    
                    # Collect results
                    try:
                        result = future.result()
                        if task_name == 'ocr':
                            text_data = result
                        elif task_name == 'moondream':
                            frame_captions = result
                        elif task_name == 'audio':
                            audio_summary = result
                    except Exception as e:
                        print(f"❌ [{task_name.upper()}] Failed: {e}")
                        # Set default values for failed tasks
                        if task_name == 'ocr':
                            text_data = {'prominent_text': [], 'frame_details': {}}
                        elif task_name == 'moondream':
                            frame_captions = []
                        elif task_name == 'audio':
                            audio_summary = None
                    break
    
    concurrent_time = time.time() - start_time
    memory_after_concurrent = get_memory_usage()
    
    print(f"✅ Concurrent processing complete in {concurrent_time:.2f}s")
    print(f"📊 Memory after concurrent processing: {memory_after_concurrent:.1f} MB (+{memory_after_concurrent-memory_start:.1f} MB)")
    
    # Light cleanup after concurrent processing
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except ImportError:
        pass
    
    memory_after_cleanup = get_memory_usage()
    if memory_after_cleanup < memory_after_concurrent:
        print(f"🧹 Post-concurrent cleanup: {memory_after_concurrent:.1f} MB → {memory_after_cleanup:.1f} MB (-{memory_after_concurrent-memory_after_cleanup:.1f} MB)")
    
    print("\nStep 5: Final comprehensive analysis with GPT-4o...")
    # Extract actual captions from frame_captions if available
    caption_texts = []
    if frame_captions and isinstance(frame_captions, list):
        for item in frame_captions:
            if isinstance(item, dict) and 'caption' in item:
                caption_texts.append(item['caption'])
    
    # Call GPT-4o with actual frame captions (not file paths)
    result = call_gpt4o(caption_texts, audio_summary, video_metadata, processed_location=processed_location)
    
    parsed = json.loads(result)
    memory_after_gpt = get_memory_usage()
    print(f"✓ Comprehensive analysis complete (Memory: {memory_after_gpt:.1f} MB)")
    
    print("\nStep 6: Saving results...")
    save_video_metadata(vid_path, parsed, processed_location=processed_location)
    print("✓ Results saved successfully")
    
    # Remove frames directory after processing
    remove_frames_dir(frames_dir)
    
    # Store result before cleanup to prevent memory leaks
    result_copy = result
    
    # Use modular cleanup function
    print("Step 7: Video processing cleanup...")
    memory_freed_by_cleanup = cleanup_video_processing(
        selected_frames=selected_frames,
        text_data=text_data,
        audio_summary=audio_summary,
        video_metadata=video_metadata,
        result=result,
        parsed=parsed,
        processed_location=processed_location,
        frame_files=frame_files,
        total_keyframes=total_keyframes,
        num_to_send=num_to_send,
        frame_metadata=frame_metadata
    )
    
    memory_final = get_memory_usage()
    memory_delta = memory_final - memory_start
    
    if memory_freed_by_cleanup > 0:
        print(f"🧹 Cleanup freed: {memory_freed_by_cleanup:.1f} MB")
    
    print(f"✓ Processing complete")
    print(f"📊 Final memory usage: {memory_final:.1f} MB")
    print(f"📊 Net memory change: {memory_delta:+.1f} MB")
    print(f"⚡ Total concurrent processing time: {concurrent_time:.2f}s")
    
    if memory_delta > 10:
        print(f"⚠️  WARNING: Memory leak detected! {memory_delta:.1f} MB increase")
    
    return result_copy

def cleanup_video_processing(selected_frames=None, text_data=None, audio_summary=None, video_metadata=None, result=None, parsed=None, processed_location=None, frame_files=None, total_keyframes=None, num_to_send=None, frame_metadata=None):
    """
    Cleanup function for video processing to prevent memory leaks.
    
    Args:
        **kwargs: Any variables that need to be cleaned up
        
    Returns:
        float: Amount of memory freed in MB
    """
    memory_before = get_memory_usage()
    
    # Delete all provided variables
    cleanup_vars = {
        'selected_frames': selected_frames,
        'text_data': text_data,
        'audio_summary': audio_summary,
        'video_metadata': video_metadata,
        'result': result,
        'parsed': parsed,
        'processed_location': processed_location,
        'frame_files': frame_files,
        'total_keyframes': total_keyframes,
        'num_to_send': num_to_send,
        'frame_metadata': frame_metadata
    }
    
    deleted_vars = []
    for var_name, var_value in cleanup_vars.items():
        if var_value is not None:
            del var_value
            deleted_vars.append(var_name)
    
    # Multiple rounds of garbage collection
    for _ in range(3):
        gc.collect()
    
    # Clear ML framework caches but keep models loaded
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        # Clear session but keep models loaded for batch processing
        tf.keras.backend.clear_session()
    except ImportError:
        pass
    
    memory_after = get_memory_usage()
    memory_freed = memory_before - memory_after
    
    if deleted_vars:
        print(f"🧹 Video cleanup: deleted {', '.join(deleted_vars)}")
    if memory_freed > 0:
        print(f"🧹 Video cleanup freed: {memory_freed:.1f} MB")
    
    return memory_freed

if __name__ == "__main__":
    if len(sys.argv) > 1:
        vid_path = sys.argv[1]
        # Check for optional model flags
        use_moondream = len(sys.argv) > 2 and sys.argv[2].lower() in ['moondream', 'moon', 'md']
        use_moondream_api = len(sys.argv) > 2 and sys.argv[2].lower() in ['api', 'moondream_api', 'moon_api']
        
        if use_moondream:
            print("🌙 Using Moondream2 local model for initial captioning")
        elif use_moondream_api:
            print("🚀 Using Moondream API for initial captioning (fastest option)")
        else:
            print("🤖 Using vision analysis for initial captioning")
            
        # Use frame compression by default for 21% speed improvement
        result = tag_video_smart_conflict_resolution(
            vid_path, use_moondream=use_moondream, use_moondream_api=use_moondream_api, compress_frames=True, max_pixels=1000
        )
        print(result)
        
        # Optional: Uncomment the line below if this is the last video you're processing
        # and you want to free up all model memory
        # cleanup_all_models()
    else:
        vid_path = input("video path: ")
        
        # Offer model choice
        print("\nChoose vision model:")
        print("1. Vision analysis (default, balanced)")
        print("2. Moondream2 (better quality, slower)")
        print("3. Moondream API (fastest, good quality)")
        
        model_choice = input("Enter choice (1-3, default=1): ").strip()
        
        use_moondream = model_choice == "2"
        use_moondream_api = model_choice == "3"
        
        if use_moondream:
            print("🌙 Using Moondream2 local model for initial captioning")
        elif use_moondream_api:
            print("🚀 Using Moondream API for initial captioning")
        else:
            print("🤖 Using vision analysis for initial captioning")
            
        result = tag_video_smart_conflict_resolution(
            vid_path, use_moondream=use_moondream, use_moondream_api=use_moondream_api, compress_frames=True, max_pixels=1000
        )
        print(result)
        
        # Optional: Uncomment the line below if this is the last video you're processing
        # and you want to free up all model memory
        # cleanup_all_models()

