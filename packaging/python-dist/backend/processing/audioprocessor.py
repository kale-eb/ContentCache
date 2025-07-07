import os
import sys
import subprocess
import tempfile
import signal
import time
from functools import wraps
from pathlib import Path
from config import get_models_cache_dir, get_ffmpeg_path, get_ffprobe_path

# Audio processing imports
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import soundfile as sf
from transformers import pipeline
import warnings
import yaml
from datetime import datetime
import shutil
from config import get_audio_metadata_path

# Try to import optional dependencies with graceful fallbacks
try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
    print("✅ Torch available for audio processing")
except ImportError as e:
    print(f"⚠️ Torch not available: {e}")
    TORCH_AVAILABLE = False
    torch = None
    torchaudio = None

# Suppress various warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*FP16 is not supported on CPU.*")

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Set up model cache directory using config
MODEL_CACHE_DIR = get_models_cache_dir()
Path(MODEL_CACHE_DIR).mkdir(exist_ok=True)

# Performance logging configuration
ENABLE_AUDIO_LOGGING = os.getenv("VERBOSE_LOGGING", "true").lower() == "true"
ENABLE_AUDIO_DEBUG = os.getenv("DEBUG_LOGGING", "false").lower() == "true"

def audio_print(*args, **kwargs):
    """Audio processing logging that can be toggled off"""
    if ENABLE_AUDIO_LOGGING:
        print(*args, **kwargs)

def audio_debug(*args, **kwargs):
    """Audio debug logging for detailed analysis"""
    if ENABLE_AUDIO_DEBUG:
        print(*args, **kwargs)

# ---------- YAMNet --------------------------------------------------------
# Set up TensorFlow Hub cache directory
os.environ['TFHUB_CACHE_DIR'] = os.path.join(MODEL_CACHE_DIR, 'tfhub')
Path(os.environ['TFHUB_CACHE_DIR']).mkdir(exist_ok=True)

yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
# YAMNet class names ─ newer TF‑Hub builds don't expose `class_names`.
try:
    yamnet_labels = yamnet_model.class_names.numpy()  # ≤ TF‑Hub v0.12
    yamnet_labels = [c.decode() for c in yamnet_labels]
except AttributeError:
    # Fallback: download the CSV with 521 labels
    import urllib.request, csv, tempfile
    labels_csv = tf.keras.utils.get_file(
        "yamnet_class_map.csv",
        "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv",
        cache_dir=MODEL_CACHE_DIR
    )
    with open(labels_csv, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        yamnet_labels = [row[2] for row in reader]  # display_name column

# Add Hugging Face zero-shot audio classifier and labels
try:
    music_classifier = pipeline(
        "zero-shot-audio-classification",
        model="laion/clap-htsat-unfused",
        cache_dir=MODEL_CACHE_DIR
    )
except Exception as e:
    music_classifier = None
    print(f"Warning: Could not load Hugging Face audio classifier: {e}")

music_labels = [
    "dramatic music", "uplifting music", "sad music", "happy music", "suspenseful music", "relaxing music", "energetic music", "ambient music", "classical music", "electronic music", "jazz music", "rock music", "pop music", "cinematic music", "acoustic music", "dark music", "inspirational music"
]

def yamnet_tags(wav_path, top_n=3, thresh=0.25):
    """
    Return up to `top_n` YAMNet labels (as *markdown* strings) whose
    mean score ≥ `thresh` for the given wav clip.
    """
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:      # stereo to mono
        audio = audio.mean(axis=1)
    scores, _, _ = yamnet_model(audio)
    mean_scores = scores.numpy().mean(axis=0)
    ids = mean_scores.argsort()[-top_n:][::-1]
    
    # Debug: show top scores
    audio_debug(f"Top {top_n} YAMNet scores:")
    for i in ids:
        audio_debug(f"  {yamnet_labels[i]}: {mean_scores[i]:.3f}")
    
    tags = [
        f"*{yamnet_labels[i]}*"
        for i in ids
        if mean_scores[i] >= thresh
    ]
    # If only tag is '*music*', try to get a more descriptive label
    if tags == ["*music*"] and music_classifier is not None:
        try:
            result = music_classifier(wav_path, candidate_labels=music_labels)
            if result and 'labels' in result and len(result['labels']) > 0:
                tags = [f"*{result['labels'][0]}*"]
        except Exception as e:
            print(f"Warning: Hugging Face music classifier failed: {e}")
    return tags


# ----------  helper -------------
def has_audio(filename):
    # More accurate method: specifically check for audio streams
    result = subprocess.run(
        [get_ffprobe_path(), "-v", "error", "-select_streams", "a", "-show_entries", 
         "stream=codec_type", "-of", "csv=p=0", filename],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    
    # Count actual audio streams
    audio_streams = result.stdout.strip().split('\n') if result.stdout.strip() else []
    audio_count = len([s for s in audio_streams if s == 'audio'])
    
    audio_debug(f"Audio detection for {os.path.basename(filename)}: {audio_count} audio streams")
    return audio_count


def cleanup_temp_files():
    """Clean up any remaining temporary audio files"""
    import glob
    temp_dir = tempfile.gettempdir()
    
    # Look for temporary WAV files that might have been left behind
    temp_patterns = [
        os.path.join(temp_dir, "tmp*.wav"),
        os.path.join(temp_dir, "temp*.wav"),
        "/tmp/tmp*.wav",  # Unix systems
        "/tmp/temp*.wav"
    ]
    
    cleaned_count = 0
    for pattern in temp_patterns:
        try:
            temp_files = glob.glob(pattern)
            for temp_file in temp_files:
                try:
                    # Only remove files that are older than 1 hour to avoid conflicts
                    file_age = time.time() - os.path.getmtime(temp_file)
                    if file_age > 3600:  # 1 hour
                        os.remove(temp_file)
                        cleaned_count += 1
                        audio_debug(f"Cleaned up old temporary file: {temp_file}")
                except Exception as e:
                    audio_debug(f"Could not clean up {temp_file}: {e}")
        except Exception:
            pass  # Pattern might not exist on this system
    
    if cleaned_count > 0:
        audio_print(f"🧹 Cleaned up {cleaned_count} old temporary audio files")

def extract_clip(src, start, end):
    """Return path to a temp wav file for [start,end] seconds."""
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    
    try:
        subprocess.run(
            [get_ffmpeg_path(), "-loglevel", "quiet", "-y",
             "-ss", str(start), "-to", str(end),
             "-i", src, "-ar", "32000", "-ac", "1", tmp],
            check=True
        )
        return tmp
    except subprocess.CalledProcessError as e:
        # Clean up temp file if ffmpeg fails
        try:
            os.remove(tmp)
        except:
            pass
        raise Exception(f"Failed to extract audio clip: {e}")


# ----------  main ---------------
def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Operation timed out")

def with_timeout(timeout_seconds):
    """Decorator to add timeout to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set up signal handler for timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel the alarm
                return result
            except TimeoutError:
                audio_print(f"⚠️ {func.__name__} timed out after {timeout_seconds}s")
                raise
            finally:
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator

def safe_transcribe(model, file_path, use_timeout=False):
    """Safely transcribe audio with optional timeout"""
    if use_timeout:
        try:
            # Only use timeout in main thread
            import threading
            if threading.current_thread() is threading.main_thread():
                audio_print(f"🎤 Starting Whisper transcription (max 2 minutes)...")
                return with_timeout(120)(model.transcribe)(file_path)
            else:
                audio_print(f"🎤 Starting Whisper transcription (no timeout in thread)...")
                return model.transcribe(file_path)
        except Exception:
            # Fallback to no timeout
            audio_print(f"🎤 Starting Whisper transcription (no timeout)...")
            return model.transcribe(file_path)
    else:
        audio_print(f"🎤 Starting Whisper transcription...")
        return model.transcribe(file_path)

def process_audio(file_path):
    audio_print("Analyzing audio")
    
    # Check if Whisper is available
    if not WHISPER_AVAILABLE or model is None:
        audio_print("⚠️ Whisper not available, using fallback audio analysis")
        return process_audio_fallback(file_path)
    
    if has_audio(file_path) == 0:
        audio_print("✅ No audio streams detected, skipping audio analysis")
        return {"segments": []}
    
    # Check if this is an audio file (not a video file)
    file_ext = os.path.splitext(file_path)[1].lower()
    audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
    is_audio_file = file_ext in audio_extensions
    
    # Extract filename for audio files
    filename_info = None
    if is_audio_file:
        filename = os.path.basename(file_path)
        filename_without_ext = os.path.splitext(filename)[0]
        filename_info = {
            "original_filename": filename,
            "title": filename_without_ext,
            "is_audio_file": True
        }
        audio_print(f"Audio file detected: {filename_without_ext}")

    try:
        result = safe_transcribe(model, file_path, use_timeout=False)  # Disable timeout in threads
        segments_out = []
        audio_debug(f"Whisper result: {result}")  # This can be very large - debug only
        audio_print(f"Segments found: {len(result.get('segments', []))}")
    except TimeoutError:
        audio_print("❌ Whisper transcription timed out, skipping audio analysis")
        return {"segments": []}
    except Exception as e:
        audio_print(f"❌ Whisper transcription failed: {e}")
        return {"segments": []}

    total_segments = len(result.get("segments", []))
    for i, seg in enumerate(result.get("segments", [])):
        # If it's speech, keep Whisper text
        if seg["no_speech_prob"] <= 0.25:
            segments_out.append({
                "start": round(seg["start"], 2),
                "end":   round(seg["end"],   2),
                "text":  seg["text"].strip()
            })
        else:
            # Non‑speech → run YAMNet
            audio_debug(f"Non-speech segment detected, running YAMNet on {seg['start']}-{seg['end']}")
            clip = None
            try:
                clip = extract_clip(file_path, seg["start"], seg["end"])
                audio_debug(f"Created clip: {clip}")
                tags = yamnet_tags(clip)
                audio_debug(f"YAMNet tags: {tags}")
                
                if tags:
                    segments_out.append({
                        "start": round(seg["start"], 2),
                        "end":   round(seg["end"],   2),
                        "text":  "",
                        "audio_tags": tags
                    })
                    audio_debug(f"Added segment with tags: {tags}")
                else:
                    audio_debug("No tags found, segment dropped")
                    
            except Exception as e:
                audio_print(f"⚠️ Error processing segment {seg['start']}-{seg['end']}: {e}")
            finally:
                # Always clean up temporary clip file
                if clip and os.path.exists(clip):
                    try:
                        os.remove(clip)
                        audio_debug(f"Cleaned up temporary clip: {clip}")
                    except Exception as e:
                        audio_print(f"⚠️ Failed to clean up temporary clip {clip}: {e}")

    # If no segments were found, run YAMNet on the entire file
    if not segments_out:
        audio_print("No segments found, running YAMNet on entire file")
        
        # Convert entire file to WAV for YAMNet processing
        audio_debug("Converting entire file to WAV for YAMNet analysis")
        fd, temp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        
        try:
            # Convert entire file to WAV
            subprocess.run(
                [get_ffmpeg_path(), "-loglevel", "quiet", "-y",
                 "-i", file_path, "-ar", "32000", "-ac", "1", temp_wav],
                check=True
            )
            audio_debug(f"Created temporary WAV file: {temp_wav}")
            
            # Run YAMNet on the converted WAV file
            tags = yamnet_tags(temp_wav, top_n=5, thresh=0.1)  # Lower threshold and more tags
            audio_debug(f"YAMNet tags for entire file: {tags}")
            
            # If still no tags, try even lower threshold
            if not tags:
                audio_debug("No tags found with thresh=0.1, trying thresh=0.05")
                tags = yamnet_tags(temp_wav, top_n=5, thresh=0.05)
                audio_debug(f"YAMNet tags with lower threshold: {tags}")
                
        except subprocess.CalledProcessError as e:
            audio_print(f"⚠️ Failed to convert file to WAV: {e}")
            tags = []
        except Exception as e:
            audio_print(f"⚠️ Error during YAMNet analysis: {e}")
            tags = []
        finally:
            # Always clean up temporary WAV file, even if it doesn't exist
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                    audio_debug(f"✅ Cleaned up temporary WAV file: {temp_wav}")
                except Exception as e:
                    audio_print(f"⚠️ Failed to clean up temporary WAV file {temp_wav}: {e}")
            else:
                audio_debug(f"Temporary WAV file {temp_wav} already cleaned up or never created")
        
        if tags:
            # Get file duration for the segment
            duration_result = subprocess.run(
                [get_ffprobe_path(), "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", file_path],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            try:
                duration = float(duration_result.stdout.strip())
            except:
                duration = 0.0
            
            segments_out.append({
                "start": 0.0,
                "end": round(duration, 2),
                "text": "",
                "audio_tags": tags
            })
            audio_debug(f"Added entire file as segment with tags: {tags}")
        else:
            audio_debug("No tags found for entire file")

    audio_debug(f"Final segments_out: {segments_out}")  # This can be large - debug only
    
    # Return segments with optional filename info for audio files
    result_data = {
        "segments": segments_out
    }
    
    if filename_info:
        result_data["filename_info"] = filename_info
        audio_print(f"Including filename info: {filename_info['title']}")
    
    # Store result before cleanup
    result_data_copy = result_data.copy()
    
    # Use modular cleanup function
    cleanup_audio_processing(
        result=result,
        segments_out=segments_out,
        filename_info=filename_info,
        result_data=result_data
    )
    
    return result_data_copy

def cleanup_audio_processing(result=None, segments_out=None, filename_info=None, result_data=None):
    """
    Cleanup function for audio processing to prevent memory leaks.
    
    Args:
        result: Whisper transcription result
        segments_out: Processed segments list
        filename_info: Filename metadata dict
        result_data: Final result dict
        
    Returns:
        float: Amount of memory freed in MB
    """
    try:
        import psutil
        import gc
        
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Delete provided variables
        cleanup_vars = {
            'result': result,
            'segments_out': segments_out,
            'filename_info': filename_info,
            'result_data': result_data
        }
        
        deleted_vars = []
        for var_name, var_value in cleanup_vars.items():
            if var_value is not None:
                del var_value
                deleted_vars.append(var_name)
        
        # Multiple rounds of garbage collection for audio processing
        for _ in range(2):
            gc.collect()
        
        # Clear ML framework caches
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except ImportError:
            pass
            
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        # Clean up temporary files
        cleanup_temp_files()
        
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        memory_freed = memory_before - memory_after
        
        if deleted_vars and memory_freed > 0:
            print(f"🧹 Audio cleanup: deleted {', '.join(deleted_vars)}, freed {memory_freed:.1f} MB")
        
        return memory_freed
        
    except ImportError:
        # If psutil not available, just do cleanup
        import gc
        for _ in range(2):
            gc.collect()
        cleanup_temp_files()
        return 0.0

# FFmpeg functions are now imported from centralized config

def process_audio_fallback(file_path):
    """Fallback audio processing when Whisper is not available"""
    audio_print("🔊 Using fallback audio analysis (Whisper not available)")
    
    if has_audio(file_path) == 0:
        audio_print("✅ No audio streams detected, skipping audio analysis")
        return {"segments": []}
    
    # Check if this is an audio file (not a video file)
    file_ext = os.path.splitext(file_path)[1].lower()
    audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
    is_audio_file = file_ext in audio_extensions
    
    # Extract filename for audio files
    filename_info = None
    if is_audio_file:
        filename = os.path.basename(file_path)
        filename_without_ext = os.path.splitext(filename)[0]
        filename_info = {
            "original_filename": filename,
            "title": filename_without_ext,
            "is_audio_file": True
        }
        audio_print(f"Audio file detected: {filename_without_ext}")
    
    # Since Whisper isn't available, try to analyze using YAMNet only
    audio_print("Running YAMNet analysis on entire file (no transcription available)")
    
    # Convert entire file to WAV for YAMNet processing
    fd, temp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    
    segments_out = []
    
    try:
        # Convert entire file to WAV
        subprocess.run(
            [get_ffmpeg_path(), "-loglevel", "quiet", "-y",
             "-i", file_path, "-ar", "32000", "-ac", "1", temp_wav],
            check=True
        )
        audio_debug(f"Created temporary WAV file: {temp_wav}")
        
        # Run YAMNet on the converted WAV file
        tags = yamnet_tags(temp_wav, top_n=5, thresh=0.1)  # Lower threshold and more tags
        audio_debug(f"YAMNet tags for entire file: {tags}")
        
        # If still no tags, try even lower threshold
        if not tags:
            audio_debug("No tags found with thresh=0.1, trying thresh=0.05")
            tags = yamnet_tags(temp_wav, top_n=5, thresh=0.05)
            audio_debug(f"YAMNet tags with lower threshold: {tags}")
            
    except subprocess.CalledProcessError as e:
        audio_print(f"⚠️ Failed to convert file to WAV: {e}")
        tags = []
    except Exception as e:
        audio_print(f"⚠️ Error during YAMNet analysis: {e}")
        tags = []
    finally:
        # Always clean up temporary WAV file
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
                audio_debug(f"✅ Cleaned up temporary WAV file: {temp_wav}")
            except Exception as e:
                audio_print(f"⚠️ Failed to clean up temporary WAV file {temp_wav}: {e}")
    
    if tags:
        # Get file duration for the segment
        try:
            duration_result = subprocess.run(
                [get_ffprobe_path(), "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", file_path],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            duration = float(duration_result.stdout.strip())
        except:
            duration = 0.0
        
        segments_out.append({
            "start": 0.0,
            "end": round(duration, 2),
            "text": "",
            "audio_tags": tags
        })
        audio_debug(f"Added entire file as segment with tags: {tags}")
    else:
        audio_debug("No tags found for entire file")
    
    # Return segments with optional filename info for audio files
    result_data = {
        "segments": segments_out
    }
    
    if filename_info:
        result_data["filename_info"] = filename_info
        audio_print(f"Including filename info: {filename_info['title']}")
    
    return result_data

def setup_ffmpeg_path():
    """Set up the PATH environment variable to include bundled ffmpeg directory"""
    try:
        # Use centralized config to get ffmpeg path
        ffmpeg_path = get_ffmpeg_path()
        
        # Extract the directory containing ffmpeg
        binaries_dir = os.path.dirname(ffmpeg_path)
        
        # Add the binaries directory to PATH if it's not already there
        current_path = os.environ.get('PATH', '')
        if binaries_dir not in current_path:
            os.environ['PATH'] = f"{binaries_dir}{os.pathsep}{current_path}"
            audio_print(f"✅ Added {binaries_dir} to PATH for Whisper")
        else:
            audio_debug(f"Directory already in PATH: {binaries_dir}")
        
        audio_print(f"✅ Set up ffmpeg PATH: {ffmpeg_path}")
        return True
        
    except Exception as e:
        audio_print(f"⚠️ Failed to set up ffmpeg PATH: {e}")
        return False

# Import Whisper and set up ffmpeg path
try:
    # Set up ffmpeg path before importing Whisper
    setup_ffmpeg_path()
    
    import whisper
    WHISPER_AVAILABLE = True
    audio_print("✅ Whisper library loaded successfully")
except ImportError as e:
    audio_print(f"⚠️ Whisper not available: {e}")
    WHISPER_AVAILABLE = False
    whisper = None

# Initialize Whisper model after setting up the environment
model = None
if WHISPER_AVAILABLE and whisper:
    try:
        model = whisper.load_model("base", download_root=MODEL_CACHE_DIR)
        audio_print("✅ Whisper model loaded successfully")
    except Exception as e:
        audio_print(f"⚠️ Failed to load Whisper model: {e}")
        model = None
        WHISPER_AVAILABLE = False