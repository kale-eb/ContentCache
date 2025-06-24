import os
import json
import subprocess
import tempfile
from audioprocessor import process_audio
import requests

def get_ffmpeg_path():
    """Get the path to the bundled ffmpeg binary or system ffmpeg."""
    # Try bundled ffmpeg first (in packaged app)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(current_dir, '..', '..', '..', '..', 'Resources', 'binaries', 'ffmpeg'),
        os.path.join(current_dir, '..', 'app.asar.unpacked', 'binaries', 'ffmpeg'),
        os.path.join(current_dir, 'binaries', 'ffmpeg'),
    ]
    
    for bundled_ffmpeg in possible_paths:
        if os.path.exists(bundled_ffmpeg):
            return bundled_ffmpeg
    
    # Fallback to system ffmpeg
    return 'ffmpeg'

def get_ffprobe_path():
    """Get the path to the bundled ffprobe binary or system ffprobe."""
    # Try bundled ffprobe first (in packaged app)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(current_dir, '..', '..', '..', '..', 'Resources', 'binaries', 'ffprobe'),
        os.path.join(current_dir, '..', 'app.asar.unpacked', 'binaries', 'ffprobe'),
        os.path.join(current_dir, 'binaries', 'ffprobe'),
    ]
    
    for bundled_ffprobe in possible_paths:
        if os.path.exists(bundled_ffprobe):
            return bundled_ffprobe
    
    # Fallback to system ffprobe
    return 'ffprobe'

def extract_audio_from_video(video_path):
    """
    Extract audio from video file to a temporary WAV file.
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        str: Path to temporary audio file, or None if extraction fails
    """
    try:
        # Create temporary audio file
        fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        
        # Extract audio using ffmpeg
        result = subprocess.run([
            get_ffmpeg_path(), 
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit audio
            "-ar", "16000",  # 16kHz sample rate for Whisper
            "-ac", "1",  # Mono
            "-y",  # Overwrite output
            temp_audio_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"⚠️ FFmpeg failed to extract audio: {result.stderr}")
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            return None
            
        return temp_audio_path
        
    except Exception as e:
        print(f"⚠️ Error extracting audio from video: {e}")
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return None

def analyze_audio_comprehensive(video_path):
    """
    Comprehensive audio analysis that works with video files.
    First extracts audio from video, then processes it with audioprocessor.
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        dict: Audio analysis results with segments and metadata
    """
    print(f"🎤 Starting comprehensive audio analysis for: {os.path.basename(video_path)}")
    
    # Check if input is already an audio file
    file_ext = os.path.splitext(video_path)[1].lower()
    audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
    
    if file_ext in audio_extensions:
        print("📻 Input is audio file, processing directly...")
        return process_audio(video_path)
    
    # Extract audio from video
    print("🎬 Extracting audio from video...")
    temp_audio_path = extract_audio_from_video(video_path)
    
    if not temp_audio_path:
        print("❌ Failed to extract audio from video")
        return {"segments": []}
    
    try:
        print("🔊 Processing extracted audio...")
        # Process the extracted audio
        result = process_audio(temp_audio_path)
        print(f"✅ Audio analysis complete: {len(result.get('segments', []))} segments found")
        return result
        
    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        return {"segments": []}
        
    finally:
        # Clean up temporary audio file
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                print(f"🧹 Cleaned up temporary audio file")
            except Exception as e:
                print(f"⚠️ Failed to clean up temporary audio file: {e}")

def analyze_audio_with_openai(audio_path):
    """
    Enhanced audio analysis using OpenAI via API server.
    This function works with both audio files and video files.
    """
    # Step 1: Process audio file to extract features, transcription, etc.
    audio_metadata = analyze_audio_comprehensive(audio_path)
        
    if not audio_metadata:
        raise ValueError("Audio processing failed or returned no data.")

    # Handle new return format (dict with segments and optional filename_info)
    if isinstance(audio_metadata, dict):
        segments = audio_metadata.get("segments", [])
        filename_info = audio_metadata.get("filename_info")
    else:
        # Backward compatibility with old format (list of segments)
        segments = audio_metadata
        filename_info = None
    
    if not segments:
        raise ValueError("Audio processing failed or returned no data.")

    # Step 2: Use API server for analysis
    try:
        from api_client import get_api_client
        client = get_api_client()
        
        # Call API server's audio analysis method directly
        response = client.openai_audio_analysis(segments, filename_info)
        print("✅ [API Server] Audio analysis complete")
        
        # Return the result from the API response
        if 'result' in response:
            return response['result']
        else:
            return response
        
    except Exception as e:
        print(f"❌ [API Server] Audio analysis failed: {e}")
        raise ValueError(f"API server audio analysis failed: {e}")

def save_audio_metadata(file_path, result, metadata_file="audio_metadata.json"):
    """Save audio analysis results to metadata file."""
    # Load existing metadata
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}
    else:
        metadata = {}
    
    # Use absolute path for consistency
    abs_path = os.path.abspath(file_path)
    
    # Save or update entry
    metadata[abs_path] = result
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Audio metadata saved for: {abs_path}")
    
    # Generate embeddings for search functionality
    try:
        from embedding_generator import generate_audio_embeddings
        generate_audio_embeddings(abs_path, result)
        print(f"✅ Generated embeddings for: {abs_path}")
    except Exception as e:
        print(f"⚠️ Failed to generate audio embeddings: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        audio_path = input("audio/video path: ")
    else:
        audio_path = sys.argv[1]
    
    # Test OpenAI analysis (optional)
    try:
        openai_result = analyze_audio_with_openai(audio_path)
        print("\n=== OpenAI Enhanced Analysis ===")
        print(json.dumps(openai_result, indent=2))
        save_audio_metadata(audio_path, openai_result)
    except Exception as e:
        print(f"⚠️ OpenAI audio analysis failed: {e}") 