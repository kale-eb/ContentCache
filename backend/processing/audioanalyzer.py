import os
import json
import subprocess
import tempfile
from audioprocessor import process_audio
import requests

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
            "ffmpeg", 
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
    Enhanced audio analysis using OpenAI for summarization.
    This function works with both audio files and video files.
    """
    # Step 1: Process audio file to extract features, transcription, etc.
    if os.path.splitext(audio_path)[1].lower() in {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}:
        # It's a video file, use comprehensive analysis
        audio_metadata = analyze_audio_comprehensive(audio_path)
    else:
        # It's an audio file, use direct processing
    audio_metadata = process_audio(audio_path)
        
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

    # Step 2: Prepare prompt for OpenAI
    base_prompt = (
        "Given the following audio metadata, including transcription, features, and tags, "
        "provide a concise summary of the audio and a list of keywords (including mood, topics, genre, and any other relevant descriptors).\n\n"
        f"Audio Segments Data (JSON):\n{json.dumps(segments, indent=2)}\n"
    )
    
    # Add filename context for audio files
    if filename_info and filename_info.get("is_audio_file"):
        title = filename_info.get("title", "")
        base_prompt += (
            f"\nAdditional Context:\n"
            f"This is an audio file with the filename/title: '{title}'\n"
            f"If the filename provides meaningful context about the audio content (e.g., describes the type of sound, "
            f"mood, or purpose), please incorporate that information into your analysis. Consider how the filename "
            f"might indicate the intended use, genre, or characteristics of the audio.\n"
        )
    
    prompt = base_prompt

    # Step 3: Define OpenAI function schema
    openai_function = {
        "type": "function",
        "function": {
            "name": "summarize_audio",
            "description": "Summarize audio and extract keywords (mood, topics, genre, etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A concise summary of the audio."
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of keywords describing mood, topics, genre, and other relevant descriptors."
                    }
                },
                "required": ["summary", "keywords"]
            }
        }
    }

    # Step 4: Call OpenAI API
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "tools": [openai_function],
        "tool_choice": {"type": "function", "function": {"name": "summarize_audio"}},
        "max_tokens": 400
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data
    )
    response.raise_for_status()
    tool_calls = response.json()["choices"][0]["message"].get("tool_calls", [])
    if tool_calls:
        arguments = tool_calls[0]["function"]["arguments"]
        import json as pyjson
        args = pyjson.loads(arguments)
        return args
    else:
        raise ValueError("No tool call found in OpenAI response")

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
    # Save or update entry
    metadata[file_path] = result
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        audio_path = input("audio/video path: ")
    else:
        audio_path = sys.argv[1]
    
    # Test comprehensive analysis
    result = analyze_audio_comprehensive(audio_path)
    print("\n=== Comprehensive Audio Analysis ===")
    print(json.dumps(result, indent=2)) 
    
    # Test OpenAI analysis (optional)
    try:
        openai_result = analyze_audio_with_openai(audio_path)
        print("\n=== OpenAI Enhanced Analysis ===")
        print(json.dumps(openai_result, indent=2))
        save_audio_metadata(audio_path, openai_result)
    except Exception as e:
        print(f"⚠️ OpenAI analysis failed: {e}") 