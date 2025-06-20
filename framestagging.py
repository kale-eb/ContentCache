import gc
from pathlib import Path
from PIL import Image
# BLIP imports commented out - using Moondream API only
# from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, BlipImageProcessor, AutoModelForCausalLM

from sentence_transformers import SentenceTransformer, util
import os
import torch
import time
import warnings
import concurrent.futures
import threading
import cv2
import base64
import io
import math
import json
from dotenv import load_dotenv
import openai
from config import get_models_cache_dir

# Load environment variables
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Suppress various warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pin_memory.*")

# Set environment variables to suppress tokenizer warnings and avoid device issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Disable MPS entirely for Moondream2 to avoid device mismatch issues on Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Constants
MODEL_CACHE_DIR = get_models_cache_dir()

# Enable offline mode to prevent network requests and rate limiting
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Optimize torch settings for faster loading
torch.set_num_threads(4)  # Limit CPU threads for faster startup

# BLIP functionality commented out - using Moondream API only
print("⚠️ BLIP functionality disabled - using Moondream API only")

# Initialize placeholder variables for BLIP (commented out)
processor = None
model = None
sentence_model = None
image_processor = None
tokenizer = None

# Previously loaded BLIP models here - now using Moondream API only

# Global variables for models (lazy loading)
moondream_model = None
moondream_device = None

def load_moondream_api():
    """Load Moondream API configuration."""
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('MOONDREAM_API_KEY')
    if not api_key:
        print("❌ MOONDREAM_API_KEY not found in .env file")
        return None
    
    # Clean the API key (remove any trailing whitespace/characters)
    api_key = api_key.strip()
    
    print(f"✅ Moondream API key loaded successfully (length: {len(api_key)})")
    return api_key

# Local model loading removed - using Moondream API only



labels_dict = {}

def encode_image_to_base64(image_path, resize=True, max_dimension=512, quality=50):
    """
    Encode image to base64 for OpenAI API with compression.
    """
    try:
        # Open and process the image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Store original dimensions for logging
            original_dimensions = img.size
            
            # Resize based on the larger dimension
            if resize:
                width, height = img.size
                if width > height:
                    new_width = max_dimension
                    new_height = int((height * max_dimension) / width)
                else:
                    new_height = max_dimension
                    new_width = int((width * max_dimension) / height)
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                resize_info = f" resized from {original_dimensions} to {img.size}"
            else:
                resize_info = f" kept at {img.size}"
            
            # Compress to JPEG in memory
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

def calculate_keyframes_to_send(total_keyframes):
    """Calculate how many keyframes to send to OpenAI based on total count."""
    if total_keyframes <= 1:
        return 1
    elif total_keyframes <= 3:
        return total_keyframes
    elif total_keyframes <= 10:
        return max(3, int(total_keyframes * 0.7))
    else:
        calculated = min(10, max(5, int(3 + 2 * math.log10(total_keyframes))))
        return calculated

def select_representative_keyframes(frames_dir, num_to_select):
    """Select representative keyframes evenly distributed across the timeline."""
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    if not frame_files:
        return []
    
    if len(frame_files) <= num_to_select:
        return [os.path.join(frames_dir, f) for f in frame_files]
    
    # Select frames evenly distributed
    selected_indices = []
    step = len(frame_files) / num_to_select
    
    for i in range(num_to_select):
        index = int(i * step)
        index = min(index, len(frame_files) - 1)
        selected_indices.append(index)
    
    # Remove duplicates and sort
    selected_indices = sorted(list(set(selected_indices)))
    
    # Add more if needed
    while len(selected_indices) < num_to_select and len(selected_indices) < len(frame_files):
        for i in range(len(frame_files)):
            if i not in selected_indices:
                selected_indices.append(i)
                if len(selected_indices) >= num_to_select:
                    break
    
    selected_frames = [os.path.join(frames_dir, frame_files[i]) for i in selected_indices[:num_to_select]]
    return selected_frames

def analyze_keyframes_with_gpt4o_vision(selected_frames):
    """Analyze selected keyframes using GPT-4o mini vision API."""
    if not selected_frames:
        print("⚠️ No keyframes available for GPT-4o vision analysis")
        return None
    
    print(f"🔍 Analyzing {len(selected_frames)} keyframes with GPT-4o mini vision...")
    
    # Prepare the analysis prompt
    analysis_prompt = (
        f"Analyze these {len(selected_frames)} keyframes from a video and provide a CONCISE summary in this exact format:\n\n"
        "1. Text Detection: [any visible text, signs, or written content - be brief]\n"
        "2. People & Expressions: [facial expressions, body language, interactions - one sentence]\n"
        "3. Objects & Details: [key objects, brands, tools with brief descriptions]\n"
        "4. Environment & Setting: [location, lighting, atmosphere - one sentence]\n"
        "5. Temporal Changes: [how scene evolves across frames - brief]\n"
        "6. Visual Style: [camera style, production quality - brief]\n"
        "7. Context Clues: [purpose, activity, or story elements - brief]\n\n"
        "Keep each point to 1-2 sentences maximum. Provide ONE overall summary, NOT per-frame analysis. "
        "If nothing significant is detected for a category, write 'none detected' or 'minimal'."
    )
    
    # Prepare messages for API call
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": analysis_prompt}
            ]
        }
    ]
    
    # Add each frame to the message
    for i, frame_path in enumerate(selected_frames):
        encoded_image = encode_image_to_base64(frame_path)
        if encoded_image:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
            })
            print(f"  📸 Added frame {i+1}/{len(selected_frames)}: {os.path.basename(frame_path)}")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500,
            temperature=0.2
        )
        
        result = response.choices[0].message.content.strip()
        print("✅ GPT-4o vision analysis complete")
        return result
        
    except Exception as e:
        print(f"❌ Error in GPT-4o vision analysis: {e}")
        return None

# Legacy functions removed - using Moondream API only



def process_frames_with_moondream_api(frames_dir, vid_path):
    """
    Process frames using Moondream API (fast cloud inference).
    
    Args:
        frames_dir (str): Directory containing extracted frames
        vid_path (str): Path to video file
    
    Returns:
        list: Frame captions with metadata
    """
    print("🚀 Processing frames with Moondream API (fast cloud inference)...")
    
    # Load API key
    api_key = load_moondream_api()
    if not api_key:
        print("❌ Moondream API not available")
        return []
    
    # Initialize Moondream model with API key
    try:
        import moondream as md
        from PIL import Image
        model = md.vl(api_key=api_key)
        print("✅ Moondream SDK initialized successfully")
    except ImportError:
        print("❌ Moondream SDK not installed. Install with: pip install moondream")
        return []
    except Exception as e:
        print(f"❌ Error initializing Moondream SDK: {e}")
        return []
    
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
    frame_files.sort()
    
    # Get video info for timestamp calculation (same as other functions)
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
    cap.release()
    
    frame_captions = []
    
    print(f"📸 Processing {len(frame_files)} frames with Moondream API...")
    
    for frame_index, frame_file in enumerate(frame_files):
        try:
            frame_path = os.path.join(frames_dir, frame_file)
            timestamp = frame_index / fps if fps > 0 else frame_index
            
            # Load image with PIL
            image = Image.open(frame_path)
            
            # Generate short caption using Moondream SDK
            start_time = time.time()
            result = model.caption(image, length="short")
            inference_time = time.time() - start_time
            
            moondream_caption = result["caption"]
            
            print(f"  📸 Frame {frame_index + 1}: {inference_time:.2f}s - {moondream_caption[:50]}...")
            
            # Calculate basic quality score
            quality_score = len(moondream_caption.split()) - 2  # Simple heuristic
            
            frame_captions.append({
                'frame_index': frame_index,
                'timestamp': timestamp,
                'frame_path': frame_path,
                'caption': moondream_caption,
                'quality_score': quality_score
            })
            
            # Optional: Print progress for longer videos
            if (frame_index + 1) % 5 == 0:
                print(f"  📸 Processed {frame_index + 1}/{len(frame_files)} frames")
    
        except Exception as e:
                print(f"❌ Error processing frame {frame_file} with Moondream API: {e}")
                # Skip this frame or fall back to a default caption
                continue
            
        print(f"✓ Moondream API processed {len(frame_captions)} frames")
        return frame_captions


def deduplicate_captions(captions, frame_paths, threshold=0.85):
    # Use the global model instance instead of creating a new one
    global sentence_model
    
    # Process captions in chunks to reduce memory usage
    chunk_size = 20
    kept = []
    seen = set()
    
    for i in range(0, len(captions), chunk_size):
        chunk_captions = captions[i:i + chunk_size]
        chunk_paths = frame_paths[i:i + chunk_size]
        
        # Generate embeddings for this chunk
        chunk_embeddings = sentence_model.encode(chunk_captions, convert_to_tensor=True)
        
        # Compare with previous chunks
        for j in range(len(chunk_captions)):
            if i + j in seen:
                continue
                
            kept.append((chunk_captions[j], chunk_paths[j]))
            
            # Compare with remaining captions in this chunk
            for k in range(j + 1, len(chunk_captions)):
                if i + k in seen:
                    continue
                sim = util.pytorch_cos_sim(chunk_embeddings[j], chunk_embeddings[k]).item()
                if sim > threshold:
                    seen.add(i + k)
        
        # Clean up chunk data
        del chunk_embeddings
        gc.collect()
        
        # Clear PyTorch cache if CUDA is available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    
    return kept

# Legacy functions removed - current implementation uses:
# 1. process_frames_with_moondream_api() for frame captioning  
# 2. analyze_keyframes_with_gpt4o_vision() for vision analysis
