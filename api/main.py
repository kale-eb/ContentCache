from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import openai
import os
import base64
import io
import json
import requests
import logging
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="ContentCache API Server",
    description="Centralized API server for video/image processing operations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this more restrictively in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging (must be before client initialization)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
try:
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.warning(f"OpenAI client initialization failed: {e}")
    openai_client = None

@app.on_event("startup")
async def startup_event():
    """Log when the application starts up."""
    logger.info("=== ContentCache API Server Starting ===")
    logger.info(f"OpenAI client: {'initialized' if openai_client else 'not available'}")
    logger.info(f"Environment variables loaded: {bool(os.getenv('OPENAI_API_KEY'))}")
    logger.info("=== Startup Complete ===")

@app.on_event("shutdown") 
async def shutdown_event():
    """Log when the application shuts down."""
    logger.info("=== ContentCache API Server Shutting Down ===")

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class OpenAIRequest(BaseModel):
    model: str = "gpt-4o-mini"
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

class VisionFrameAnalysisRequest(BaseModel):
    images_base64: List[str]
    num_frames: Optional[int] = None  # For dynamic prompt generation

class VideoSummaryRequest(BaseModel):
    frame_captions: List[str]
    transcript_segments: Any
    video_metadata: Dict[str, Any]
    vision_analysis: Optional[str] = None
    text_data: Optional[Dict[str, Any]] = None
    processed_location: Optional[str] = None

class MoondreamRequest(BaseModel):
    image_base64: str
    prompt: Optional[str] = "Describe this image in detail."

class GoogleMapsRequest(BaseModel):
    latitude: float
    longitude: float
    radius: Optional[int] = 500

class AudioTranscriptionRequest(BaseModel):
    audio_base64: str
    file_format: str = "mp3"

class AudioAnalysisRequest(BaseModel):
    segments: List[Dict[str, Any]]
    filename_info: Optional[Dict[str, Any]] = None

class ImageAnalysisRequest(BaseModel):
    image_base64: str
    ocr_text_data: Optional[Dict[str, Any]] = None
    processed_location: Optional[str] = None

class ImageSummaryRequest(BaseModel):
    caption: str
    objects: List[str]
    text_data: Optional[Dict[str, Any]] = None
    processed_location: Optional[str] = None

class TextAnalysisRequest(BaseModel):
    file_path: str
    text_content: str

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def decode_base64_to_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint that also serves as a backup health check."""
    logger.info("Root endpoint called")
    return {
        "message": "ContentCache API Server", 
        "status": "running",
        "version": "1.0.0",
        "health_check": "/health",
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """Simple health check that always returns 200 OK for Railway deployment."""
    try:
        logger.info("Health check endpoint called")
        status = {
            "status": "healthy",
            "message": "ContentCache API Server is running",
            "timestamp": "2025-01-15T00:00:00Z",
            "services": {
                "openai": bool(os.getenv("OPENAI_API_KEY")),
                "moondream": bool(os.getenv("MOONDREAM_API_KEY")),
                "google_maps": bool(os.getenv("GOOGLE_MAPS_API_KEY"))
            },
            "app_info": {
                "version": "1.0.0",
                "python_version": "3.10+",
                "environment": "production"
            }
        }
        logger.info(f"Health check returning: {status}")
        return status
    except Exception as e:
        logger.error(f"Health check error: {e}")
        # Even if there's an error, still return 200 OK for Railway
        return {
            "status": "healthy", 
            "message": "API server running despite health check error",
            "error": str(e)
        }

# ============================================================================
# OPENAI ENDPOINTS
# ============================================================================

@app.post("/api/openai/chat")
async def openai_chat(request: OpenAIRequest):
    """Generic OpenAI chat completions endpoint."""
    try:
        if not openai_client:
            raise HTTPException(status_code=503, detail="OpenAI client not available - check API key configuration")
        
        response = openai_client.chat.completions.create(
            model=request.model,
            messages=request.messages,
            tools=request.tools,
            tool_choice=request.tool_choice,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        return response.model_dump()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/openai/vision-frame-analysis")
async def openai_vision_frame_analysis(request: VisionFrameAnalysisRequest):
    """Analyze video frames using GPT-4o vision with built-in prompt."""
    try:
        num_frames = request.num_frames or len(request.images_base64)
        
        # Built-in prompt for frame analysis
        prompt = (
            f"Analyze these {num_frames} keyframes from a video and provide a CONCISE summary in this exact format:\n\n"
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
        
        # Prepare messages with images
        content = [{"type": "text", "text": prompt}]
        
        for img_base64 in request.images_base64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            })
        
        messages = [{"role": "user", "content": content}]
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500,
            temperature=0.2
        )
        
        return {
            "analysis": response.choices[0].message.content,
            "usage": response.usage.model_dump() if response.usage else None
        }
    except Exception as e:
        logger.error(f"OpenAI Vision Frame Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/openai/video-summary")
async def openai_video_summary(request: VideoSummaryRequest):
    """Generate comprehensive video summary using GPT-4o with built-in prompt and function calling."""
    try:
        # Build the content string with all available information
        content_parts = [
            f"Frame Captions (BLIP Analysis): {request.frame_captions}",
            f"Transcript Segments: {request.transcript_segments}",
            f"Full Metadata JSON: {json.dumps(request.video_metadata)}"
        ]
        
        # Add vision analysis if available
        if request.vision_analysis:
            content_parts.insert(1, f"Enhanced Vision Analysis: {request.vision_analysis}")
        
        # Add processed location if available
        if request.processed_location:
            content_parts.insert(-1, f"Pre-processed Location: {request.processed_location}")
        
        # Add text data if available
        if request.text_data and request.text_data.get('prominent_text'):
            prominent_text_str = ', '.join(request.text_data['prominent_text'])
            content_parts.insert(-1, f"Prominent Text from Frames: {prominent_text_str}")
            
            if request.text_data.get('frames_with_text', 0) > 0:
                content_parts.insert(-1, f"Text Detection Summary: Found text in {request.text_data['frames_with_text']}/{request.text_data['total_frames_processed']} frames")
        
        # Built-in system message and user instruction
        system_message = "You are a helpful assistant that analyzes videos"
        user_instruction = (
            "You will be given frame captions from BLIP analysis, transcript segments, and the full metadata JSON from ffprobe for a video file. "
            + ("You may also receive enhanced vision analysis from GPT-4o mini vision API. " if request.vision_analysis else "")
            + ("You may also receive prominent text extracted from video frames using OCR. " if request.text_data and request.text_data.get('prominent_text') else "")
            + ("Location information has been pre-processed from GPS coordinates if available. " if request.processed_location else "")
            + "Extract and return the following if present: any included description or comment, creation date, and modification date. "
            + "IMPORTANT: Convert any dates to UTC ISO format (YYYY-MM-DDTHH:MM:SS.000000Z) for consistent searching. "
            + "Then, using all the provided information (metadata, frame captions, transcript"
            + (", vision analysis" if request.vision_analysis else "")
            + (", extracted text" if request.text_data and request.text_data.get('prominent_text') else "")
            + (", and pre-processed location" if request.processed_location else "")
            + "), return a comprehensive summary, tags, and metadata. You MUST include a metadata key even if there is no metadata. "
            + "IMPORTANT: If metadata is provided (like location or description), give it extra weight in your analysis. "
            + ("If vision analysis is provided, use it to enhance and validate the BLIP frame captions. " if request.vision_analysis else "")
            + ("If text data is provided, use it to identify signs, labels, or other textual content in the video. " if request.text_data and request.text_data.get('prominent_text') else "")
        )
        
        # Function definition for structured output (moved from videotagger.py)
        function_def = {
            "name": "tag_video_metadata",
            "description": "Returns summary and keyword tags about a video based on frames, transcript, and existing metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_summary": {
                        "type": "string",
                        "description": "A 2 sentence summary of the video depicted"
                    },
                    "tags": {
                        "type": "object",
                        "properties": {
                            "mood": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Keywords describing the mood"
                            },
                            "locations": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Types of locations shown"
                            },
                            "context": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "What kind of content or story this is"
                            },
                            "objects": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Notable objects in the video"
                            },
                            "video_style": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Video format or camera style (e.g., b-roll, product video, selfie, talking head, action video, vlog)"
                            },
                            "actions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Actions depicted in the video (if any are described)"
                            },
                            "people": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "People in the video (such as friends, family, crowd, specific names mentioned in either the transcript or the metadata, etc.)"
                            }
                        },
                        "required": ["mood", "locations", "context", "objects", "video_style", "actions", "people"]
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "date_recorded": {
                                "type": "string",
                                "description": "Date when the video was recorded/created/added (if available) in UTC ISO format (YYYY-MM-DDTHH:MM:SS.000000Z), 'None' if not available"
                            },
                            "location": {
                                "type": "string",
                                "description": "Location where the video was recorded (pre-processed from metadata), 'None' if not available"
                            },
                            "included_description": {
                                "type": "string",
                                "description": "Description provided in the video metadata (if available), 'None' if not"
                            }
                        }
                    }
                },
                "required": ["video_summary", "tags", "metadata"]
            }
        }
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_instruction},
                {"role": "user", "content": "\n\n".join(content_parts)}
            ],
            tools=[
                {"type": "function", "function": function_def}
            ],
            tool_choice="auto"
        )
        
        # Extract the structured JSON output from API response
        if response.choices[0].message.tool_calls:
            return {
                "result": response.choices[0].message.tool_calls[0].function.arguments,
                "usage": response.usage.model_dump() if response.usage else None
            }
        else:
            # Fallback if no tool call was made
            return {
                "result": response.choices[0].message.content,
                "usage": response.usage.model_dump() if response.usage else None
            }
            
    except Exception as e:
        logger.error(f"OpenAI Video Summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/openai/transcribe")
async def openai_transcribe_audio(request: AudioTranscriptionRequest):
    """Transcribe audio using OpenAI Whisper."""
    try:
        # Decode base64 audio
        audio_data = base64.b64decode(request.audio_base64)
        
        # Create a temporary file-like object
        audio_file = io.BytesIO(audio_data)
        audio_file.name = f"audio.{request.file_format}"
        
        # Transcribe with Whisper
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        
        return {"transcript": transcript.text}
    except Exception as e:
        logger.error(f"OpenAI Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/openai/audio-analysis")
async def openai_audio_analysis(request: AudioAnalysisRequest):
    """Analyze audio segments using GPT-4o with built-in prompt and function calling."""
    try:
        # Built-in prompt for audio analysis
        base_prompt = (
            "Given the following audio metadata, including transcription, features, and tags, "
            "provide a concise summary of the audio and a list of keywords (including mood, topics, genre, and any other relevant descriptors).\n\n"
            f"Audio Segments Data (JSON):\n{json.dumps(request.segments, indent=2)}\n"
        )
        
        # Add filename context for audio files
        if request.filename_info and request.filename_info.get("is_audio_file"):
            title = request.filename_info.get("title", "")
            base_prompt += (
                f"\nAdditional Context:\n"
                f"This is an audio file with the filename/title: '{title}'\n"
                f"If the filename provides meaningful context about the audio content (e.g., describes the type of sound, "
                f"mood, or purpose), please incorporate that information into your analysis. Consider how the filename "
                f"might indicate the intended use, genre, or characteristics of the audio.\n"
            )
        
        # Built-in function schema
        function_def = {
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
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": base_prompt}
            ],
            tools=[{"type": "function", "function": function_def}],
            tool_choice={"type": "function", "function": {"name": "summarize_audio"}},
            max_tokens=400
        )
        
        # Extract the structured JSON output from API response
        if response.choices[0].message.tool_calls:
            function_args = response.choices[0].message.tool_calls[0].function.arguments
            return {
                "result": function_args,
                "usage": response.usage.model_dump() if response.usage else None
            }
        else:
            raise HTTPException(status_code=500, detail="No tool call found in OpenAI response")
            
    except Exception as e:
        logger.error(f"OpenAI Audio Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/openai/image-analysis")
async def openai_image_analysis(request: ImageAnalysisRequest):
    """Analyze single image using GPT-4o vision with built-in prompt."""
    try:
        # Built-in analysis prompt
        analysis_prompt = (
            "Analyze this image efficiently. Skip categories that don't add value.\n\n"
            "Format (skip irrelevant sections):\n"
            "1. Summary: [1-3 sentence(s) describing the main content and scene]\n"
            "2. Objects: [key items, people] (keywords only)\n"
            "3. Text: [meaningful visible text ONLY - skip if garbled/unclear]\n"
            "4. Setting: [location type, environment] (keywords only)\n"
            "5. Activities: [main actions] (keywords only)\n"
            "6. Mood: [atmosphere keywords] (keywords only)\n"
            "7. Style: [visual characteristics] (keywords only)\n"
            "8. Details: [notable elements] (keywords only)\n\n"
        )
        
        # Add OCR text information if available and meaningful
        if request.ocr_text_data and request.ocr_text_data.get('prominent_text'):
            prominent_text = ', '.join(request.ocr_text_data['prominent_text'])
            # Only include OCR if it contains meaningful words
            meaningful_words = [word for word in prominent_text.split() if len(word) > 2 and word.isalpha()]
            if len(meaningful_words) > 0:
                analysis_prompt += (
                    f"IMPORTANT: OCR detected meaningful text: {prominent_text}\n"
                    f"Include this in section 3 if clearly readable. If the OCR text appears garbled or meaningless, skip section 3 entirely.\n\n"
                )
        
        # Add location information if available
        if request.processed_location:
            analysis_prompt += (
                f"LOCATION: This image was taken at: {request.processed_location}\n"
                f"Please incorporate this location information into your analysis, especially for section 4 (setting/location).\n\n"
            )
        
        analysis_prompt += "Be thorough and specific in your analysis."
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an efficient image analyst focused on extracting meaningful keywords. "
                        "Provide concise, keyword-based analysis that maximizes information density while minimizing token usage. "
                        "Skip verbose descriptions and garbled text. Use comma-separated keywords for categories. "
                        "Only include categories that add genuine value to understanding the image content."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": analysis_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{request.image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=800
        )
        
        return {
            "analysis": response.choices[0].message.content,
            "usage": response.usage.model_dump() if response.usage else None
        }
        
    except Exception as e:
        logger.error(f"OpenAI Image Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/openai/image-summary")
async def openai_image_summary(request: ImageSummaryRequest):
    """Generate image summary using GPT-4o with built-in prompt and function calling."""
    try:
        # Built-in prompt for image summarization
        prompt = (
            "Given the following image caption and detected objects, "
            "provide a concise summary of the image and a list of objects in the image.\n\n"
            f"Caption: {request.caption}\nObjects: {request.objects}\n"
        )
        
        # Add OCR text information if available
        if request.text_data and request.text_data.get('prominent_text'):
            prominent_text = ', '.join(request.text_data['prominent_text'])
            prompt += f"Detected Text (OCR): {prominent_text}\n"
            prompt += "Please incorporate any relevant text information into your summary.\n\n"
        
        # Add location information if available
        if request.processed_location and request.processed_location != 'None':
            prompt += f"Location: {request.processed_location}\n"
            prompt += "Please incorporate this location information into your summary where relevant.\n\n"
        
        # Built-in function schema
        function_def = {
            "name": "summarize_image",
            "description": "Summarize image and extract objects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A concise summary of the image."
                    },
                    "objects": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of objects detected in the image."
                    }
                },
                "required": ["summary", "objects"]
            }
        }
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            tools=[{"type": "function", "function": function_def}],
            tool_choice={"type": "function", "function": {"name": "summarize_image"}},
            max_tokens=400
        )
        
        # Extract the structured JSON output from API response
        if response.choices[0].message.tool_calls:
            function_args = response.choices[0].message.tool_calls[0].function.arguments
            return {
                "result": function_args,
                "usage": response.usage.model_dump() if response.usage else None
            }
        else:
            raise HTTPException(status_code=500, detail="No tool call found in OpenAI response")
            
    except Exception as e:
        logger.error(f"OpenAI Image Summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/openai/text-analysis")
async def openai_text_analysis(request: TextAnalysisRequest):
    """Analyze text documents using GPT-4o with built-in prompt and function calling."""
    try:
        import os
        from pathlib import Path
        
        filename = os.path.basename(request.file_path)
        file_ext = Path(request.file_path).suffix.lower()
        
        # Built-in prompt for text analysis
        prompt = f"""Analyze this text document and provide the following specific information:

FILENAME: {filename}
FILE TYPE: {file_ext}
CONTENT LENGTH: {len(request.text_content)} characters

DOCUMENT CONTENT:
{request.text_content}

Please provide exactly these 4 fields:
1. summary - Brief summary of file content (1-2 sentences)
2. key_topics - Array of keywords representing main topics
3. tone - Writing tone (e.g., professional, casual, formal, academic, creative)
4. language - Primary language used (e.g., English, Spanish, etc.)

Format your response as clean JSON with exactly these field names."""

        # Built-in function schema
        function_def = {
            "name": "analyze_text",
            "description": "Analyze a text document and provide structured metadata",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of file content (1-2 sentences)"
                    },
                    "key_topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of keywords representing main topics"
                    },
                    "tone": {
                        "type": "string",
                        "description": "Writing tone (e.g., professional, casual, formal, academic, creative)"
                    },
                    "language": {
                        "type": "string",
                        "description": "Primary language used (e.g., English, Spanish, etc.)"
                    },
                },
                "required": ["summary", "key_topics", "tone", "language"]
            }
        }

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            tools=[{"type": "function", "function": function_def}],
            tool_choice={"type": "function", "function": {"name": "analyze_text"}},
            max_tokens=1500
        )

        # Extract the structured JSON output from API response
        if response.choices[0].message.tool_calls:
            function_args = response.choices[0].message.tool_calls[0].function.arguments
            return {
                "result": function_args,
                "usage": response.usage.model_dump() if response.usage else None
            }
        else:
            raise HTTPException(status_code=500, detail="No tool call found in OpenAI response")
            
    except Exception as e:
        logger.error(f"OpenAI Text Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MOONDREAM ENDPOINTS
# ============================================================================

@app.post("/api/moondream/caption")
async def moondream_caption(request: MoondreamRequest):
    """Generate caption using Moondream API."""
    try:
        moondream_api_key = os.getenv("MOONDREAM_API_KEY")
        if not moondream_api_key:
            raise HTTPException(status_code=500, detail="MOONDREAM_API_KEY not configured")
        
        # Initialize Moondream SDK
        import moondream as md
        model = md.vl(api_key=moondream_api_key)
        
        # Decode base64 image
        image = decode_base64_to_image(request.image_base64)
        
        # Generate caption
        result = model.caption(image, length="normal")
        
        return {"caption": result.get("caption", "")}
    except ImportError:
        raise HTTPException(status_code=500, detail="Moondream SDK not installed")
    except Exception as e:
        logger.error(f"Moondream API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/moondream/batch")
async def moondream_batch_process(request: Dict[str, List[str]]):
    """Process multiple images with Moondream API."""
    try:
        moondream_api_key = os.getenv("MOONDREAM_API_KEY")
        if not moondream_api_key:
            raise HTTPException(status_code=500, detail="MOONDREAM_API_KEY not configured")
        
        import moondream as md
        model = md.vl(api_key=moondream_api_key)
        
        images_base64 = request.get("images_base64", [])
        results = []
        
        for i, img_base64 in enumerate(images_base64):
            try:
                image = decode_base64_to_image(img_base64)
                result = model.caption(image, length="normal")
                results.append({
                    "index": i,
                    "caption": result.get("caption", ""),
                    "success": True
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "error": str(e),
                    "success": False
                })
        
        return {"results": results}
    except ImportError:
        raise HTTPException(status_code=500, detail="Moondream SDK not installed")
    except Exception as e:
        logger.error(f"Moondream batch processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# GOOGLE MAPS ENDPOINTS
# ============================================================================

@app.post("/api/google/reverse-geocode")
async def google_reverse_geocode(request: GoogleMapsRequest):
    """Reverse geocode coordinates using Google Maps API."""
    try:
        api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GOOGLE_MAPS_API_KEY not configured")
        
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            "latlng": f"{request.latitude},{request.longitude}",
            "key": api_key
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        logger.error(f"Google reverse geocoding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/google/nearby-search")
async def google_nearby_search(request: GoogleMapsRequest):
    """Search for nearby places using Google Maps API."""
    try:
        api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GOOGLE_MAPS_API_KEY not configured")
        
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{request.latitude},{request.longitude}",
            "radius": request.radius,
            "key": api_key
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        logger.error(f"Google nearby search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# UPLOAD ENDPOINTS
# ============================================================================

@app.post("/api/upload/process-image")
async def process_uploaded_image(file: UploadFile = File(...)):
    """Process an uploaded image file."""
    try:
        # Read file content
        content = await file.read()
        
        # Encode to base64
        encoded_image = base64.b64encode(content).decode('utf-8')
        
        return {
            "filename": file.filename,
            "size": len(content),
            "base64": encoded_image
        }
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 