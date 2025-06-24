from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import openai
import os
import sys
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
openai_client = None
try:
    # Log OpenAI version for debugging
    logger.info(f"OpenAI library version: {openai.__version__}")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        logger.info("OpenAI API key found, attempting client initialization...")
        # Try different initialization methods for version compatibility
        try:
            openai_client = openai.OpenAI(api_key=api_key)
            logger.info("✅ OpenAI client initialized successfully")
        except Exception as init_error:
            logger.warning(f"OpenAI initialization failed: {init_error}")
            openai_client = None
    else:
        logger.warning("⚠️ No OpenAI API key provided - OpenAI features will be disabled")
        openai_client = None
except Exception as e:
    logger.error(f"❌ OpenAI client initialization failed: {e}")
    openai_client = None

@app.on_event("startup")
async def startup_event():
    """Log when the application starts up."""
    try:
        logger.info("=== ContentCache API Server Starting ===")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"FastAPI available: {bool(app)}")
        logger.info(f"OpenAI client: {'initialized' if openai_client else 'not available'}")
        logger.info(f"Environment variables:")
        logger.info(f"  - OPENAI_API_KEY: {'present' if os.getenv('OPENAI_API_KEY') else 'missing'}")
        logger.info(f"  - MOONDREAM_API_KEY: {'present' if os.getenv('MOONDREAM_API_KEY') else 'missing'}")
        logger.info(f"  - GOOGLE_MAPS_API_KEY: {'present' if os.getenv('GOOGLE_MAPS_API_KEY') else 'missing'}")
        logger.info(f"  - PORT: {os.getenv('PORT', '8000')}")
        logger.info("=== Startup Complete ===")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        # Don't raise the error to prevent startup failure

@app.on_event("shutdown") 
async def shutdown_event():
    """Log when the application shuts down."""
    try:
        logger.info("=== ContentCache API Server Shutting Down ===")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

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
    audio_summary: Any
    video_metadata: Dict[str, Any]
    vision_analysis: Optional[str] = None
    text_data: Optional[Dict[str, Any]] = None
    processed_location: Optional[Union[str, Dict[str, Any]]] = None

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
    processed_location: Optional[Union[str, Dict[str, Any]]] = None

class ImageSummaryRequest(BaseModel):
    caption: str
    objects: List[str]
    filename: str
    coordinates: Optional[Dict[str, float]] = None  # {latitude, longitude}
    included_description: Optional[str] = None

class TextAnalysisRequest(BaseModel):
    file_path: str
    text_content: str

class GoogleForwardGeocodeRequest(BaseModel):
    location_text: str

class SearchQueryParseRequest(BaseModel):
    query: str

class MoondreamAnalysisRequest(BaseModel):
    image_base64: str

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
    return {"status": "healthy", "message": "API server running"}

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
        if not openai_client:
            logger.error("OpenAI client not available for vision analysis")
            raise HTTPException(
                status_code=503, 
                detail="OpenAI client not available - check API key configuration"
            )
        
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
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"OpenAI Vision Frame Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/openai/video-summary")
async def openai_video_summary(request: VideoSummaryRequest):
    """Generate comprehensive video summary using GPT-4o with built-in prompt and function calling."""
    try:
        if not openai_client:
            logger.error("OpenAI client not available for video summary")
            raise HTTPException(
                status_code=503, 
                detail="OpenAI client not available - check API key configuration"
            )
        
        # Build the content string with all available information
        content_parts = [
            f"Frame Captions: {request.frame_captions}",
            f"Audio Summary: {request.audio_summary}",
            f"Essential Metadata: {json.dumps(request.video_metadata)}"
        ]
        
        # Add vision analysis if available
        if request.vision_analysis:
            content_parts.insert(1, f"Enhanced Vision Analysis: {request.vision_analysis}")
        
        # Add processed location if available (handle both old string format and new dict format)
        if request.processed_location:
            if isinstance(request.processed_location, dict):
                if request.processed_location.get('type') == 'coordinates':
                    lat, lon = request.processed_location['latitude'], request.processed_location['longitude']
                    content_parts.insert(-1, f"GPS Coordinates: {lat}, {lon}")
                elif request.processed_location.get('type') == 'text':
                    content_parts.insert(-1, f"Location Text: {request.processed_location['location_text']}")
            else:
                # Legacy string format
                content_parts.insert(-1, f"Pre-processed Location: {request.processed_location}")
        
        # Add text data if available
        if request.text_data and request.text_data.get('prominent_text'):
            prominent_text_str = ', '.join(request.text_data['prominent_text'])
            content_parts.insert(-1, f"Prominent Text from Frames: {prominent_text_str}")
            
            if request.text_data.get('frames_with_text', 0) > 0:
                content_parts.insert(-1, f"Text Detection Summary: Found text in {request.text_data['frames_with_text']}/{request.text_data['total_frames_processed']} frames")
        
        # Built-in system message and user instruction
        system_message = "Analyze videos using provided data, which will include frame captions, audio summary, and metadata. May also include enhanced vision analysis OCR, and location coordinates."
        
        # Build dynamic instruction based on available data
        data_types = ["frame captions", "audio summary", "metadata"]
        if request.vision_analysis:
            data_types.append("vision analysis")
        if request.text_data and request.text_data.get('prominent_text'):
            data_types.append("OCR text")
        if request.processed_location:
            data_types.append("location")
            
        user_instruction = f"Analyze using: {', '.join(data_types)}. Convert dates to UTC ISO format. Return summary, tags, and metadata (required even if empty)."
        
        # Function definition for structured output (moved from videotagger.py)
        function_def = {
            "name": "tag_video_metadata",
            "description": "Extract video summary and tags",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_summary": {
                        "type": "string",
                        "description": "2 sentence summary"
                    },
                    "tags": {
                        "type": "object",
                        "properties": {
                            "mood": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Mood keywords"
                            },
                            "locations": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Locations shown (if applicable)"
                            },
                            "context": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Content type"
                            },
                            "objects": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Key objects (if applicable)"
                            },
                            "video_style": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Camera/format style"
                            },
                            "actions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Actions shown (if applicable)"
                            },
                            "people": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "People mentioned (if applicable)"
                            }
                        },
                        "required": ["mood", "locations", "context", "objects", "video_style", "actions", "people"]
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "date_recorded": {
                                "type": "string",
                                "description": "Recording date in UTC ISO or 'None'"
                            },
                            "location": {
                                "type": "string",
                                "description": "Coordinates or 'None'"
                            },
                            "included_description": {
                                "type": "string",
                                "description": "Metadata description or 'None'"
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
        logger.info(f"Starting audio analysis with {len(request.segments)} segments")
        
        # Validate input data
        if not request.segments:
            raise HTTPException(status_code=400, detail="No audio segments provided")
        
        # Built-in prompt for audio analysis
        base_prompt = f"Analyze audio data and provide summary + keywords.\n\nData: {json.dumps(request.segments, indent=2)}\n"
        
        # Add filename context for audio files
        if request.filename_info and request.filename_info.get("is_audio_file"):
            title = request.filename_info.get("title", "")
            base_prompt += f"\nFilename: '{title}' (use for context if relevant)\n"
            logger.info(f"Processing audio file: {title}")
        
        # Check if OpenAI client is available
        if not openai_client:
            logger.error("OpenAI client not initialized")
            raise HTTPException(status_code=503, detail="OpenAI client not available")
        
        # Built-in function schema
        function_def = {
            "name": "summarize_audio",
            "description": "Audio summary and keywords",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Concise summary"
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Mood, topics, genre keywords"
                    }
                },
                "required": ["summary", "keywords"]
            }
        }
        
        logger.info("Making OpenAI API call for audio analysis")
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": base_prompt}
            ],
            tools=[{"type": "function", "function": function_def}],
            tool_choice={"type": "function", "function": {"name": "summarize_audio"}},
            max_tokens=400
        )
        
        logger.info("OpenAI API call completed successfully")
        
        # Extract the structured JSON output from API response
        if response.choices[0].message.tool_calls:
            function_args = response.choices[0].message.tool_calls[0].function.arguments
            # Parse the JSON string to ensure it's valid JSON
            try:
                parsed_result = json.loads(function_args)
                return {
                    "result": parsed_result,
                    "usage": response.usage.model_dump() if response.usage else None
                }
            except json.JSONDecodeError as json_error:
                logger.error(f"Failed to parse OpenAI function response as JSON: {json_error}")
                logger.error(f"Raw function_args: {function_args}")
                # Return the raw string if JSON parsing fails
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
                analysis_prompt += f"OCR text: {prominent_text}\n"
        
        # Add location information if available
        if request.processed_location:
            if isinstance(request.processed_location, dict):
                if request.processed_location.get('type') == 'coordinates':
                    lat, lon = request.processed_location['latitude'], request.processed_location['longitude']
                    analysis_prompt += f"GPS: {lat}, {lon}\n"
                elif request.processed_location.get('type') == 'text':
                    analysis_prompt += f"Location: {request.processed_location['location_text']}\n"
            else:
                # Legacy string format
                analysis_prompt += f"Location: {request.processed_location}\n"
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Efficient image analysis. Use keywords, skip irrelevant sections."
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
    """Generate comprehensive image summary using GPT-4o with built-in prompt and function calling."""
    try:
        # Build the content string with all available information
        content_parts = [
            f"Image Caption: {request.caption}",
            f"Detected Objects: {', '.join(request.objects)}",
            f"Filename: {request.filename}"
        ]
        
        # Add coordinates if available
        if request.coordinates:
            lat, lon = request.coordinates.get('latitude'), request.coordinates.get('longitude')
            if lat is not None and lon is not None:
                content_parts.append(f"GPS Coordinates: {lat}, {lon}")
        
        # Add included description if available
        if request.included_description:
            content_parts.append(f"Included Description: {request.included_description}")
        
        # Built-in system message and user instruction
        system_message = "Analyze images for metadata"
        
        data_types = ["caption", "objects", "filename"]
        if request.coordinates:
            data_types.append("GPS")
        if request.included_description:
            data_types.append("description")
            
        user_instruction = f"Extract from: {', '.join(data_types)}. Convert dates to UTC ISO. Return summary, tags, metadata."
        
        # Function definition for structured output
        function_def = {
            "name": "tag_image_metadata",
            "description": "Extract image summary and tags",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_summary": {
                        "type": "string",
                        "description": "2 sentence summary"
                    },
                    "tags": {
                        "type": "object",
                        "properties": {
                            "mood": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Mood keywords"
                            },
                            "locations": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Location types"
                            },
                            "context": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Scene type"
                    },
                    "objects": {
                        "type": "array",
                        "items": {"type": "string"},
                                "description": "Key objects"
                            },
                            "style": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Visual style"
                            },
                            "activities": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Activities shown"
                            },
                            "people": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "People mentioned"
                            }
                        },
                        "required": ["mood", "locations", "context", "objects", "style", "activities", "people"]
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "date_taken": {
                                "type": "string",
                                "description": "Date in UTC ISO or 'None'"
                            },
                            "location": {
                                "type": "string",
                                "description": "Location or 'None'"
                            },
                            "included_description": {
                                "type": "string",
                                "description": "Metadata description or 'None'"
                            }
                        }
                    }
                },
                "required": ["image_summary", "tags", "metadata"]
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
        prompt = f"""Analyze: {filename} ({file_ext}, {len(request.text_content)} chars)

Content:
{request.text_content}

Return: summary (1-2 sentences), key_topics (keywords), tone, language."""

        # Built-in function schema
        function_def = {
            "name": "analyze_text",
            "description": "Text analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "1-2 sentence summary"
                    },
                    "key_topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Topic keywords"
                    },
                    "tone": {
                        "type": "string",
                        "description": "Writing tone"
                    },
                    "language": {
                        "type": "string",
                        "description": "Primary language"
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

@app.post("/api/openai/parse-search-query")
async def parse_search_query(request: SearchQueryParseRequest):
    """Parse search query to extract semantic components using OpenAI."""
    try:
        # Get current date for date calculations
        from datetime import datetime, timezone
        current_date = datetime.now(timezone.utc).isoformat()
        
        # Built-in prompt for search query parsing
        prompt = f"""Parse: "{request.query}" (Current: {current_date})

Extract:
- search_query: core terms (no location/date)
- location: specific place or null
- search_radius: km based on specificity (determine appropriate radius based on location. more specific locations should have smaller radius) 
- date: UTC range object or null

Date examples: "last month" → {{"start": "2025-05-20T...", "end": "2025-06-20T..."}}
Location examples: "brown" → "Brown University", "mit" → "MIT" """

        # Function definition for structured output
        function_def = {
            "name": "parse_search_components",
            "description": "Parse search query",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "Core terms only"
                    },
                    "location": {
                        "type": ["string", "null"],
                        "description": "Place name or null"
                    },
                    "search_radius": {
                        "type": ["number", "null"],
                        "description": "Radius in km or null"
                    },
                    "date": {
                        "type": ["object", "null"],
                        "description": "Date range or null",
                        "properties": {
                            "start": {
                                "type": "string",
                                "description": "Start UTC"
                            },
                            "end": {
                                "type": "string", 
                                "description": "End UTC"
                            }
                        }
                    }
                },
                "required": ["search_query", "location", "search_radius", "date"]
            }
        }

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Parse search queries for filtering."},
                {"role": "user", "content": prompt}
            ],
            tools=[{"type": "function", "function": function_def}],
            tool_choice={"type": "function", "function": {"name": "parse_search_components"}},
            max_tokens=400
        )

        # Extract the structured JSON output
        if response.choices[0].message.tool_calls:
            function_args = response.choices[0].message.tool_calls[0].function.arguments
            import json
            parsed_result = json.loads(function_args)
            
            return {
                "original_query": request.query,
                "parsed": parsed_result,
                "usage": response.usage.model_dump() if response.usage else None
            }
        else:
            raise HTTPException(status_code=500, detail="No function call found in OpenAI response")
            
    except Exception as e:
        logger.error(f"Search query parsing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MOONDREAM ENDPOINTS
# ============================================================================

@app.post("/api/moondream/image-analysis")
async def moondream_analysis(request: MoondreamAnalysisRequest):
    """Generate caption and extract prominent objects using Moondream API."""
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
        caption_result = model.caption(image, length="normal")
        caption = caption_result.get("caption", "")
        
        # Extract prominent objects using a targeted query
        objects_result = model.query(image, "What are the most prominent objects, people, or items visible in this image? List them as a comma-separated list.")
        objects_text = objects_result.get("answer", "")
        
        # Parse objects from the response (split by commas and clean up)
        objects = []
        if objects_text:
            objects = [obj.strip() for obj in objects_text.split(',') if obj.strip()]
        
        return {
            "caption": caption,
            "objects": objects,
            "raw_objects_response": objects_text
        }
    except ImportError:
        raise HTTPException(status_code=500, detail="Moondream SDK not installed")
    except Exception as e:
        logger.error(f"Moondream Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        result = model.caption(image, length="short")
        
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

@app.post("/api/google/forward-geocode")
async def google_forward_geocode(request: GoogleForwardGeocodeRequest):
    """Forward geocode location text to coordinates using Google Maps API."""
    try:
        # Check if Google Maps API key is available
        google_maps_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not google_maps_key:
            raise HTTPException(status_code=503, detail="Google Maps API key not configured")
        
        import httpx
        
        # Google Maps Geocoding API URL for forward geocoding
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            'address': request.location_text,
            'key': google_maps_key
        }
        
        # Make the API request
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Google Forward Geocoding API response status: {result.get('status')}")
            
            return result
            
    except httpx.HTTPError as e:
        logger.error(f"Google Forward Geocoding API HTTP error: {e}")
        raise HTTPException(status_code=502, detail="Google Maps API request failed")
    except Exception as e:
        logger.error(f"Google Forward Geocoding API error: {e}")
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