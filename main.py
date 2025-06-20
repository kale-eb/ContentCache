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
import easyocr
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

# Initialize clients
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize EasyOCR (expensive operation, do once)
try:
    easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    print("✅ EasyOCR initialized successfully")
except Exception as e:
    easyocr_reader = None
    print(f"⚠️ Could not initialize EasyOCR: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class VisionAnalysisRequest(BaseModel):
    images_base64: List[str]
    prompt: str
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.2

class MoondreamRequest(BaseModel):
    image_base64: str
    prompt: Optional[str] = "Describe this image in detail."

class GoogleMapsRequest(BaseModel):
    latitude: float
    longitude: float
    radius: Optional[int] = 500

class OCRRequest(BaseModel):
    images_base64: List[str]

class AudioTranscriptionRequest(BaseModel):
    audio_base64: str
    file_format: str = "mp3"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def encode_image_to_base64(image_path: str, resize: bool = True, max_dimension: int = 512, quality: int = 50) -> str:
    """Encode image to base64 with optional compression."""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            if resize:
                width, height = img.size
                if width > height:
                    new_width = max_dimension
                    new_height = int((height * max_dimension) / width)
                else:
                    new_height = max_dimension
                    new_width = int((width * max_dimension) / height)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            buffer.seek(0)
            
            return base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        return None

def decode_base64_to_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {"message": "ContentCache API Server", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "moondream": bool(os.getenv("MOONDREAM_API_KEY")),
            "google_maps": bool(os.getenv("GOOGLE_MAPS_API_KEY")),
            "easyocr": easyocr_reader is not None
        }
    }

# ============================================================================
# OPENAI ENDPOINTS
# ============================================================================

@app.post("/api/openai/chat")
async def openai_chat(request: OpenAIRequest):
    """Generic OpenAI chat completions endpoint."""
    try:
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

@app.post("/api/openai/vision")
async def openai_vision_analysis(request: VisionAnalysisRequest):
    """Analyze images using GPT-4o vision."""
    try:
        # Prepare messages with images
        content = [{"type": "text", "text": request.prompt}]
        
        for img_base64 in request.images_base64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            })
        
        messages = [{"role": "user", "content": content}]
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return {
            "analysis": response.choices[0].message.content,
            "usage": response.usage.model_dump() if response.usage else None
        }
    except Exception as e:
        logger.error(f"OpenAI Vision API error: {e}")
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
async def moondream_batch_process(images_base64: List[str] = Form(...)):
    """Process multiple images with Moondream API."""
    try:
        moondream_api_key = os.getenv("MOONDREAM_API_KEY")
        if not moondream_api_key:
            raise HTTPException(status_code=500, detail="MOONDREAM_API_KEY not configured")
        
        import moondream as md
        model = md.vl(api_key=moondream_api_key)
        
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
# OCR ENDPOINTS
# ============================================================================

@app.post("/api/ocr/extract-text")
async def ocr_extract_text(request: OCRRequest):
    """Extract text from images using EasyOCR."""
    try:
        if not easyocr_reader:
            raise HTTPException(status_code=500, detail="EasyOCR not initialized")
        
        results = []
        for i, img_base64 in enumerate(request.images_base64):
            try:
                # Decode base64 image
                image_data = base64.b64decode(img_base64)
                image_array = Image.open(io.BytesIO(image_data))
                
                # Convert PIL image to numpy array for EasyOCR
                import numpy as np
                image_np = np.array(image_array)
                
                # Extract text
                text_results = easyocr_reader.readtext(image_np)
                
                # Process results
                extracted_texts = []
                for (bbox, text, confidence) in text_results:
                    if confidence > 0.5:  # Filter low confidence results
                        extracted_texts.append({
                            "text": text,
                            "confidence": confidence,
                            "bbox": bbox
                        })
                
                results.append({
                    "index": i,
                    "texts": extracted_texts,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "error": str(e),
                    "success": False
                })
        
        return {"results": results}
    except Exception as e:
        logger.error(f"OCR processing error: {e}")
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