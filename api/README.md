# ContentCache API Server

FastAPI server that centralizes all AI and external API calls for the ContentCache system.

## 🚀 Quick Start

### Local Development

```bash
# From the main contentcache directory
cd api
uvicorn main:app --reload --port 8000
```

### View API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📋 API Endpoints

### OpenAI Services
- `POST /api/openai/chat` - Generic chat completions  
- `POST /api/openai/vision` - GPT-4o vision analysis
- `POST /api/openai/transcribe` - Whisper audio transcription

### Moondream Services  
- `POST /api/moondream/caption` - Single image captioning
- `POST /api/moondream/batch` - Batch image processing

### Google Maps Services
- `POST /api/google/reverse-geocode` - Convert coordinates to addresses
- `POST /api/google/nearby-search` - Find nearby places

### OCR Services
- `POST /api/ocr/extract-text` - Extract text from images

## 🚂 Railway Deployment

This API can be deployed to Railway by pointing to this subdirectory:

1. Create GitHub repo from this entire project
2. In Railway, deploy the root directory
3. Railway will use the `railway.toml` configuration which runs from the `api/` subdirectory
4. Set environment variables in Railway dashboard

## 🔧 Environment Variables

Set these in Railway (or local `.env`):
```
OPENAI_API_KEY=your_key
MOONDREAM_API_KEY=your_key  
GOOGLE_MAPS_API_KEY=your_key
```

## 📝 Usage

The main ContentCache application will call these endpoints instead of direct API calls, providing:
- Centralized API management
- Better error handling  
- Easier scaling
- Cost monitoring 