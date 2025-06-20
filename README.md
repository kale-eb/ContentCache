# ContentCache API Server

A FastAPI server that centralizes all AI and external API calls for the ContentCache video/image processing system.

## 🚀 Features

- **OpenAI Integration**: Chat completions, GPT-4o Vision, and Whisper transcription
- **Moondream API**: Image captioning and analysis
- **Google Maps API**: Reverse geocoding and nearby places search
- **EasyOCR**: Text extraction from images
- **Concurrent Processing**: Optimized for batch operations

## 📋 API Endpoints

### Health & Status
- `GET /` - Root endpoint
- `GET /health` - Health check with service status

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

### File Upload
- `POST /api/upload/process-image` - Process uploaded image files

## 🛠️ Setup

### Local Development

1. **Clone and setup:**
   ```bash
   git clone <your-repo>
   cd contentcache-api
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

3. **Run the server:**
   ```bash
   uvicorn main:app --reload --port 8000
   ```

4. **View API docs:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Railway Deployment

1. **Create GitHub repository:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo>
   git push -u origin main
   ```

2. **Deploy to Railway:**
   - Go to [Railway](https://railway.app)
   - Connect your GitHub repository
   - Add environment variables in Railway dashboard
   - Deploy automatically triggers

3. **Configure environment variables in Railway:**
   - `OPENAI_API_KEY`
   - `MOONDREAM_API_KEY`
   - `GOOGLE_MAPS_API_KEY`

## 🔧 Configuration

### Required Environment Variables

```bash
OPENAI_API_KEY=your_openai_api_key
MOONDREAM_API_KEY=your_moondream_api_key
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
```

### Optional Variables

```bash
LOG_LEVEL=INFO
```

## 📝 Usage Examples

### OpenAI Vision Analysis
```python
import requests

response = requests.post("http://your-api-url/api/openai/vision", json={
    "images_base64": ["base64_encoded_image"],
    "prompt": "Analyze this image",
    "max_tokens": 500
})
```

### Moondream Captioning
```python
response = requests.post("http://your-api-url/api/moondream/caption", json={
    "image_base64": "base64_encoded_image"
})
```

### OCR Text Extraction
```python
response = requests.post("http://your-api-url/api/ocr/extract-text", json={
    "images_base64": ["base64_encoded_image1", "base64_encoded_image2"]
})
```

## 🏗️ Architecture

This API server is designed to be called by the main ContentCache application, replacing direct API calls with centralized endpoints that can be:

- **Scaled independently**
- **Cached for performance**
- **Monitored centrally**
- **Rate-limited appropriately**

## 🔒 Security Notes

- Configure CORS appropriately for production
- Use environment variables for all API keys
- Consider adding API key authentication for your endpoints
- Monitor usage and implement rate limiting as needed

## 📊 Monitoring

The `/health` endpoint provides service status:
```json
{
  "status": "healthy",
  "services": {
    "openai": true,
    "moondream": true,
    "google_maps": true,
    "easyocr": true
  }
}
```

## 🚢 Deployment Status

Once deployed to Railway, your API will be available at:
`https://your-app-name.railway.app` 