# ContentCache Model Packaging Guide

## Overview

ContentCache uses several ML models that need to be downloaded and cached for the app to work. This guide explains how models are downloaded, cached, and how to package the app for distribution.

## Model Architecture

### Models Used

1. **SentenceTransformer (`all-MiniLM-L6-v2`)** - 87MB
   - Used for semantic embeddings in search
   - Downloaded from Hugging Face Hub
   - Cached in `.contentcache/models/models--sentence-transformers--all-MiniLM-L6-v2/`

2. **Whisper (`base`)** - 139MB 
   - Used for audio transcription
   - Downloaded from OpenAI
   - Cached in `.contentcache/models/base.pt`

3. **YAMNet** - 17MB
   - Used for audio classification
   - Downloaded from TensorFlow Hub
   - Cached in `.contentcache/models/tfhub/`

**Total Model Size: ~243MB**

## Model Download Behavior

### First-Time Installation

When models are not cached, they are automatically downloaded:

1. **SentenceTransformer**: 
   ```python
   # backend/search/search_server.py:314
   self.sentence_model = SentenceTransformer(
       'all-MiniLM-L6-v2',
       cache_folder=self.model_cache_dir,  # .contentcache/models/
       device=device
   )
   ```

2. **Whisper**:
   ```python
   # backend/processing/audioprocessor.py:43
   model = whisper.load_model("base", download_root=MODEL_CACHE_DIR)
   ```

3. **YAMNet**:
   ```python
   # backend/processing/audioprocessor.py:49
   os.environ['TFHUB_CACHE_DIR'] = os.path.join(MODEL_CACHE_DIR, 'tfhub')
   yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
   ```

### Cache Directory Structure

```
.contentcache/models/
├── .locks/                                    # Download locks
├── base.pt                                   # Whisper model (139MB)
├── datasets/                                 # Hugging Face datasets cache
├── models--sentence-transformers--all-MiniLM-L6-v2/  # SentenceTransformer (87MB)
│   ├── blobs/
│   ├── refs/
│   └── snapshots/
└── tfhub/                                   # TensorFlow Hub cache (17MB)
```

## Packaging Strategies

### Option 1: Include Pre-Downloaded Models

**Pros**: No internet required, faster startup
**Cons**: Large app size (~243MB extra)

1. Download models in development:
   ```bash
   # Run the app once to download all models
   cd backend/search && python3 search_server.py
   # Process some content to trigger all model downloads
   ```

2. Include `.contentcache/models/` in the app package

3. Set environment variables to prevent re-downloading:
   ```python
   os.environ['HF_HUB_OFFLINE'] = '1'
   os.environ['TRANSFORMERS_OFFLINE'] = '1'
   ```

### Option 2: Download on First Run

**Pros**: Smaller app size
**Cons**: Requires internet, slower first startup

1. Package app without models
2. Show download progress on first run
3. Handle download failures gracefully

### Option 3: Hybrid Approach (Recommended)

**Pros**: Balance of size and reliability
**Cons**: More complex

1. Include only critical models (SentenceTransformer - 87MB)
2. Download others on demand
3. Provide offline mode with reduced functionality

## Implementation for Packaging

### 1. Pre-Download Script

```python
#!/usr/bin/env python3
"""
Pre-download all models for packaging
"""
import os
from backend.processing.config import get_models_cache_dir
from sentence_transformers import SentenceTransformer
import whisper
import tensorflow_hub as hub

def download_all_models():
    """Download all models to cache directory"""
    model_cache_dir = get_models_cache_dir()
    print(f"Downloading models to: {model_cache_dir}")
    
    # SentenceTransformer
    print("Downloading SentenceTransformer...")
    SentenceTransformer('all-MiniLM-L6-v2', cache_folder=model_cache_dir)
    
    # Whisper
    print("Downloading Whisper...")
    whisper.load_model("base", download_root=model_cache_dir)
    
    # YAMNet
    print("Downloading YAMNet...")
    os.environ['TFHUB_CACHE_DIR'] = os.path.join(model_cache_dir, 'tfhub')
    hub.load("https://tfhub.dev/google/yamnet/1")
    
    print("All models downloaded successfully!")

if __name__ == "__main__":
    download_all_models()
```

### 2. Offline Mode Detection

```python
def check_offline_mode():
    """Check if we're in offline mode or models are cached"""
    model_cache_dir = get_models_cache_dir()
    
    # Check if critical models exist
    sentence_transformer_path = os.path.join(
        model_cache_dir, 
        "models--sentence-transformers--all-MiniLM-L6-v2"
    )
    whisper_path = os.path.join(model_cache_dir, "base.pt")
    
    if os.path.exists(sentence_transformer_path) and os.path.exists(whisper_path):
        return True
    
    # Test internet connectivity
    try:
        import requests
        requests.get("https://huggingface.co", timeout=5)
        return False
    except:
        return True
```

### 3. Graceful Model Loading

```python
def load_model_with_fallback(model_name, cache_dir):
    """Load model with graceful fallback"""
    try:
        # Try normal loading
        return SentenceTransformer(model_name, cache_folder=cache_dir)
    except Exception as e:
        if "offline" in str(e).lower() or "connection" in str(e).lower():
            # Try offline mode
            try:
                return SentenceTransformer(model_name, cache_folder=cache_dir, local_files_only=True)
            except:
                raise RuntimeError(f"Model {model_name} not cached and no internet connection")
        raise e
```

## Electron Packaging Considerations

### 1. Include Models in Resources

```javascript
// In main.js
const modelsPath = app.isPackaged 
    ? path.join(process.resourcesPath, '.contentcache', 'models')
    : path.join(__dirname, '..', '.contentcache', 'models');

// Set environment variable for Python processes
process.env.CONTENTCACHE_MODELS_DIR = modelsPath;
```

### 2. Update Config for Packaged App

```python
# In config.py
def get_models_cache_dir():
    """Get the directory for cached models."""
    # Check if running in packaged app
    if os.environ.get('CONTENTCACHE_MODELS_DIR'):
        return os.environ.get('CONTENTCACHE_MODELS_DIR')
    
    # Default behavior
    models_dir = os.path.join(get_app_cache_dir(), "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir
```

### 3. Package.json Configuration

```json
{
  "build": {
    "extraResources": [
      {
        "from": ".contentcache/models",
        "to": ".contentcache/models",
        "filter": ["**/*"]
      }
    ],
    "compression": "maximum",
    "nsis": {
      "allowToChangeInstallationDirectory": true,
      "createDesktopShortcut": "always"
    }
  }
}
```

## Download Progress UI

### Show Download Progress

```javascript
// In main.js
function showModelDownloadProgress(modelName, progress) {
    mainWindow.webContents.send('model-download-progress', {
        model: modelName,
        progress: progress,
        message: `Downloading ${modelName}... ${progress}%`
    });
}
```

```tsx
// In React component
useEffect(() => {
    window.electronAPI?.onModelDownloadProgress?.((event, data) => {
        setDownloadProgress(data);
    });
}, []);
```

## Testing Offline Mode

1. **Simulate Offline Environment**:
   ```bash
   # Block internet access
   sudo iptables -A OUTPUT -d huggingface.co -j DROP
   # Test app startup
   ```

2. **Verify Model Loading**:
   ```python
   # Test script
   os.environ['HF_HUB_OFFLINE'] = '1'
   model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
   ```

## Recommended Packaging Approach

1. **Include SentenceTransformer model** (87MB) - Critical for search
2. **Download Whisper on demand** - Only needed for audio processing
3. **Download YAMNet as needed** - For audio classification
4. **Provide offline mode** - Reduced functionality without internet
5. **Show clear progress indicators** - For model downloads
6. **Handle failures gracefully** - With helpful error messages

This approach balances app size (~100MB extra) with functionality and user experience. 