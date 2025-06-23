# silk.ai - Packaged Application

This directory contains the packaged version of silk.ai ready for building into a single executable.

## Features

- **On-demand model downloads**: Models are downloaded automatically on first launch
- **Cross-platform support**: Builds for macOS, Windows, and Linux
- **Single executable**: All dependencies bundled except Python models
- **Automatic model management**: Smart caching and offline mode support

## Building the Application

### Prerequisites

1. **Node.js** (v18 or higher)
2. **Python 3.8+** with required packages
3. **Platform-specific tools**:
   - macOS: Xcode Command Line Tools
   - Windows: Visual Studio Build Tools
   - Linux: build-essential

### Install Dependencies

```bash
# Install Node.js dependencies
npm install

# Install Python dependencies (for model downloader)
pip install -r requirements.txt
```

### Build Commands

```bash
# Build for current platform
npm run dist

# Build for specific platforms
npm run dist-mac    # macOS (Intel + Apple Silicon)
npm run dist-win    # Windows x64
npm run dist-linux  # Linux x64

# Development mode
npm run electron-dev
```

### Build Output

Built applications will be in the `dist/` directory:

- **macOS**: `silk.ai.dmg` (universal binary)
- **Windows**: `silk.ai Setup.exe` (NSIS installer)
- **Linux**: `silk.ai.AppImage` (portable executable)

## Model Management

### First Launch Behavior

1. App checks for required models in `~/.silk-ai/models/`
2. If missing, downloads SentenceTransformer (87MB) automatically
3. Shows progress dialog during download
4. Additional models (Whisper, YAMNet) download on-demand when needed

### Manual Model Management

Users can manually download models via the app's settings or using the CLI:

```bash
# Check model status
python python/model_downloader.py --status

# Download required models only
python python/model_downloader.py --required-only
```

### Offline Mode

- App works offline after initial model download
- Search functionality fully available offline
- Audio/video processing requires internet for API calls

## Directory Structure

```
packaging/
├── app/                    # Next.js app
├── backend/               # Python backend (copied)
├── python/               # Unified service + model downloader
├── main.js              # Electron main process
├── preload.js           # Electron preload script
├── package.json         # Build configuration
└── dist/               # Build output (created during build)
```

## Configuration

### App Metadata

- **App Name**: silk.ai
- **App ID**: com.silk.ai
- **Version**: 1.0.0
- **Category**: Productivity

### Model Storage

- **Development**: `.contentcache/models/`
- **Production**: `~/.silk-ai/models/`

### Supported File Types

- **Video**: mp4, mov, avi, mkv, wmv, flv, webm
- **Audio**: mp3, wav, aac, flac, m4a
- **Text**: txt, md, pdf, docx, rtf
- **Images**: jpg, jpeg, png, bmp, tiff, webp, heic

## Troubleshooting

### Build Issues

1. **Python not found**: Ensure Python 3.8+ is in PATH
2. **Native dependencies**: Install platform build tools
3. **Memory issues**: Increase Node.js heap size: `export NODE_OPTIONS="--max-old-space-size=8192"`

### Runtime Issues

1. **Models not downloading**: Check internet connection
2. **Search not working**: Verify SentenceTransformer model downloaded
3. **Python errors**: Check Python dependencies in requirements.txt

### Model Download Issues

1. **Connection timeout**: Retry or check firewall settings
2. **Disk space**: Ensure 300MB+ free space for all models
3. **Permissions**: Verify write access to `~/.silk-ai/` directory

## Performance

### App Size

- **Installer**: ~200MB (without models)
- **With required models**: ~300MB total
- **With all models**: ~500MB total

### Memory Usage

- **Base app**: ~200MB RAM
- **With models loaded**: ~400-600MB RAM
- **During processing**: ~800MB-1.2GB RAM

### Startup Time

- **First launch**: 30-60 seconds (model download)
- **Subsequent launches**: 3-5 seconds
- **Model loading**: 5-10 seconds

## License

This packaged version includes all dependencies and follows their respective licenses. 