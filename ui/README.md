# ContentCache Desktop UI

This directory contains the Electron-based desktop application for ContentCache.

## Directory Structure

```
ui/
├── main.js          # Electron main process
├── renderer.js      # UI logic and event handling
├── preload.js       # Secure bridge between main and renderer
├── index.html       # Main UI layout
├── styles.css       # Application styling
├── package.json     # Node.js dependencies and scripts
└── node_modules/    # Node.js dependencies (auto-generated)
```

## Getting Started

### Prerequisites
- Node.js (v16 or higher)
- Python 3.8+ (for backend services)

### Installation
```bash
# From the ui directory
npm install
```

### Running the Application
```bash
# Start the desktop application
npm start

# Development mode (with DevTools)
npm run dev
```

### Building for Distribution
```bash
# Build for current platform
npm run build

# Build for specific platforms
npm run build-mac
npm run build-win
npm run build-linux
```

## Backend Integration

The UI automatically starts the backend search server when launched. The backend services are located in `../backend/` relative to this directory.

## Features

- **File Processing**: Drag & drop or select files/directories for AI analysis
- **Smart Search**: Semantic search across videos, images, text, and audio
- **Real-time Results**: Live updates during processing
- **Cross-platform**: Works on macOS, Windows, and Linux 