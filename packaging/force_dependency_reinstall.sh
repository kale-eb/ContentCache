#!/bin/bash

# Force Dependency Reinstall Script
# This script clears the dependency marker to force fresh installation on next app launch

echo "🔄 Forcing dependency reinstallation..."

# Remove the marker file that indicates dependencies are installed
MARKER_FILE="$HOME/.silk-ai-deps-installed"

if [ -f "$MARKER_FILE" ]; then
    rm "$MARKER_FILE"
    echo "✅ Removed dependency marker file: $MARKER_FILE"
    echo "📋 Next app launch will reinstall all dependencies with version 2.0.0"
    echo "📋 This includes:"
    echo "   - TensorFlow & TensorFlow Hub (video processing)"
    echo "   - PyTorch & TorchAudio (audio processing)"
    echo "   - OpenCV & EasyOCR (image/text processing)"
    echo "   - Scikit-image (frame similarity)"
    echo "   - Whisper (audio transcription)"
    echo "   - Complete document processing suite"
    echo "   - All search functionality packages"
else
    echo "⚠️ Marker file not found: $MARKER_FILE"
    echo "📋 Dependencies will be installed fresh on next app launch"
fi

echo ""
echo "🚀 Next steps:"
echo "1. Launch the silk.ai app"
echo "2. Wait for dependency installation to complete"
echo "3. Try processing your video again"
echo ""
echo "💡 The app will show a loading screen during installation" 