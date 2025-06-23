#!/bin/bash

# silk.ai Build Script
# Builds the complete application for distribution

set -e  # Exit on any error

echo "🚀 Building silk.ai for distribution..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
print_status "Checking prerequisites..."

# Check Node.js
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js 18+ and try again."
    exit 1
fi

NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    print_error "Node.js version 18+ is required. Current version: $(node --version)"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

print_success "Prerequisites check passed"

# Clean previous builds
print_status "Cleaning previous builds..."
rm -rf dist out .next
print_success "Cleaned previous builds"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    print_status "Installing Node.js dependencies..."
    npm install
    print_success "Node.js dependencies installed"
else
    print_status "Node.js dependencies already installed"
fi

# Build Next.js app
print_status "Building Next.js application..."
npm run build
print_success "Next.js build completed"

# Test model downloader
print_status "Testing model downloader..."
python3 python/model_downloader.py --status > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_success "Model downloader test passed"
else
    print_warning "Model downloader test failed, but continuing build"
fi

# Build Electron app
print_status "Building Electron application..."

# Determine platform
PLATFORM=$(uname -s)
case $PLATFORM in
    Darwin)
        print_status "Building for macOS..."
        npm run dist-mac
        DIST_FILE="dist/silk.ai.dmg"
        ;;
    Linux)
        print_status "Building for Linux..."
        npm run dist-linux
        DIST_FILE="dist/silk.ai.AppImage"
        ;;
    MINGW*|CYGWIN*|MSYS*)
        print_status "Building for Windows..."
        npm run dist-win
        DIST_FILE="dist/silk.ai Setup.exe"
        ;;
    *)
        print_status "Unknown platform, building for current platform..."
        npm run dist
        DIST_FILE="dist/*"
        ;;
esac

print_success "Electron build completed"

# Check build output
print_status "Checking build output..."
if [ -d "dist" ] && [ "$(ls -A dist)" ]; then
    print_success "Build artifacts created in dist/ directory"
    
    # Show build summary
    echo ""
    echo "📦 Build Summary:"
    echo "=================="
    ls -lh dist/
    
    # Calculate total size
    TOTAL_SIZE=$(du -sh dist/ | cut -f1)
    echo ""
    echo "📊 Total package size: $TOTAL_SIZE"
    
    # Show next steps
    echo ""
    echo "🎉 Build completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Test the application: Open the built package"
    echo "2. Distribute: Share the package with users"
    echo "3. First launch: App will download required models (~87MB)"
    echo ""
    
else
    print_error "Build failed - no artifacts found in dist/ directory"
    exit 1
fi

print_success "silk.ai build process completed!" 