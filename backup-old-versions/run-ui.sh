#!/bin/bash

# ContentCache UI Launcher
# This script allows you to run the desktop UI from the project root

echo "🚀 Starting ContentCache Desktop UI..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js is not installed. Please install Node.js v16 or higher."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ Error: npm is not installed. Please install npm."
    exit 1
fi

# Navigate to UI directory
cd ui

# Check if node_modules exists, if not install dependencies
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Start the application
echo "🎯 Launching ContentCache..."
npm start 