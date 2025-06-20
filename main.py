#!/usr/bin/env python3
"""
ContentCache API Server Entry Point
This file allows Railway to detect this as a Python project
and properly run the FastAPI server from the api subdirectory.
"""

import sys
import os
import subprocess

# Add the api directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

if __name__ == "__main__":
    # Get the port from environment variable (Railway provides this)
    port = os.environ.get('PORT', '8000')
    
    # Change to the api directory and run uvicorn
    os.chdir('api')
    
    # Run the FastAPI server
    cmd = [
        'uvicorn', 
        'main:app', 
        '--host', '0.0.0.0', 
        '--port', port
    ]
    
    print(f"Starting ContentCache API server on port {port}")
    subprocess.run(cmd) 