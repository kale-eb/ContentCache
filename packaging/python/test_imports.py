#!/usr/bin/env python3
"""
Diagnostic script to test all critical imports for ContentCache backend.
This helps identify missing dependencies before the main application starts.
"""

import sys
import os
from pathlib import Path

def test_import(module_name, description=""):
    """Test importing a module and return success status."""
    try:
        __import__(module_name)
        print(f"✅ {module_name} - {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - {description}: {e}")
        return False
    except Exception as e:
        print(f"⚠️ {module_name} - {description}: Unexpected error: {e}")
        return False

def test_backend_imports():
    """Test all backend-specific imports."""
    print("🔧 Testing backend module imports...")
    
    # Add backend paths to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_backend_dirs = [
        os.path.join(current_dir, 'backend', 'processing'),
        os.path.join(current_dir, 'python-dist', 'backend', 'processing'),
        os.path.join(os.path.dirname(current_dir), 'backend', 'processing'),
        os.path.join(os.path.dirname(current_dir), 'python-dist', 'backend', 'processing'),
    ]
    
    backend_dir = None
    for possible_dir in possible_backend_dirs:
        if os.path.exists(possible_dir):
            backend_dir = possible_dir
            sys.path.insert(0, possible_dir)
            print(f"📁 Added to Python path: {possible_dir}")
            break
    
    if not backend_dir:
        print("❌ No backend directory found!")
        return False
    
    # Test backend modules
    success = True
    success &= test_import("videotagger", "Video processing")
    success &= test_import("imageprocessor", "Image processing")
    success &= test_import("textprocessor", "Text processing")
    success &= test_import("audioanalyzer", "Audio processing")
    success &= test_import("tagdirectory", "Directory processing")
    success &= test_import("config", "Configuration")
    success &= test_import("framesegmentation", "Frame extraction")
    success &= test_import("framestagging", "Frame analysis")
    
    return success

def main():
    """Run all diagnostic tests."""
    print("🔍 ContentCache Import Diagnostics")
    print("=" * 50)
    
    print(f"🐍 Python version: {sys.version}")
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📁 Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print()
    
    # Test critical dependencies
    print("🔧 Testing critical dependencies...")
    dependencies_ok = True
    dependencies_ok &= test_import("numpy", "Numerical computing")
    dependencies_ok &= test_import("cv2", "OpenCV for image/video processing")
    dependencies_ok &= test_import("PIL", "Python Imaging Library")
    dependencies_ok &= test_import("natsort", "Natural sorting for file operations")
    dependencies_ok &= test_import("requests", "HTTP requests")
    dependencies_ok &= test_import("flask", "Web framework for search server")
    dependencies_ok &= test_import("nltk", "Natural language processing")
    dependencies_ok &= test_import("rank_bm25", "BM25 search algorithm")
    dependencies_ok &= test_import("sentence_transformers", "Sentence embeddings")
    dependencies_ok &= test_import("torch", "PyTorch machine learning")
    print()
    
    # Test backend imports
    backend_ok = test_backend_imports()
    print()
    
    # Final summary
    print("📊 Diagnostic Summary:")
    print(f"  Critical dependencies: {'✅ OK' if dependencies_ok else '❌ FAILED'}")
    print(f"  Backend modules: {'✅ OK' if backend_ok else '❌ FAILED'}")
    
    if dependencies_ok and backend_ok:
        print("🎉 All imports successful! Backend should work properly.")
        return 0
    else:
        print("⚠️ Some imports failed. Check error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 