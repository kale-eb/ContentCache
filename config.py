#!/usr/bin/env python3
"""
Configuration management for ContentCache
Handles all directory paths and ensures consistent location management.
"""

import os
import tempfile
import time
from pathlib import Path

# Base directories
def get_app_cache_dir():
    """Get the main application cache directory."""
    cache_dir = os.path.join(os.getcwd(), ".contentcache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def get_metadata_dir():
    """Get the directory for all metadata JSON files."""
    metadata_dir = os.path.join(get_app_cache_dir(), "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    return metadata_dir

def get_models_cache_dir():
    """Get the directory for cached models."""
    models_dir = os.path.join(get_app_cache_dir(), "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir

def get_embeddings_cache_dir():
    """Get the directory for cached embeddings."""
    embeddings_dir = os.path.join(get_app_cache_dir(), "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    return embeddings_dir

def get_temp_frames_dir(video_path):
    """Get a temporary directory for video frames."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = int(time.time())
    temp_dir = os.path.join(tempfile.gettempdir(), f"contentcache_{video_name}_frames_{timestamp}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

# Metadata file paths
def get_video_metadata_path():
    """Get path to video metadata JSON file."""
    return os.path.join(get_metadata_dir(), "video_metadata.json")

def get_audio_metadata_path():
    """Get path to audio metadata JSON file."""
    return os.path.join(get_metadata_dir(), "audio_metadata.json")

def get_text_metadata_path():
    """Get path to text metadata JSON file."""
    return os.path.join(get_metadata_dir(), "text_metadata.json")

def get_image_metadata_path():
    """Get path to image metadata JSON file."""
    return os.path.join(get_metadata_dir(), "image_metadata.json")

def get_memory_log_path():
    """Get path to memory log JSON file."""
    return os.path.join(get_metadata_dir(), "memory_log.json")

def get_failed_files_path():
    """Get path to failed files JSON file."""
    return os.path.join(get_metadata_dir(), "failed_files.json")

# Utility functions
def migrate_existing_metadata():
    """Move existing metadata files from current directory to cache directory."""
    current_dir = os.getcwd()
    metadata_dir = get_metadata_dir()
    
    metadata_files = [
        "video_metadata.json",
        "audio_metadata.json", 
        "text_metadata.json",
        "image_metadata.json",
        "memory_log.json",
        "failed_files.json"
    ]
    
    migrated_files = []
    
    for filename in metadata_files:
        old_path = os.path.join(current_dir, filename)
        new_path = os.path.join(metadata_dir, filename)
        
        if os.path.exists(old_path) and not os.path.exists(new_path):
            try:
                os.rename(old_path, new_path)
                migrated_files.append(filename)
                print(f"📁 Migrated {filename} to cache directory")
            except Exception as e:
                print(f"⚠️ Failed to migrate {filename}: {e}")
    
    if migrated_files:
        print(f"✅ Migrated {len(migrated_files)} metadata files to {metadata_dir}")
    else:
        print("📁 No metadata files to migrate")
    
    return migrated_files

def cleanup_temp_frames():
    """Clean up any leftover temp frame directories."""
    temp_dir = tempfile.gettempdir()
    cleaned = 0
    
    try:
        for item in os.listdir(temp_dir):
            if item.startswith("contentcache_") and item.endswith("_frames"):
                full_path = os.path.join(temp_dir, item)
                if os.path.isdir(full_path):
                    try:
                        import shutil
                        shutil.rmtree(full_path)
                        cleaned += 1
                    except Exception as e:
                        print(f"⚠️ Failed to clean temp directory {item}: {e}")
    except Exception as e:
        print(f"⚠️ Failed to access temp directory: {e}")
    
    if cleaned > 0:
        print(f"🧹 Cleaned up {cleaned} temporary frame directories")
    
    return cleaned

def print_directory_structure():
    """Print the current directory structure for debugging."""
    print("\n📁 ContentCache Directory Structure:")
    print(f"├── App Cache: {get_app_cache_dir()}")
    print(f"│   ├── Metadata: {get_metadata_dir()}")
    print(f"│   ├── Models: {get_models_cache_dir()}")
    print(f"│   └── Embeddings: {get_embeddings_cache_dir()}")
    print(f"└── Temp Frames: {tempfile.gettempdir()}/contentcache_*_frames_*")
    print()

if __name__ == "__main__":
    # Test the configuration
    print("🧪 Testing ContentCache configuration...")
    print_directory_structure()
    
    # Test migration
    migrate_existing_metadata()
    
    # Test cleanup
    cleanup_temp_frames()
    
    print("✅ Configuration test complete") 