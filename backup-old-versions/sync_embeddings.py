#!/usr/bin/env python3
"""
Embedding Synchronization Utility for ContentCache

This script ensures all metadata files have corresponding embeddings for search functionality.
Run this after updating metadata or if you notice discrepancies between metadata and embeddings.
"""

import os
import sys
import json
from pathlib import Path

# Add backend/processing to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'processing'))

from config import (
    get_video_metadata_path, get_image_metadata_path,
    get_audio_metadata_path, get_text_metadata_path,
    get_embeddings_cache_dir
)
from embedding_generator import generate_embeddings_from_metadata_file

def count_existing_embeddings():
    """Count existing embeddings by content type."""
    embeddings_dir = get_embeddings_cache_dir()
    if not os.path.exists(embeddings_dir):
        return {'video': 0, 'image': 0, 'audio': 0, 'text': 0}
    
    counts = {'video': 0, 'image': 0, 'audio': 0, 'text': 0}
    
    for filename in os.listdir(embeddings_dir):
        if filename.endswith('.pkl'):
            for content_type in counts.keys():
                if filename.startswith(f'{content_type}_embeddings_'):
                    # Load and count
                    try:
                        import pickle
                        with open(os.path.join(embeddings_dir, filename), 'rb') as f:
                            data = pickle.load(f)
                            if isinstance(data, dict) and 'paths' in data:
                                counts[content_type] += len(data['paths'])
                            elif isinstance(data, dict):
                                counts[content_type] += len(data)
                    except Exception as e:
                        print(f"⚠️ Failed to read {filename}: {e}")
    
    return counts

def count_metadata_items():
    """Count metadata items by content type."""
    metadata_paths = {
        'video': get_video_metadata_path(),
        'image': get_image_metadata_path(),
        'audio': get_audio_metadata_path(),
        'text': get_text_metadata_path()
    }
    
    counts = {}
    for content_type, metadata_file in metadata_paths.items():
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    counts[content_type] = len(metadata)
            except Exception as e:
                print(f"⚠️ Failed to read {content_type} metadata: {e}")
                counts[content_type] = 0
        else:
            counts[content_type] = 0
    
    return counts

def sync_embeddings_for_type(content_type, force_regenerate=False):
    """Sync embeddings for a specific content type."""
    metadata_paths = {
        'video': get_video_metadata_path(),
        'image': get_image_metadata_path(),
        'audio': get_audio_metadata_path(),
        'text': get_text_metadata_path()
    }
    
    metadata_file = metadata_paths[content_type]
    
    if not os.path.exists(metadata_file):
        print(f"⚠️ No {content_type} metadata file found at: {metadata_file}")
        return False
    
    print(f"🔄 Syncing {content_type} embeddings...")
    
    try:
        result = generate_embeddings_from_metadata_file(content_type, metadata_file, force_regenerate)
        if result:
            print(f"✅ {content_type.title()} embeddings synced successfully")
            return True
        else:
            print(f"❌ Failed to sync {content_type} embeddings")
            return False
    except Exception as e:
        print(f"❌ Error syncing {content_type} embeddings: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Sync embeddings with metadata files")
    parser.add_argument("--content-type", choices=["video", "image", "audio", "text", "all"], 
                       default="all", help="Content type to sync (default: all)")
    parser.add_argument("--force-regenerate", action="store_true",
                       help="Force regeneration of all embeddings from scratch")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check for discrepancies, don't sync")
    
    args = parser.parse_args()
    
    print("🚀 ContentCache Embedding Synchronization Tool")
    print("=" * 50)
    
    # Count current state
    print("\n📊 Current State:")
    metadata_counts = count_metadata_items()
    embedding_counts = count_existing_embeddings()
    
    total_metadata = sum(metadata_counts.values())
    total_embeddings = sum(embedding_counts.values())
    
    for content_type in ['video', 'image', 'audio', 'text']:
        meta_count = metadata_counts[content_type]
        emb_count = embedding_counts[content_type]
        status = "✅" if meta_count == emb_count else "⚠️"
        print(f"  {status} {content_type.title()}: {meta_count} metadata, {emb_count} embeddings")
    
    print(f"\n📈 Summary: {total_metadata} metadata items, {total_embeddings} embeddings")
    
    if total_metadata == total_embeddings and not args.force_regenerate:
        print("✅ All embeddings are in sync!")
        return
    
    if args.check_only:
        if total_metadata != total_embeddings:
            print(f"⚠️ Found {total_metadata - total_embeddings} items missing embeddings")
        return
    
    # Perform synchronization
    print(f"\n🔄 Starting synchronization...")
    if args.force_regenerate:
        print("🔥 Force regenerating all embeddings from scratch")
    
    content_types = ['video', 'image', 'audio', 'text'] if args.content_type == 'all' else [args.content_type]
    
    success_count = 0
    for content_type in content_types:
        if sync_embeddings_for_type(content_type, args.force_regenerate):
            success_count += 1
    
    print(f"\n📊 Synchronization complete!")
    print(f"✅ Successfully synced {success_count}/{len(content_types)} content types")
    
    # Show final state
    print("\n📊 Final State:")
    final_metadata_counts = count_metadata_items()
    final_embedding_counts = count_existing_embeddings()
    
    for content_type in ['video', 'image', 'audio', 'text']:
        meta_count = final_metadata_counts[content_type]
        emb_count = final_embedding_counts[content_type]
        status = "✅" if meta_count == emb_count else "⚠️"
        print(f"  {status} {content_type.title()}: {meta_count} metadata, {emb_count} embeddings")
    
    final_total_metadata = sum(final_metadata_counts.values())
    final_total_embeddings = sum(final_embedding_counts.values())
    print(f"\n📈 Final Summary: {final_total_metadata} metadata items, {final_total_embeddings} embeddings")
    
    if final_total_metadata == final_total_embeddings:
        print("🎉 All embeddings are now perfectly synced!")
    else:
        print(f"⚠️ Still {abs(final_total_metadata - final_total_embeddings)} items out of sync")

if __name__ == "__main__":
    main() 