#!/usr/bin/env python3
"""
Embedding Generator for ContentCache

Generates and manages semantic embeddings for all content types (text, audio, video, image)
using SentenceTransformer and BM25 for fast search functionality.

This module:
1. Uses the same model (all-MiniLM-L6-v2) as existing embeddings
2. Maintains the existing format: {'bm25': BM25Okapi, 'embeddings': ndarray, 'paths': list}
3. Can update existing embeddings or create new ones
4. Handles all content types consistently
"""

import os
import pickle
import hashlib
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

# ML imports
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Local imports
from config import get_embeddings_cache_dir, get_models_cache_dir

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Generates and manages semantic embeddings for content search.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: SentenceTransformer model to use (default: all-MiniLM-L6-v2)
        """
        self.model_name = model_name
        self.embeddings_cache_dir = get_embeddings_cache_dir()
        self.model_cache_dir = get_models_cache_dir()
        
        # Ensure directories exist
        Path(self.embeddings_cache_dir).mkdir(exist_ok=True)
        Path(self.model_cache_dir).mkdir(exist_ok=True)
        
        # Initialize model (lazy loading)
        self.sentence_model = None
        
        logger.info(f"EmbeddingGenerator initialized with model: {model_name}")
    
    def _load_model(self):
        """Load the SentenceTransformer model if not already loaded."""
        if self.sentence_model is None:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self.sentence_model = SentenceTransformer(
                self.model_name,
                cache_folder=self.model_cache_dir
            )
            logger.info("✅ SentenceTransformer model loaded successfully")
    
    def _get_cache_filename(self, content_type: str, content_paths: List[str]) -> str:
        """
        Generate a cache filename based on content type and paths hash.
        
        Args:
            content_type: Type of content (text, audio, video, image)
            content_paths: List of file paths being embedded
            
        Returns:
            Cache filename
        """
        # Create a hash of all paths for uniqueness
        paths_str = "\n".join(sorted(content_paths))
        paths_hash = hashlib.md5(paths_str.encode()).hexdigest()
        return f"{content_type}_embeddings_{paths_hash}.pkl"
    
    def _extract_text_content(self, content_type: str, file_path: str, metadata: Dict[str, Any]) -> str:
        """
        Extract searchable text content from metadata based on content type.
        
        Args:
            content_type: Type of content (text, audio, video, image)
            file_path: Path to the content file
            metadata: Content metadata
            
        Returns:
            Concatenated text content for embedding
        """
        text_parts = []
        
        # Add filename (without extension) as searchable content
        filename = Path(file_path).stem
        text_parts.append(filename)
        
        if content_type == "video":
            # Extract from video summary and tags
            if "video_summary" in metadata:
                summary = metadata["video_summary"]
                if isinstance(summary, str):
                    text_parts.append(summary)
                elif isinstance(summary, dict):
                    # Handle structured summary
                    for key, value in summary.items():
                        if isinstance(value, str):
                            text_parts.append(value)
                        elif isinstance(value, list):
                            text_parts.extend([str(v) for v in value])
            
            if "tags" in metadata:
                tags = metadata["tags"]
                if isinstance(tags, dict):
                    for key, value in tags.items():
                        if isinstance(value, list):
                            text_parts.extend(value)
                        elif isinstance(value, str):
                            text_parts.append(value)
                elif isinstance(tags, list):
                    text_parts.extend([str(t) for t in tags])
            
            # Add location info
            if "metadata" in metadata and "location" in metadata["metadata"]:
                location = metadata["metadata"]["location"]
                if location and location != "None":
                    text_parts.append(location)
        
        elif content_type == "image":
            # Extract from image analysis
            if "analysis" in metadata:
                analysis = metadata["analysis"]
                if isinstance(analysis, str):
                    text_parts.append(analysis)
                elif isinstance(analysis, dict):
                    for key, value in analysis.items():
                        if isinstance(value, str):
                            text_parts.append(value)
                        elif isinstance(value, list):
                            text_parts.extend([str(v) for v in value])
            
            if "summary" in metadata:
                text_parts.append(str(metadata["summary"]))
            
            # Add location info
            if "location" in metadata and metadata["location"]:
                text_parts.append(metadata["location"])
        
        elif content_type == "audio":
            # Extract from audio analysis
            if "analysis" in metadata:
                analysis = metadata["analysis"]
                if isinstance(analysis, str):
                    text_parts.append(analysis)
                elif isinstance(analysis, dict):
                    for key, value in analysis.items():
                        if isinstance(value, str):
                            text_parts.append(value)
                        elif isinstance(value, list):
                            text_parts.extend([str(v) for v in value])
            
            if "summary" in metadata:
                text_parts.append(str(metadata["summary"]))
        
        elif content_type == "text":
            # Extract from text analysis
            if "analysis" in metadata:
                analysis = metadata["analysis"]
                if isinstance(analysis, str):
                    text_parts.append(analysis)
                elif isinstance(analysis, dict):
                    for key, value in analysis.items():
                        if isinstance(value, str):
                            text_parts.append(value)
                        elif isinstance(value, list):
                            text_parts.extend([str(v) for v in value])
            
            # Add content preview if available
            if "content_preview" in metadata:
                text_parts.append(metadata["content_preview"])
        
        # Join all text parts and clean up
        combined_text = " ".join(text_parts)
        
        # Clean up the text (remove extra whitespace, newlines, etc.)
        cleaned_text = " ".join(combined_text.split())
        
        return cleaned_text
    
    def _load_existing_embeddings(self, cache_file: str) -> Optional[Dict[str, Any]]:
        """
        Load existing embeddings from cache file.
        
        Args:
            cache_file: Path to cache file
            
        Returns:
            Existing embeddings data or None if file doesn't exist
        """
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Loaded existing embeddings: {len(data.get('paths', []))} items")
                return data
            except Exception as e:
                logger.warning(f"Failed to load existing embeddings from {cache_file}: {e}")
        return None
    
    def _create_embeddings_data(self, paths: List[str], texts: List[str]) -> Dict[str, Any]:
        """
        Create embeddings data structure with BM25 and SentenceTransformer embeddings.
        
        Args:
            paths: List of file paths
            texts: List of text content for embedding
            
        Returns:
            Embeddings data structure
        """
        self._load_model()
        
        logger.info(f"Generating embeddings for {len(texts)} items...")
        
        # Generate SentenceTransformer embeddings
        embeddings = self.sentence_model.encode(texts, convert_to_tensor=False)
        embeddings_array = np.array(embeddings)
        
        # Create BM25 index for keyword search
        # Tokenize texts for BM25 (simple word splitting)
        tokenized_texts = [text.lower().split() for text in texts]
        bm25 = BM25Okapi(tokenized_texts)
        
        # Create the data structure
        data = {
            'bm25': bm25,
            'embeddings': embeddings_array,
            'paths': paths,
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name
        }
        
        logger.info(f"✅ Generated embeddings: shape {embeddings_array.shape}")
        return data
    
    def _merge_embeddings_data(self, existing_data: Dict[str, Any], new_paths: List[str], 
                              new_texts: List[str]) -> Dict[str, Any]:
        """
        Merge new embeddings with existing data, avoiding duplicates.
        
        Args:
            existing_data: Existing embeddings data
            new_paths: New file paths to add
            new_texts: New text content for embedding
            
        Returns:
            Merged embeddings data
        """
        existing_paths = existing_data.get('paths', [])
        existing_embeddings = existing_data.get('embeddings', np.array([]))
        
        # Find truly new paths (not already in existing data)
        unique_new_paths = []
        unique_new_texts = []
        
        for path, text in zip(new_paths, new_texts):
            if path not in existing_paths:
                unique_new_paths.append(path)
                unique_new_texts.append(text)
        
        if not unique_new_paths:
            logger.info("No new items to add - all paths already exist in embeddings")
            return existing_data
        
        logger.info(f"Adding {len(unique_new_paths)} new items to existing {len(existing_paths)} items")
        
        # Generate embeddings for new items only
        self._load_model()
        new_embeddings = self.sentence_model.encode(unique_new_texts, convert_to_tensor=False)
        new_embeddings_array = np.array(new_embeddings)
        
        # Merge paths
        all_paths = existing_paths + unique_new_paths
        
        # Merge embeddings
        if existing_embeddings.size > 0:
            all_embeddings = np.vstack([existing_embeddings, new_embeddings_array])
        else:
            all_embeddings = new_embeddings_array
        
        # Create new BM25 index with all texts
        all_texts = []
        
        # Get existing texts (reconstruct from paths - this is a limitation)
        # For now, we'll recreate BM25 from scratch with all available texts
        # In a future improvement, we could store original texts in the cache
        for path in existing_paths:
            # This is a fallback - ideally we'd have the original text
            all_texts.append(Path(path).stem)  # Use filename as fallback
        
        # Add new texts
        all_texts.extend(unique_new_texts)
        
        # Create new BM25 index
        tokenized_texts = [text.lower().split() for text in all_texts]
        bm25 = BM25Okapi(tokenized_texts)
        
        # Create merged data structure
        merged_data = {
            'bm25': bm25,
            'embeddings': all_embeddings,
            'paths': all_paths,
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name
        }
        
        logger.info(f"✅ Merged embeddings: {len(all_paths)} total items, shape {all_embeddings.shape}")
        return merged_data
    
    def generate_embeddings(self, content_type: str, metadata_dict: Dict[str, Dict[str, Any]], 
                          force_regenerate: bool = False) -> str:
        """
        Generate embeddings for a collection of content items.
        
        Args:
            content_type: Type of content (text, audio, video, image)
            metadata_dict: Dictionary mapping file paths to metadata
            force_regenerate: If True, regenerate all embeddings from scratch
            
        Returns:
            Path to the saved embeddings cache file
        """
        if not metadata_dict:
            logger.warning("No metadata provided for embedding generation")
            return None
        
        logger.info(f"Generating {content_type} embeddings for {len(metadata_dict)} items...")
        
        # Extract file paths and text content
        paths = list(metadata_dict.keys())
        texts = []
        
        for file_path, metadata in metadata_dict.items():
            text_content = self._extract_text_content(content_type, file_path, metadata)
            texts.append(text_content)
        
        # Generate cache filename
        cache_filename = self._get_cache_filename(content_type, paths)
        cache_file = os.path.join(self.embeddings_cache_dir, cache_filename)
        
        # Load existing embeddings if available and not forcing regeneration
        existing_data = None if force_regenerate else self._load_existing_embeddings(cache_file)
        
        if existing_data is not None:
            # Merge with existing embeddings
            final_data = self._merge_embeddings_data(existing_data, paths, texts)
        else:
            # Create new embeddings
            final_data = self._create_embeddings_data(paths, texts)
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(final_data, f)
        
        logger.info(f"✅ Embeddings saved to: {cache_filename}")
        return cache_file
    
    def add_single_item(self, content_type: str, file_path: str, metadata: Dict[str, Any]) -> str:
        """
        Add a single item to the embeddings cache.
        
        Args:
            content_type: Type of content (text, audio, video, image)
            file_path: Path to the content file
            metadata: Content metadata
            
        Returns:
            Path to the updated embeddings cache file
        """
        logger.info(f"Adding {content_type} embedding for: {os.path.basename(file_path)}")
        
        # Extract text content for embedding
        text_content = self._extract_text_content(content_type, file_path, metadata)
        
        if not text_content.strip():
            logger.warning(f"No text content extracted for {file_path}")
            return None
        
        # Load model
        self._load_model()
        
        # Find existing cache file for this content type
        cache_files = [f for f in os.listdir(self.embeddings_cache_dir) 
                      if f.startswith(f'{content_type}_embeddings_') and f.endswith('.pkl')]
        
        if cache_files:
            # Use the first (most recent) cache file
            cache_file = os.path.join(self.embeddings_cache_dir, cache_files[0])
            
            # Load existing data
            try:
                with open(cache_file, 'rb') as f:
                    existing_data = pickle.load(f)
                
                # Check if this path already exists
                existing_paths = existing_data.get('paths', [])
                if file_path in existing_paths:
                    logger.info(f"Path already exists in embeddings: {file_path}")
                    return cache_file
                
                # Add new item to existing data
                logger.info(f"Adding to existing cache with {len(existing_paths)} items")
                
                # Generate embedding for new item
                new_embedding = self.sentence_model.encode([text_content], convert_to_tensor=False)
                new_embedding_array = np.array(new_embedding)
                
                # Merge with existing embeddings
                existing_embeddings = existing_data.get('embeddings', np.array([]))
                if existing_embeddings.size > 0:
                    all_embeddings = np.vstack([existing_embeddings, new_embedding_array])
                else:
                    all_embeddings = new_embedding_array
                
                # Update paths
                all_paths = existing_paths + [file_path]
                
                # Recreate BM25 index (simple approach)
                all_texts = []
                for path in existing_paths:
                    all_texts.append(Path(path).stem)  # Use filename as fallback
                all_texts.append(text_content)
                
                tokenized_texts = [text.lower().split() for text in all_texts]
                bm25 = BM25Okapi(tokenized_texts)
                
                # Update data structure
                existing_data.update({
                    'bm25': bm25,
                    'embeddings': all_embeddings,
                    'paths': all_paths,
                    'timestamp': datetime.now().isoformat(),
                    'model_name': self.model_name
                })
                
                # Save updated data
                with open(cache_file, 'wb') as f:
                    pickle.dump(existing_data, f)
                
                logger.info(f"✅ Updated embeddings cache: {len(all_paths)} total items")
                return cache_file
                
            except Exception as e:
                logger.error(f"Failed to update existing cache: {e}")
                # Fall through to create new cache
        
        # Create new cache file
        logger.info("Creating new embeddings cache")
        
        # Generate embedding
        embedding = self.sentence_model.encode([text_content], convert_to_tensor=False)
        embeddings_array = np.array(embedding)
        
        # Create BM25 index
        tokenized_text = [text_content.lower().split()]
        bm25 = BM25Okapi(tokenized_text)
        
        # Create data structure
        data = {
            'bm25': bm25,
            'embeddings': embeddings_array,
            'paths': [file_path],
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name
        }
        
        # Generate cache filename
        paths_hash = hashlib.md5(file_path.encode()).hexdigest()
        cache_filename = f"{content_type}_embeddings_{paths_hash}.pkl"
        cache_file = os.path.join(self.embeddings_cache_dir, cache_filename)
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"✅ Created new embeddings cache: {cache_filename}")
        return cache_file


# Convenience functions for easy integration
def generate_video_embeddings(video_path: str, metadata: Dict[str, Any]) -> str:
    """Generate embeddings for a single video."""
    generator = EmbeddingGenerator()
    return generator.add_single_item("video", video_path, metadata)

def generate_image_embeddings(image_path: str, metadata: Dict[str, Any]) -> str:
    """Generate embeddings for a single image."""
    generator = EmbeddingGenerator()
    return generator.add_single_item("image", image_path, metadata)

def generate_audio_embeddings(audio_path: str, metadata: Dict[str, Any]) -> str:
    """Generate embeddings for a single audio file."""
    generator = EmbeddingGenerator()
    return generator.add_single_item("audio", audio_path, metadata)

def generate_text_embeddings(text_path: str, metadata: Dict[str, Any]) -> str:
    """Generate embeddings for a single text file."""
    generator = EmbeddingGenerator()
    return generator.add_single_item("text", text_path, metadata)

def generate_embeddings_from_metadata_file(content_type: str, metadata_file_path: str, 
                                         force_regenerate: bool = False) -> str:
    """
    Generate embeddings from a metadata JSON file.
    
    Args:
        content_type: Type of content (text, audio, video, image)
        metadata_file_path: Path to metadata JSON file
        force_regenerate: If True, regenerate all embeddings from scratch
        
    Returns:
        Path to the saved embeddings cache file
    """
    if not os.path.exists(metadata_file_path):
        logger.error(f"Metadata file not found: {metadata_file_path}")
        return None
    
    try:
        with open(metadata_file_path, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)
        
        generator = EmbeddingGenerator()
        return generator.generate_embeddings(content_type, metadata_dict, force_regenerate)
    
    except Exception as e:
        logger.error(f"Failed to generate embeddings from metadata file: {e}")
        return None


if __name__ == "__main__":
    """
    CLI interface for embedding generation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for ContentCache")
    parser.add_argument("content_type", choices=["text", "audio", "video", "image"], 
                       help="Type of content to process")
    parser.add_argument("--metadata-file", 
                       help="Path to metadata JSON file (default: uses config paths)")
    parser.add_argument("--force-regenerate", action="store_true",
                       help="Force regeneration of all embeddings from scratch")
    
    args = parser.parse_args()
    
    # Determine metadata file path
    if args.metadata_file:
        metadata_file = args.metadata_file
    else:
        # Use config paths
        from config import (get_video_metadata_path, get_image_metadata_path, 
                          get_audio_metadata_path, get_text_metadata_path)
        
        metadata_paths = {
            "video": get_video_metadata_path(),
            "image": get_image_metadata_path(),
            "audio": get_audio_metadata_path(),
            "text": get_text_metadata_path()
        }
        metadata_file = metadata_paths[args.content_type]
    
    print(f"🚀 Generating {args.content_type} embeddings from {metadata_file}")
    
    result = generate_embeddings_from_metadata_file(
        args.content_type, 
        metadata_file, 
        args.force_regenerate
    )
    
    if result:
        print(f"✅ Embeddings generated successfully: {result}")
    else:
        print("❌ Failed to generate embeddings") 