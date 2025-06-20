#!/usr/bin/env python3
"""
Search Server for ContentCache
Loads embeddings and metadata for all content types to provide fast semantic search functionality.
"""

import os
import json
import pickle
import time
from typing import Dict, List, Tuple, Any, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sentence_transformers import SentenceTransformer, util
import logging
from pathlib import Path
from config import (get_models_cache_dir, get_embeddings_cache_dir, 
                   get_video_metadata_path, get_text_metadata_path,
                   get_image_metadata_path, get_audio_metadata_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentCacheSearchServer:
    def __init__(self, port=5001):
        self.port = port
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Model and data storage
        self.sentence_model = None
        
        # Separate storage for each content type
        self.content_embeddings = {
            'video': {},
            'text': {},
            'audio': {},
            'image': {}
        }
        self.content_metadata = {
            'video': {},
            'text': {},
            'audio': {},
            'image': {}
        }
        
        self.embeddings_cache_dir = get_embeddings_cache_dir()
        self.model_cache_dir = get_models_cache_dir()
        
        # Ensure directories exist
        Path(self.embeddings_cache_dir).mkdir(exist_ok=True)
        Path(self.model_cache_dir).mkdir(exist_ok=True)
        
        # Setup routes
        self._setup_routes()
        
        print("🚀 Initializing ContentCache Search Server...")
        self._load_models()
        self._load_embeddings_and_metadata()
        print("✅ Search server ready!")

    def _setup_routes(self):
        """Setup Flask routes for the search API."""
        
        @self.app.route('/search', methods=['GET', 'POST'])
        def search():
            """Main search endpoint with content type filtering."""
            try:
                if request.method == 'GET':
                    query = request.args.get('q', '')
                    content_type = request.args.get('type', 'all')
                    top_k = int(request.args.get('top_k', 10))
                else:
                    data = request.get_json() or {}
                    query = data.get('query', '')
                    content_type = data.get('type', 'all')
                    top_k = data.get('top_k', 10)
                
                if not query:
                    return jsonify({'error': 'No query provided'}), 400
                
                results = self._perform_search(query, content_type, top_k)
                return jsonify({
                    'query': query,
                    'content_type': content_type,
                    'results': results,
                    'total_results': len(results)
                })
                
            except Exception as e:
                logger.error(f"Search error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/status', methods=['GET'])
        def status():
            """Server status endpoint."""
            total_embeddings = sum(len(embeddings) for embeddings in self.content_embeddings.values())
            total_metadata = sum(len(metadata) for metadata in self.content_metadata.values())
            
            return jsonify({
                'status': 'running',
                'model_loaded': self.sentence_model is not None,
                'content_stats': {
                    'video': {
                        'metadata': len(self.content_metadata['video']),
                        'embeddings': len(self.content_embeddings['video'])
                    },
                    'text': {
                        'metadata': len(self.content_metadata['text']),
                        'embeddings': len(self.content_embeddings['text'])
                    },
                    'audio': {
                        'metadata': len(self.content_metadata['audio']),
                        'embeddings': len(self.content_embeddings['audio'])
                    },
                    'image': {
                        'metadata': len(self.content_metadata['image']),
                        'embeddings': len(self.content_embeddings['image'])
                    }
                },
                'totals': {
                    'metadata': total_metadata,
                    'embeddings': total_embeddings
                }
            })

        @self.app.route('/content/<content_type>', methods=['GET'])
        def list_content(content_type):
            """List all processed content of a specific type."""
            if content_type not in self.content_metadata:
                return jsonify({'error': f'Invalid content type: {content_type}'}), 400
            
            content_list = []
            for file_path, metadata in self.content_metadata[content_type].items():
                item = {
                    'path': file_path,
                    'filename': os.path.basename(file_path),
                    'type': content_type
                }
                
                # Add type-specific metadata
                if content_type == 'video':
                    item.update({
                        'summary': metadata.get('video_summary', ''),
                        'tags': metadata.get('tags', {}),
                        'duration': metadata.get('metadata', {}).get('duration', 0)
                    })
                elif content_type == 'text':
                    item.update({
                        'summary': metadata.get('analysis', {}).get('summary', ''),
                        'file_type': metadata.get('file_type', ''),
                        'file_size': metadata.get('file_size', 0)
                    })
                elif content_type == 'image':
                    item.update({
                        'analysis': metadata.get('analysis', ''),
                        'summary': metadata.get('summary', '')
                    })
                elif content_type == 'audio':
                    item.update({
                        'analysis': metadata.get('analysis', ''),
                        'summary': metadata.get('summary', '')
                    })
                
                content_list.append(item)
            
            return jsonify({
                'content_type': content_type,
                'items': content_list,
                'total': len(content_list)
            })

        @self.app.route('/content/<content_type>/<path:file_path>', methods=['GET'])
        def get_content_details(content_type, file_path):
            """Get detailed metadata for a specific content item."""
            if content_type not in self.content_metadata:
                return jsonify({'error': f'Invalid content type: {content_type}'}), 400
            
            if file_path in self.content_metadata[content_type]:
                return jsonify(self.content_metadata[content_type][file_path])
            else:
                return jsonify({'error': f'{content_type.title()} not found'}), 404

    def _load_models(self):
        """Load the sentence transformer model."""
        try:
            print("📥 Loading SentenceTransformer model...")
            self.sentence_model = SentenceTransformer(
                'all-MiniLM-L6-v2',
                cache_folder=self.model_cache_dir
            )
            print("✅ SentenceTransformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}")
            raise

    def _load_embeddings_and_metadata(self):
        """Load embeddings and metadata for all content types."""
        print("📊 Loading embeddings and metadata for all content types...")
        
        # Metadata file paths
        metadata_paths = {
            'video': get_video_metadata_path(),
            'text': get_text_metadata_path(),
            'audio': get_audio_metadata_path(),
            'image': get_image_metadata_path()
        }
        
        # Load metadata for each content type
        for content_type, metadata_file in metadata_paths.items():
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        self.content_metadata[content_type] = json.load(f)
                    print(f"✅ Loaded {content_type} metadata: {len(self.content_metadata[content_type])} items")
                except Exception as e:
                    logger.warning(f"Failed to load {content_type} metadata: {e}")
                    self.content_metadata[content_type] = {}
            else:
                print(f"⚠️ No {content_type} metadata file found: {metadata_file}")
                self.content_metadata[content_type] = {}
        
        # Load embeddings for each content type
        for content_type in ['video', 'text', 'audio', 'image']:
            self._load_content_embeddings(content_type)

    def _load_content_embeddings(self, content_type: str):
        """Load embeddings for a specific content type."""
        embedding_files = [f for f in os.listdir(self.embeddings_cache_dir) 
                          if f.startswith(f'{content_type}_embeddings_') and f.endswith('.pkl')]
        
        embeddings_loaded = 0
        for embedding_file in embedding_files:
            try:
                with open(os.path.join(self.embeddings_cache_dir, embedding_file), 'rb') as f:
                    embeddings_data = pickle.load(f)
                    
                # Handle different embedding formats
                if isinstance(embeddings_data, dict):
                    if 'embeddings' in embeddings_data and 'paths' in embeddings_data:
                        # New format with BM25 and SentenceTransformer embeddings
                        embeddings_array = embeddings_data['embeddings']
                        paths = embeddings_data['paths']
                        
                        # Convert to expected format: {file_path: embedding}
                        for i, path in enumerate(paths):
                            if i < len(embeddings_array):
                                self.content_embeddings[content_type][path] = embeddings_array[i]
                        
                        embeddings_loaded += len(paths)
                    else:
                        # Legacy format: direct {file_path: embedding} mapping
                        self.content_embeddings[content_type].update(embeddings_data)
                        embeddings_loaded += len(embeddings_data)
                    
            except Exception as e:
                logger.warning(f"Failed to load {content_type} embeddings from {embedding_file}: {e}")

        if embeddings_loaded > 0:
            print(f"✅ Loaded {content_type} embeddings: {embeddings_loaded} items from {len(embedding_files)} cache files")

    def _perform_search(self, query: str, content_type: str = 'all', top_k: int = 10) -> List[Dict]:
        """Perform semantic search on content with optional type filtering."""
        if not self.sentence_model:
            raise RuntimeError("SentenceTransformer model not loaded")
        
        # Determine which content types to search
        if content_type == 'all':
            search_types = ['video', 'text', 'audio', 'image']
        elif content_type in self.content_embeddings:
            search_types = [content_type]
        else:
            raise ValueError(f"Invalid content type: {content_type}")
        
        # Collect all embeddings to search
        all_embeddings = {}
        for ctype in search_types:
            all_embeddings.update({
                f"{ctype}:{path}": embedding 
                for path, embedding in self.content_embeddings[ctype].items()
            })
        
        if not all_embeddings:
            return []

        # Encode the search query
        query_embedding = self.sentence_model.encode([query])
        
        # Calculate similarities
        similarities = []
        for type_path, embedding in all_embeddings.items():
            if embedding is not None:
                # Handle different embedding formats
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                
                # Calculate cosine similarity
                similarity = util.pytorch_cos_sim(query_embedding, embedding).item()
                similarities.append((type_path, similarity))
        
        # Sort by similarity and get top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        # Format results
        results = []
        for type_path, similarity in top_results:
            # Parse content type and path
            ctype, file_path = type_path.split(':', 1)
            
            result = {
                'content_type': ctype,
                'file_path': file_path,
                'filename': os.path.basename(file_path),
                'similarity_score': round(similarity, 4)
            }
            
            # Add type-specific metadata if available
            if file_path in self.content_metadata[ctype]:
                metadata = self.content_metadata[ctype][file_path]
                
                if ctype == 'video':
                    result.update({
                        'summary': metadata.get('video_summary', ''),
                        'tags': metadata.get('tags', {}),
                        'duration': metadata.get('metadata', {}).get('duration', 0),
                        'location': metadata.get('metadata', {}).get('location', '')
                    })
                elif ctype == 'text':
                    analysis = metadata.get('analysis', {})
                    result.update({
                        'summary': analysis.get('summary', ''),
                        'key_topics': analysis.get('key_topics', []),
                        'file_type': metadata.get('file_type', ''),
                        'content_preview': metadata.get('content_preview', '')[:200] + '...' if metadata.get('content_preview', '') else ''
                    })
                elif ctype == 'image':
                    result.update({
                        'analysis': metadata.get('analysis', ''),
                        'summary': metadata.get('summary', ''),
                        'location': metadata.get('location', '')
                    })
                elif ctype == 'audio':
                    result.update({
                        'analysis': metadata.get('analysis', ''),
                        'summary': metadata.get('summary', '')
                    })
            
            results.append(result)
        
        return results

    def run(self, debug=False):
        """Start the Flask server."""
        total_embeddings = sum(len(embeddings) for embeddings in self.content_embeddings.values())
        total_metadata = sum(len(metadata) for metadata in self.content_metadata.values())
        
        print(f"🌐 Starting search server on http://localhost:{self.port}")
        print(f"📊 Ready to search across all content types:")
        for ctype in ['video', 'text', 'audio', 'image']:
            meta_count = len(self.content_metadata[ctype])
            emb_count = len(self.content_embeddings[ctype])
            print(f"  - {ctype}: {meta_count} metadata, {emb_count} embeddings")
        print(f"📊 Total: {total_metadata} items with metadata, {total_embeddings} with embeddings")
        
        self.app.run(host='0.0.0.0', port=self.port, debug=debug, threaded=True)

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ContentCache Search Server')
    parser.add_argument('--port', type=int, default=5001, help='Server port (default: 5001)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    server = ContentCacheSearchServer(port=args.port)
    server.run(debug=args.debug)

if __name__ == '__main__':
    main() 