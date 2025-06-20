#!/usr/bin/env python3
"""
Search Server for ContentCache
Loads video embeddings and metadata to provide fast semantic search functionality.
"""

import os
import json
import pickle
import time
from typing import Dict, List, Tuple, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sentence_transformers import SentenceTransformer, util
import logging
from pathlib import Path
from config import get_models_cache_dir, get_embeddings_cache_dir

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
        self.video_embeddings = {}
        self.video_metadata = {}
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
            """Main search endpoint."""
            try:
                if request.method == 'GET':
                    query = request.args.get('q', '')
                else:
                    data = request.get_json()
                    query = data.get('query', '') if data else ''
                
                if not query:
                    return jsonify({'error': 'No query provided'}), 400
                
                results = self._perform_search(query)
                return jsonify({
                    'query': query,
                    'results': results,
                    'total_results': len(results)
                })
                
            except Exception as e:
                logger.error(f"Search error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/status', methods=['GET'])
        def status():
            """Server status endpoint."""
            return jsonify({
                'status': 'running',
                'loaded_videos': len(self.video_metadata),
                'embeddings_loaded': len(self.video_embeddings),
                'model_loaded': self.sentence_model is not None
            })

        @self.app.route('/videos', methods=['GET'])
        def list_videos():
            """List all processed videos."""
            videos = []
            for video_path, metadata in self.video_metadata.items():
                videos.append({
                    'path': video_path,
                    'filename': os.path.basename(video_path),
                    'tags': metadata.get('tags', []),
                    'summary': metadata.get('summary', ''),
                    'duration': metadata.get('duration', 0)
                })
            return jsonify({'videos': videos})

        @self.app.route('/video/<path:video_path>', methods=['GET'])
        def get_video_details(video_path):
            """Get detailed metadata for a specific video."""
            if video_path in self.video_metadata:
                return jsonify(self.video_metadata[video_path])
            else:
                return jsonify({'error': 'Video not found'}), 404

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
        """Load video embeddings and metadata from cache files."""
        print("📊 Loading video embeddings and metadata...")
        
        # Load video metadata
        metadata_file = "video_metadata.json"
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    self.video_metadata = json.load(f)
                print(f"✅ Loaded metadata for {len(self.video_metadata)} videos")
            except Exception as e:
                logger.warning(f"Failed to load video metadata: {e}")
                self.video_metadata = {}
        else:
            print("⚠️ No video metadata file found")
            self.video_metadata = {}

        # Load video embeddings from cache
        embedding_files = [f for f in os.listdir(self.embeddings_cache_dir) 
                          if f.startswith('video_embeddings_') and f.endswith('.pkl')]
        
        embeddings_loaded = 0
        for embedding_file in embedding_files:
            try:
                with open(os.path.join(self.embeddings_cache_dir, embedding_file), 'rb') as f:
                    embeddings_data = pickle.load(f)
                    
                # Merge embeddings (assuming format: {video_path: embedding})
                if isinstance(embeddings_data, dict):
                    self.video_embeddings.update(embeddings_data)
                    embeddings_loaded += len(embeddings_data)
                    
            except Exception as e:
                logger.warning(f"Failed to load embeddings from {embedding_file}: {e}")

        print(f"✅ Loaded embeddings for {embeddings_loaded} videos from {len(embedding_files)} cache files")

    def _perform_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Perform semantic search on video content."""
        if not self.sentence_model:
            raise RuntimeError("SentenceTransformer model not loaded")
        
        if not self.video_embeddings:
            return []

        # Encode the search query
        query_embedding = self.sentence_model.encode([query])
        
        # Calculate similarities
        similarities = []
        for video_path, video_embedding in self.video_embeddings.items():
            if video_embedding is not None:
                # Handle different embedding formats
                if isinstance(video_embedding, list):
                    video_embedding = np.array(video_embedding)
                
                # Calculate cosine similarity
                similarity = util.pytorch_cos_sim(query_embedding, video_embedding).item()
                similarities.append((video_path, similarity))
        
        # Sort by similarity and get top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        # Format results
        results = []
        for video_path, similarity in top_results:
            result = {
                'video_path': video_path,
                'filename': os.path.basename(video_path),
                'similarity_score': round(similarity, 4)
            }
            
            # Add metadata if available
            if video_path in self.video_metadata:
                metadata = self.video_metadata[video_path]
                result.update({
                    'summary': metadata.get('summary', ''),
                    'tags': metadata.get('tags', []),
                    'duration': metadata.get('duration', 0),
                    'location': metadata.get('location', ''),
                    'transcript_highlights': metadata.get('transcript_highlights', [])
                })
            
            results.append(result)
        
        return results

    def run(self, debug=False):
        """Start the Flask server."""
        print(f"🌐 Starting search server on http://localhost:{self.port}")
        print(f"📊 Ready to search {len(self.video_metadata)} videos")
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