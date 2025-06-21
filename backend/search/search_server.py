#!/usr/bin/env python3
"""
Search Server for ContentCache
Loads embeddings and metadata for all content types to provide fast semantic search functionality.
"""

import os
import sys
import json
import pickle
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sentence_transformers import SentenceTransformer, util
import logging
from pathlib import Path

# --- Add backend/processing to Python path ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'processing'))

# Handle both standalone script and module import
try:
    from backend.processing.config import (get_models_cache_dir, get_embeddings_cache_dir, 
                       get_video_metadata_path, get_text_metadata_path,
                       get_image_metadata_path, get_audio_metadata_path)
except ImportError:
from config import (get_models_cache_dir, get_embeddings_cache_dir, 
                   get_video_metadata_path, get_text_metadata_path,
                   get_image_metadata_path, get_audio_metadata_path)

import math
import re

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
        """Perform semantic search with intelligent filtering on results."""
        if not self.sentence_model:
            raise RuntimeError("SentenceTransformer model not loaded")
        
        # Parse the query using OpenAI to extract semantic components
        try:
            try:
                from backend.processing.api_client import get_api_client
            except ImportError:
            from api_client import get_api_client
            client = get_api_client()
            parse_result = client.parse_search_query(query)
            
            parsed = parse_result.get('parsed', {})
            core_query = parsed.get('search_query', query)
            location_text = parsed.get('location')
            date_filter = parsed.get('date')
            
            print(f"🧠 Query parsed - Core: '{core_query}', Location: {location_text}, Date: {date_filter}")
            
        except Exception as e:
            print(f"⚠️ Query parsing failed, using original query: {e}")
            core_query = query
            location_text = None
            date_filter = None
        
        # Step 1: Perform semantic search with threshold
        search_types = ['video', 'text', 'audio', 'image'] if content_type == 'all' else [content_type]
        
        # Collect all embeddings to search
        all_embeddings = {}
        for ctype in search_types:
            all_embeddings.update({
                f"{ctype}:{path}": embedding 
                for path, embedding in self.content_embeddings[ctype].items()
            })
        
        semantic_results = []
        if all_embeddings:
            # Encode the core search query (without location/date terms)
            query_embedding = self.sentence_model.encode([core_query])
            
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
            
            # Filter by similarity threshold (0.25) and sort
            filtered_similarities = [(tp, sim) for tp, sim in similarities if sim >= 0.25]
            filtered_similarities.sort(key=lambda x: x[1], reverse=True)
            
            print(f"🔍 Semantic search: {len(similarities)} total, {len(filtered_similarities)} above 0.25 threshold")
            
            # Format semantic results
            for type_path, similarity in filtered_similarities:
                # Parse content type and path
                ctype, file_path = type_path.split(':', 1)
                
                result = {
                    'content_type': ctype,
                    'file_path': file_path,
                    'filename': os.path.basename(file_path),
                    'similarity_score': round(similarity, 4),
                    'search_type': 'semantic'
                }
                
                # Add metadata from content metadata
                if file_path in self.content_metadata[ctype]:
                    metadata = self.content_metadata[ctype][file_path]
                    
                    # Add content-specific metadata
                    if ctype == 'video':
                        result['summary'] = metadata.get('video_summary', '')[:200] + '...' if len(metadata.get('video_summary', '')) > 200 else metadata.get('video_summary', '')
                        result['tags'] = metadata.get('tags', {})
                    elif ctype == 'text':
                        result['summary'] = metadata.get('analysis', {}).get('summary', '')
                        result['file_type'] = metadata.get('file_type', '')
                    elif ctype in ['image', 'audio']:
                        result['summary'] = metadata.get('analysis', '')[:200] + '...' if len(metadata.get('analysis', '')) > 200 else metadata.get('analysis', '')
                
                semantic_results.append(result)
        
        # Step 2: Apply location filtering if specified
        if location_text:
            semantic_results = self._apply_location_filter(semantic_results, location_text)
        
        # Step 3: Apply date filtering with bucketing if specified
        if date_filter:
            semantic_results = self._apply_date_filter_with_buckets(semantic_results, date_filter)
        
        # Return top results
        return semantic_results[:top_k]
    
    def _apply_location_filter(self, results: List[Dict], location_text: str) -> List[Dict]:
        """Apply location filtering to search results."""
        print(f"🌍 Applying location filter for: '{location_text}'")
        
        # Try to geocode the location
        coordinates = location_search.forward_geocode(location_text)
        
        if not coordinates:
            print(f"⚠️ Could not geocode location: {location_text}")
            return results
        
        target_lat, target_lon = coordinates
        print(f"📍 Target coordinates: {target_lat}, {target_lon}")
        
        location_filtered = []
        for result in results:
            file_path = result['file_path']
            content_type = result['content_type']
            
            if file_path in self.content_metadata[content_type]:
                metadata = self.content_metadata[content_type][file_path]
                location_data = metadata.get('metadata', {}).get('location')
                
                # Handle both new coordinate format and old string format
                content_coords = None
                
                if isinstance(location_data, dict) and location_data.get('type') == 'coordinates':
                    # New coordinate format
                    content_coords = (location_data.get('latitude'), location_data.get('longitude'))
                elif isinstance(location_data, str) and location_data not in ['None', '']:
                    # Old string format - try to extract coordinates or geocode
                    # For now, we'll include all string-based locations as potential matches
                    # TODO: Could implement reverse-geocoding of old string format
                    print(f"📍 Found string location: {location_data}")
                    # Include in results but without distance calculation
                    result['location_match'] = 'text_based'
                    result['location_text'] = location_data
                    location_filtered.append(result)
                    continue
                
                if content_coords and content_coords[0] is not None and content_coords[1] is not None:
                    # Calculate distance
                    distance = location_search.calculate_distance(
                        target_lat, target_lon, content_coords[0], content_coords[1]
                    )
                    
                    # Include if within 50km
                    if distance <= 50:
                        result['location_match'] = 'coordinate_based'
                        result['distance_km'] = round(distance, 2)
                        result['location'] = {
                            'coordinates': [content_coords[0], content_coords[1]],
                            'distance_from_query': f"{round(distance, 2)} km"
                        }
                        location_filtered.append(result)
        
        print(f"🗺️ Location filter: {len(results)} → {len(location_filtered)} results")
        
        # Sort by distance if we have coordinate-based matches
        coordinate_matches = [r for r in location_filtered if r.get('location_match') == 'coordinate_based']
        text_matches = [r for r in location_filtered if r.get('location_match') == 'text_based']
        
        # Sort coordinate matches by distance, keep text matches by similarity
        coordinate_matches.sort(key=lambda x: x.get('distance_km', float('inf')))
        
        return coordinate_matches + text_matches
    
    def _apply_date_filter_with_buckets(self, results: List[Dict], date_filter: Dict) -> List[Dict]:
        """Apply date filtering with bucketing: in-range results first, then out-of-range."""
        if not date_filter or not date_filter.get('start') or not date_filter.get('end'):
            return results
        
        try:
            from datetime import datetime
            start_date = datetime.fromisoformat(date_filter['start'].replace('Z', '+00:00'))
            end_date = datetime.fromisoformat(date_filter['end'].replace('Z', '+00:00'))
            
            print(f"📅 Applying date filter: {start_date.date()} to {end_date.date()}")
            
            in_range_results = []
            out_of_range_results = []
            
            for result in results:
                file_path = result['file_path']
                content_type = result['content_type']
                
                if file_path in self.content_metadata[content_type]:
                    metadata = self.content_metadata[content_type][file_path]
                    
                    # Get date from metadata
                    date_recorded_str = metadata.get('metadata', {}).get('date_recorded')
                    if date_recorded_str and date_recorded_str != 'None':
                        try:
                            date_recorded = datetime.fromisoformat(date_recorded_str.replace('Z', '+00:00'))
                            
                            # Add date info to result
                            result['date_recorded'] = date_recorded.date().isoformat()
                            
                            # Check if date falls within filter range
                            if start_date <= date_recorded <= end_date:
                                result['date_match'] = 'in_range'
                                in_range_results.append(result)
                            else:
                                result['date_match'] = 'out_of_range'
                                out_of_range_results.append(result)
                        except ValueError:
                            # If date parsing fails, put in out-of-range bucket
                            result['date_match'] = 'parse_error'
                            out_of_range_results.append(result)
                    else:
                        # If no date metadata, put in out-of-range bucket
                        result['date_match'] = 'no_date'
                        out_of_range_results.append(result)
                else:
                    # If no metadata, put in out-of-range bucket
                    result['date_match'] = 'no_metadata'
                    out_of_range_results.append(result)
            
            print(f"📅 Date bucketing: {len(in_range_results)} in range, {len(out_of_range_results)} out of range")
            
            # Return in-range results first, then out-of-range
            return in_range_results + out_of_range_results
            
        except Exception as e:
            print(f"⚠️ Date filtering error: {e}")
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

class LocationSearch:
    """Handle location-based search functionality"""
    
    def __init__(self):
        self.location_patterns = [
            # University patterns
            r'\b([a-zA-Z\s]+(?:university|college|school|institute|academy))\b',
            # City, State patterns
            r'\b([a-zA-Z\s]+),\s*([a-zA-Z\s]+)\b',
            # Landmark patterns
            r'\b(golden gate bridge|statue of liberty|eiffel tower|times square|central park)\b',
            # General location words
            r'\b(downtown|uptown|city center|campus|park|beach|mountain|lake|river)\s+([a-zA-Z\s]+)\b',
            r'\b([a-zA-Z\s]+)\s+(downtown|uptown|city center|campus|park|beach|mountain|lake|river)\b',
        ]
    
    def detect_location_query(self, query: str) -> Optional[str]:
        """Detect if query contains location references"""
        query_lower = query.lower()
        
        # First check for explicit location keywords that indicate spatial queries
        explicit_location_indicators = [
            'near', 'at', 'in', 'around', 'close to', 'nearby', 'from'
        ]
        
        has_location_indicator = any(indicator in query_lower for indicator in explicit_location_indicators)
        
        # Check for location patterns only if we have location indicators OR specific landmarks
        for pattern in self.location_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                matched_text = match.group(0).strip()
                
                # For university/landmark patterns, always consider them location queries
                if any(word in matched_text for word in ['university', 'college', 'school', 'institute', 'academy', 'bridge', 'park']):
                    return matched_text
                
                # For other patterns, only if we have explicit location indicators
                if has_location_indicator:
                    return matched_text
        
        # Check for location keywords with following text (only if explicit indicators present)
        if has_location_indicator:
            location_keywords = [
                'near', 'at', 'in', 'around', 'close to', 'nearby', 'from'
            ]
            
            for keyword in location_keywords:
                if keyword in query_lower:
                    # Extract potential location after the keyword
                    parts = query_lower.split(keyword, 1)
                    if len(parts) > 1:
                        potential_location = parts[1].strip()
                        # Clean up and return meaningful location text (must be reasonable length)
                        if len(potential_location) > 2 and len(potential_location) < 50:
                            return f"{keyword} {potential_location}".strip()
        
        return None
    
    def forward_geocode(self, location_text: str) -> Optional[Tuple[float, float]]:
        """Convert location text to coordinates using Google Maps API"""
        try:
            try:
                from backend.processing.api_client import get_api_client
            except ImportError:
            from api_client import get_api_client
            client = get_api_client()
            response = client.google_forward_geocode(location_text)
            
            if response and response.get('status') == 'OK' and response.get('results'):
                result = response['results'][0]
                geometry = result.get('geometry', {})
                location = geometry.get('location', {})
                
                lat = location.get('lat')
                lng = location.get('lng')
                
                if lat is not None and lng is not None:
                    return (float(lat), float(lng))
            
        except Exception as e:
            print(f"⚠️ Forward geocoding failed: {e}")
        
        return None
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in kilometers using Haversine formula"""
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371
        return c * r
    
    def find_nearby_content(self, target_lat: float, target_lon: float, all_metadata: Dict, 
                          max_distance_km: float = 50) -> List[Tuple[str, float]]:
        """Find content within max_distance_km of target coordinates"""
        nearby_content = []
        
        for file_path, metadata in all_metadata.items():
            location_data = metadata.get('metadata', {}).get('location')
            
            if location_data and isinstance(location_data, dict) and location_data.get('type') == 'coordinates':
                content_lat = location_data.get('latitude')
                content_lon = location_data.get('longitude')
                
                if content_lat is not None and content_lon is not None:
                    distance = self.calculate_distance(target_lat, target_lon, content_lat, content_lon)
                    
                    if distance <= max_distance_km:
                        nearby_content.append((file_path, distance))
        
        # Sort by distance (closest first)
        nearby_content.sort(key=lambda x: x[1])
        return nearby_content

# Initialize location search
location_search = LocationSearch()

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