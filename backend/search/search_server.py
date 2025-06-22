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
import subprocess
import signal
import atexit
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

# Global server instance for cleanup
_server_instance = None

def cleanup_resources():
    """Clean up resources on shutdown"""
    global _server_instance
    if _server_instance:
        try:
            print("🧹 Cleaning up server resources...")
            # Add any specific cleanup here
            _server_instance = None
            print("✅ Resource cleanup completed")
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")

def signal_handler(signum, frame):
    """Handle termination signals gracefully"""
    print(f"📡 Received signal {signum}, shutting down gracefully...")
    cleanup_resources()
    sys.exit(0)

# Register cleanup handlers
atexit.register(cleanup_resources)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def kill_processes_on_port(port):
    """Kill any existing processes running on the specified port."""
    try:
        print(f"🔍 Checking for existing processes on port {port}...")
        
        # Use lsof to find processes using the port
        result = subprocess.run(['lsof', '-ti', f':{port}'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            print(f"🔄 Found {len(pids)} process(es) on port {port}: {', '.join(pids)}")
            
            for pid in pids:
                try:
                    pid_int = int(pid.strip())
                    print(f"🛑 Killing process {pid_int}...")
                    os.kill(pid_int, signal.SIGTERM)
                    print(f"✅ Killed process {pid_int}")
                except (ValueError, ProcessLookupError, PermissionError) as e:
                    print(f"⚠️ Could not kill process {pid}: {e}")
            
            # Wait a moment for processes to terminate
            time.sleep(1)
            print(f"✅ Port {port} cleanup completed")
        else:
            print(f"✅ No existing processes found on port {port}")
            
    except subprocess.TimeoutExpired:
        print(f"⚠️ Timeout while checking port {port}")
    except FileNotFoundError:
        print(f"⚠️ lsof command not found, skipping port cleanup")
    except Exception as e:
        print(f"⚠️ Error during port cleanup: {e}")


class ContentCacheSearchServer:
    def __init__(self, port=5001, auto_sync=True):
        global _server_instance
        self.port = port
        self.auto_sync = auto_sync
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Register this instance for cleanup
        _server_instance = self
        
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
            """Main search endpoint with content type filtering and optional bucketing."""
            try:
                if request.method == 'GET':
                    query = request.args.get('q', '')
                    content_type = request.args.get('type', 'all')
                    top_k = int(request.args.get('top_k', 10))
                    # New filtering parameters
                    date_filter = request.args.get('date_filter', '')
                    location_filter = request.args.get('location_filter', '')
                else:
                    data = request.get_json() or {}
                    query = data.get('query', '')
                    content_type = data.get('type', 'all')
                    top_k = data.get('top_k', 10)
                    # New filtering parameters
                    date_filter = data.get('date_filter', '')
                    location_filter = data.get('location_filter', '')
                
                if not query:
                    return jsonify({'error': 'No query provided'}), 400
                
                # Check if manual filters are provided
                has_manual_filters = bool(date_filter.strip() or location_filter.strip())
                
                # Always try bucketing first to see if OpenAI parsing extracts filters
                results = self._perform_search_with_buckets(query, content_type, top_k, date_filter, location_filter)
                
                # Check if bucketing actually found filters (either manual or AI-parsed)
                has_buckets = 'buckets' in results and len(results['buckets']) > 0
                
                if not has_buckets:
                    # No filters found - fall back to regular search
                    flat_results = self._perform_search(query, content_type, top_k)
                    results = {'results': flat_results}
                
                return jsonify({
                    'query': query,
                    'content_type': content_type,
                    'date_filter': date_filter,
                    'location_filter': location_filter,
                    'has_buckets': has_buckets,
                    **results,  # This will include 'results' or 'buckets' depending on the method
                    'total_results': len(results.get('results', [])) if 'results' in results else sum(len(bucket) for bucket in results.get('buckets', {}).values())
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
                'stats': {
                    'video': len(self.content_metadata['video']),
                    'text': len(self.content_metadata['text']),
                    'audio': len(self.content_metadata['audio']),
                    'image': len(self.content_metadata['image'])
                },
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
            
            # Set device before loading to prevent multiprocessing issues
            import torch
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            
            # Load with specific device and proper configuration
            self.sentence_model = SentenceTransformer(
                'all-MiniLM-L6-v2',
                cache_folder=self.model_cache_dir,
                device=device
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
        
        # Check for discrepancies and auto-sync if needed
        if self.auto_sync:
            self._auto_sync_embeddings_if_needed()
        else:
            self._report_embedding_status()

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

    def _auto_sync_embeddings_if_needed(self):
        """Automatically sync embeddings if discrepancies are detected."""
        print("\n🔍 Checking embedding synchronization...")
        
        discrepancies = []
        for content_type in ['video', 'text', 'audio', 'image']:
            metadata_count = len(self.content_metadata[content_type])
            embedding_count = len(self.content_embeddings[content_type])
            
            if metadata_count != embedding_count:
                discrepancy = {
                    'content_type': content_type,
                    'metadata_count': metadata_count,
                    'embedding_count': embedding_count,
                    'missing': metadata_count - embedding_count
                }
                discrepancies.append(discrepancy)
                print(f"⚠️ {content_type.title()}: {metadata_count} metadata, {embedding_count} embeddings ({abs(discrepancy['missing'])} {'missing' if discrepancy['missing'] > 0 else 'extra'})")
        
        if not discrepancies:
            print("✅ All embeddings are in sync!")
            return
        
        print(f"\n🔄 Found discrepancies in {len(discrepancies)} content types. Auto-syncing...")
        
        # Import embedding generator
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'processing'))
            from embedding_generator import generate_embeddings_from_metadata_file
        except ImportError as e:
            print(f"❌ Failed to import embedding generator: {e}")
            print("⚠️ Continuing with existing embeddings...")
            return
        
        # Sync each content type with discrepancies
        synced_count = 0
        for discrepancy in discrepancies:
            content_type = discrepancy['content_type']
            
            # Skip if no metadata to sync
            if discrepancy['metadata_count'] == 0:
                print(f"⏭️ Skipping {content_type} (no metadata)")
                continue
            
            print(f"🔄 Syncing {content_type} embeddings...")
            
            try:
                # Get metadata file path
                metadata_paths = {
                    'video': get_video_metadata_path(),
                    'text': get_text_metadata_path(),
                    'audio': get_audio_metadata_path(),
                    'image': get_image_metadata_path()
                }
                
                metadata_file = metadata_paths[content_type]
                
                # Generate embeddings from metadata file
                result = generate_embeddings_from_metadata_file(content_type, metadata_file, force_regenerate=False)
                
                if result:
                    # Reload embeddings for this content type
                    self.content_embeddings[content_type] = {}
                    self._load_content_embeddings(content_type)
                    
                    new_embedding_count = len(self.content_embeddings[content_type])
                    print(f"✅ {content_type.title()} synced: {discrepancy['metadata_count']} metadata → {new_embedding_count} embeddings")
                    synced_count += 1
                else:
                    print(f"❌ Failed to sync {content_type} embeddings")
                    
            except Exception as e:
                print(f"❌ Error syncing {content_type}: {e}")
        
        if synced_count > 0:
            print(f"\n✅ Successfully synced {synced_count}/{len(discrepancies)} content types")
            
            # Show final counts
            print("\n📊 Final embedding counts:")
            for content_type in ['video', 'text', 'audio', 'image']:
                metadata_count = len(self.content_metadata[content_type])
                embedding_count = len(self.content_embeddings[content_type])
                status = "✅" if metadata_count == embedding_count else "⚠️"
                print(f"  {status} {content_type.title()}: {metadata_count} metadata, {embedding_count} embeddings")
        else:
            print("\n⚠️ No embeddings were successfully synced")

    def _report_embedding_status(self):
        """Report embedding status without syncing."""
        print("\n📊 Embedding status check (auto-sync disabled):")
        
        total_metadata = 0
        total_embeddings = 0
        discrepancies_found = False
        
        for content_type in ['video', 'text', 'audio', 'image']:
            metadata_count = len(self.content_metadata[content_type])
            embedding_count = len(self.content_embeddings[content_type])
            
            total_metadata += metadata_count
            total_embeddings += embedding_count
            
            if metadata_count != embedding_count:
                discrepancies_found = True
                missing = metadata_count - embedding_count
                print(f"  ⚠️ {content_type.title()}: {metadata_count} metadata, {embedding_count} embeddings ({abs(missing)} {'missing' if missing > 0 else 'extra'})")
            else:
                print(f"  ✅ {content_type.title()}: {metadata_count} items (in sync)")
        
        print(f"\n📈 Total: {total_metadata} metadata items, {total_embeddings} embeddings")
        
        if discrepancies_found:
            print("💡 Tip: Restart with --auto-sync to automatically fix discrepancies")
            print("💡 Or run: python sync_embeddings.py")
        else:
            print("✅ All embeddings are perfectly synced!")

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
            search_radius = parsed.get('search_radius')
            date_filter = parsed.get('date')
            
            print(f"🧠 Query parsed - Core: '{core_query}', Location: {location_text}, Radius: {search_radius}km, Date: {date_filter}")
            
        except Exception as e:
            print(f"⚠️ Query parsing failed, using original query: {e}")
            core_query = query
            location_text = None
            search_radius = None
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
                    'type': ctype,
                    'content_type': ctype,
                    'file_path': file_path,
                    'filename': os.path.basename(file_path),
                    'score': round(similarity, 4),
                    'similarity_score': round(similarity, 4),
                    'search_type': 'semantic'
                }
                
                # Add metadata from content metadata
                if file_path in self.content_metadata[ctype]:
                    metadata = self.content_metadata[ctype][file_path]
                    
                    # Add content-specific metadata
                    if ctype == 'video':
                        summary = metadata.get('video_summary', '')
                        result['summary'] = summary[:200] + '...' if len(summary) > 200 else summary
                        result['content'] = result['summary']
                        result['tags'] = metadata.get('tags', {})
                    elif ctype == 'text':
                        summary = metadata.get('analysis', {}).get('summary', '')
                        result['summary'] = summary
                        result['content'] = summary
                        result['file_type'] = metadata.get('file_type', '')
                    elif ctype in ['image', 'audio']:
                        analysis = metadata.get('analysis', '')
                        result['summary'] = analysis[:200] + '...' if len(analysis) > 200 else analysis
                        result['content'] = result['summary']
                
                semantic_results.append(result)
        
        # Step 2: Apply location filtering if specified
        if location_text:
            semantic_results = self._apply_location_filter(semantic_results, location_text, search_radius)
        
        # Step 3: Apply date filtering with bucketing if specified
        if date_filter:
            semantic_results = self._apply_date_filter_with_buckets(semantic_results, date_filter)
        
        # Return top results
        return semantic_results[:top_k]
    
    def _perform_search_with_buckets(self, query: str, content_type: str = 'all', top_k: int = 20, 
                                   date_filter: str = '', location_filter: str = '') -> Dict[str, Any]:
        """Perform search and return results organized in buckets based on date/location filters."""
        
        # Parse the query using OpenAI to extract semantic components (like _perform_search does)
        parsed_location = None
        search_radius = None
        parsed_date = None
        
        try:
            try:
                from backend.processing.api_client import get_api_client
            except ImportError:
                from api_client import get_api_client
            client = get_api_client()
            parse_result = client.parse_search_query(query)
            
            parsed = parse_result.get('parsed', {})
            parsed_location = parsed.get('location')
            search_radius = parsed.get('search_radius')
            parsed_date = parsed.get('date')
            
            print(f"🧠 Query parsed for bucketing - Location: {parsed_location}, Radius: {search_radius}km, Date: {parsed_date}")
            
        except Exception as e:
            print(f"⚠️ Query parsing failed for bucketing, using manual filters: {e}")
        
        # Use parsed location/date if available, otherwise fall back to manual filters
        effective_location = parsed_location or (location_filter.strip() if location_filter.strip() else None)
        effective_date = parsed_date or (date_filter.strip() if date_filter.strip() else None)
        
        print(f"🔍 Effective filters - Location: '{effective_location}', Date: '{effective_date}'")
        
        # First get all semantic search results (without applying filters)
        all_results = self._perform_search_no_filters(query, content_type, top_k * 3)  # Get more to fill buckets
        
        has_date_filter = bool(effective_date)
        has_location_filter = bool(effective_location)
        
        if not has_date_filter and not has_location_filter:
            # No filters - return empty buckets so main search route can fall back to regular search
            return {'buckets': {}}
        
        # Forward geocode location if we have one
        location_coordinates = None
        if has_location_filter:
            try:
                location_coordinates = location_search.forward_geocode(effective_location)
                if location_coordinates:
                    print(f"📍 Geocoded '{effective_location}' to {location_coordinates}")
                else:
                    print(f"⚠️ Could not geocode '{effective_location}'")
            except Exception as e:
                print(f"⚠️ Geocoding error: {e}")
        
        # Initialize buckets
        buckets = {}
        
        if has_date_filter and has_location_filter:
            # Both filters - create 4 buckets
            buckets = {
                "📅📍 Date & Location Match": [],
                "📅 Date Match Only": [],
                "📍 Location Match Only": [],
                "📄 Other Results": []
            }
            
            for result in all_results:
                date_match = self._check_date_match_parsed(result, effective_date)
                location_match = self._check_location_match_geocoded(result, effective_location, location_coordinates, search_radius)
                
                if date_match and location_match:
                    buckets["📅📍 Date & Location Match"].append(result)
                elif date_match:
                    buckets["📅 Date Match Only"].append(result)
                elif location_match:
                    buckets["📍 Location Match Only"].append(result)
                else:
                    buckets["📄 Other Results"].append(result)
                    
        elif has_date_filter:
            # Only date filter - create 2 buckets
            buckets = {
                "📅 Date Match": [],
                "📄 Other Results": []
            }
            
            for result in all_results:
                date_match = self._check_date_match_parsed(result, effective_date)
                if date_match:
                    buckets["📅 Date Match"].append(result)
                else:
                    buckets["📄 Other Results"].append(result)
                    
        elif has_location_filter:
            # Only location filter - create 2 buckets
            buckets = {
                "📍 Location Match": [],
                "📄 Other Results": []
            }
            
            for result in all_results:
                location_match = self._check_location_match_geocoded(result, effective_location, location_coordinates, search_radius)
                if location_match:
                    buckets["📍 Location Match"].append(result)
                else:
                    buckets["📄 Other Results"].append(result)
        
        # Remove empty buckets and limit results per bucket
        final_buckets = {}
        max_per_bucket = max(8, top_k // len([k for k, v in buckets.items() if v]))  # At least 8 per bucket
        
        for bucket_name, bucket_results in buckets.items():
            if bucket_results:  # Only include non-empty buckets
                # Sort by similarity score and limit
                bucket_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
                final_buckets[bucket_name] = bucket_results[:max_per_bucket]
        
        return {'buckets': final_buckets}
    
    def _perform_search_no_filters(self, query: str, content_type: str = 'all', top_k: int = 20) -> List[Dict]:
        """Perform semantic search without applying any location/date filters."""
        if not self.sentence_model:
            raise RuntimeError("SentenceTransformer model not loaded")
        
        # Parse the query to extract core search terms (without location/date)
        core_query = query  # Default fallback
        try:
            try:
                from backend.processing.api_client import get_api_client
            except ImportError:
                from api_client import get_api_client
            client = get_api_client()
            parse_result = client.parse_search_query(query)
            
            parsed = parse_result.get('parsed', {})
            core_query = parsed.get('search_query', query)
            
            print(f"🔍 Core query for embedding search: '{core_query}' (from original: '{query}')")
            
        except Exception as e:
            print(f"⚠️ Query parsing failed in no_filters, using original query: {e}")
        
        # Just do semantic search without any filtering
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
            
            # Format semantic results
            for type_path, similarity in filtered_similarities:
                # Parse content type and path
                ctype, file_path = type_path.split(':', 1)
                
                result = {
                    'type': ctype,
                    'content_type': ctype,
                    'file_path': file_path,
                    'filename': os.path.basename(file_path),
                    'score': round(similarity, 4),
                    'similarity_score': round(similarity, 4),
                    'search_type': 'semantic'
                }
                
                # Add metadata from content metadata
                if file_path in self.content_metadata[ctype]:
                    metadata = self.content_metadata[ctype][file_path]
                    
                    # Add content-specific metadata
                    if ctype == 'video':
                        summary = metadata.get('video_summary', '')
                        result['summary'] = summary[:200] + '...' if len(summary) > 200 else summary
                        result['content'] = result['summary']
                        result['tags'] = metadata.get('tags', {})
                    elif ctype == 'text':
                        summary = metadata.get('analysis', {}).get('summary', '')
                        result['summary'] = summary
                        result['content'] = summary
                        result['file_type'] = metadata.get('file_type', '')
                    elif ctype in ['image', 'audio']:
                        analysis = metadata.get('analysis', '')
                        result['summary'] = analysis[:200] + '...' if len(analysis) > 200 else analysis
                        result['content'] = result['summary']
                
                semantic_results.append(result)
        
        return semantic_results[:top_k]
    
    def _check_date_match(self, result: Dict, date_filter: str) -> bool:
        """Check if a result matches the date filter."""
        if not date_filter.strip():
            return False
            
        file_path = result['file_path']
        content_type = result['content_type']
        
        if file_path in self.content_metadata[content_type]:
            metadata = self.content_metadata[content_type][file_path]
            
            # Get date from metadata or filename
            date_recorded = metadata.get('metadata', {}).get('date_recorded')
            if date_recorded and date_recorded != 'None':
                return date_filter in date_recorded
            
            # Try extracting date from filename
            import re
            date_match = re.search(r'(\d{4})[_-]?(\d{2})[_-]?(\d{2})', file_path)
            if date_match:
                file_date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
                return date_filter in file_date
        
        return False
    
    def _check_location_match(self, result: Dict, location_filter: str) -> bool:
        """Check if a result matches the location filter (legacy string matching)."""
        if not location_filter.strip():
            return False
            
        file_path = result['file_path']
        content_type = result['content_type']
        
        if file_path in self.content_metadata[content_type]:
            metadata = self.content_metadata[content_type][file_path]
            
            # Check location in metadata
            location_data = metadata.get('metadata', {}).get('location')
            if not location_data or location_data == 'None':
                # For images, check the 'coordinates' field as well
                location_data = metadata.get('coordinates')
            
            # Check if location_filter appears in location text
            if isinstance(location_data, str) and location_data not in ['None', '']:
                return location_filter.lower() in location_data.lower()
            elif isinstance(location_data, dict):
                # Check if it's a place name or coordinates
                location_text = str(location_data)
                return location_filter.lower() in location_text.lower()
        
        return False
    
    def _check_location_match_geocoded(self, result: Dict, location_filter: str, target_coordinates: Optional[tuple], search_radius: Optional[float]) -> bool:
        """Check if a result matches the location filter using forward geocoding and radius."""
        if not location_filter or not target_coordinates:
            return False
            
        target_lat, target_lon = target_coordinates
        radius_km = search_radius if search_radius is not None else 50.0
        
        file_path = result['file_path']
        content_type = result['content_type']
        
        if file_path in self.content_metadata[content_type]:
            metadata = self.content_metadata[content_type][file_path]
            
            # Check location in metadata
            location_data = metadata.get('metadata', {}).get('location')
            if not location_data or location_data == 'None':
                # For images, check the 'coordinates' field as well
                location_data = metadata.get('coordinates')
            
            # Handle multiple location formats
            content_coords = None
            
            if isinstance(location_data, dict) and location_data.get('type') == 'coordinates':
                # Legacy nested format (videos)
                content_coords = (location_data.get('latitude'), location_data.get('longitude'))
            elif isinstance(location_data, dict) and 'latitude' in location_data and 'longitude' in location_data:
                # Image coordinates format: {'latitude': lat, 'longitude': lon}
                content_coords = (location_data.get('latitude'), location_data.get('longitude'))
            elif isinstance(location_data, str) and location_data not in ['None', '']:
                # String format - could be coordinates "lat, lon" or place name
                import re
                coord_pattern = r'([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)'
                match = re.search(coord_pattern, location_data)
                
                if match:
                    # Found coordinate string like "37.7749, -122.4194"
                    try:
                        lat, lon = float(match.group(1)), float(match.group(2))
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            content_coords = (lat, lon)
                        else:
                            print(f"⚠️ Invalid coordinate ranges in string: {lat}, {lon}")
                    except ValueError:
                        pass
                
                # If not coordinates, treat as place name text (fallback to string matching)
                if not content_coords:
                    return location_filter.lower() in location_data.lower()
            
            if content_coords and content_coords[0] is not None and content_coords[1] is not None:
                # Calculate distance
                distance = location_search.calculate_distance(
                    target_lat, target_lon, content_coords[0], content_coords[1]
                )
                
                # Include if within radius
                return distance <= radius_km
        
        return False
    
    def _check_date_match_parsed(self, result: Dict, date_filter) -> bool:
        """Check if a result matches the parsed date filter."""
        if not date_filter:
            return False
        
        file_path = result['file_path']
        content_type = result['content_type']
        
        if file_path in self.content_metadata[content_type]:
            metadata = self.content_metadata[content_type][file_path]
            
            # Get date from metadata
            date_recorded_str = metadata.get('metadata', {}).get('date_recorded')
            if date_recorded_str and date_recorded_str != 'None':
                try:
                    from datetime import datetime
                    date_recorded = datetime.fromisoformat(date_recorded_str.replace('Z', '+00:00'))
                    
                    # Handle different date filter formats
                    if isinstance(date_filter, dict) and 'start' in date_filter and 'end' in date_filter:
                        # Parsed date range from OpenAI
                        start_date = datetime.fromisoformat(date_filter['start'].replace('Z', '+00:00'))
                        end_date = datetime.fromisoformat(date_filter['end'].replace('Z', '+00:00'))
                        return start_date <= date_recorded <= end_date
                    elif isinstance(date_filter, str):
                        # Simple string matching (fallback)
                        return date_filter in date_recorded_str
                    
                except ValueError:
                    pass
        
        return False
    
    def _apply_location_filter(self, results: List[Dict], location_text: str, search_radius: Optional[float] = None) -> List[Dict]:
        """Apply location filtering to search results with dynamic radius."""
        # Use intelligent radius or fallback to 50km
        radius_km = search_radius if search_radius is not None else 50.0
        print(f"🌍 Applying location filter for: '{location_text}' (radius: {radius_km}km)")
        
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
                
                # Check both 'location' (videos) and 'coordinates' (images) fields
                location_data = metadata.get('metadata', {}).get('location')
                if not location_data or location_data == 'None':
                    # For images, check the 'coordinates' field as well
                    location_data = metadata.get('coordinates')
                
                # Handle multiple location formats
                content_coords = None
                
                if isinstance(location_data, dict) and location_data.get('type') == 'coordinates':
                    # Legacy nested format (images still use this in coordinates field)
                    content_coords = (location_data.get('latitude'), location_data.get('longitude'))
                elif isinstance(location_data, dict) and 'latitude' in location_data and 'longitude' in location_data:
                    # Image coordinates format: {'latitude': lat, 'longitude': lon}
                    content_coords = (location_data.get('latitude'), location_data.get('longitude'))
                elif isinstance(location_data, str) and location_data not in ['None', '']:
                    # String format - could be coordinates "lat, lon" or place name
                    import re
                    coord_pattern = r'([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)'
                    match = re.search(coord_pattern, location_data)
                    
                    if match:
                        # Found coordinate string like "37.7749, -122.4194"
                        try:
                            lat, lon = float(match.group(1)), float(match.group(2))
                            if -90 <= lat <= 90 and -180 <= lon <= 180:
                                content_coords = (lat, lon)
                                print(f"📍 Parsed coordinates from string: {lat}, {lon}")
                            else:
                                print(f"⚠️ Invalid coordinate ranges in string: {lat}, {lon}")
                        except ValueError:
                            pass
                    
                    # If not coordinates, treat as place name text
                    if not content_coords:
                        print(f"📍 Found text location: {location_data}")
                    result['location_match'] = 'text_based'
                    result['location_text'] = location_data
                    location_filtered.append(result)
                    continue
                
                if content_coords and content_coords[0] is not None and content_coords[1] is not None:
                    # Calculate distance
                    distance = location_search.calculate_distance(
                        target_lat, target_lon, content_coords[0], content_coords[1]
                    )
                    
                    # Include if within dynamic radius
                    if distance <= radius_km:
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
        
        # Kill any existing processes on the port before starting
        kill_processes_on_port(self.port)
        
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
# Initialize location search
location_search = LocationSearch()

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ContentCache Search Server')
    parser.add_argument('--port', type=int, default=5001, help='Server port (default: 5001)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-auto-sync', action='store_true', help='Disable automatic embedding synchronization on startup')
    
    args = parser.parse_args()
    
    # Auto-sync is enabled by default, disabled if --no-auto-sync is passed
    auto_sync = not args.no_auto_sync
    
    if not auto_sync:
        print("🔄 Auto-sync disabled. Server will start faster but may have embedding discrepancies.")
    
    server = ContentCacheSearchServer(port=args.port, auto_sync=auto_sync)
    server.run(debug=args.debug)

if __name__ == '__main__':
    main() 