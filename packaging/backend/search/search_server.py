#!/usr/bin/env python3
"""
Search Server for ContentCache
Loads embeddings and metadata for all content types to provide fast semantic search functionality.

CRITICAL: Environment isolation must happen BEFORE any imports
to prevent conflicts with system NumPy 2.x and other incompatible packages.
"""

# STEP 1: Setup isolated environment BEFORE any imports
import sys
import os
from pathlib import Path

def setup_isolated_environment():
    """Setup isolated Python environment to avoid system package conflicts."""
    print("🔧 [Search Server] Setting up isolated Python environment...")
    
    # Get the app cache directory
    platform_name = os.uname().sysname if hasattr(os, 'uname') else 'Unknown'
    if platform_name == 'Darwin':  # macOS
        app_cache = os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', 'silk.ai')
    elif platform_name == 'Windows' or os.name == 'nt':
        app_cache = os.path.join(os.environ.get('APPDATA', os.path.expanduser('~')), 'silk.ai')
    else:  # Linux and others
        app_cache = os.path.join(os.path.expanduser('~'), '.config', 'silk.ai')
    
    user_base = os.path.join(app_cache, 'python_packages')
    user_site_packages = os.path.join(user_base, 'lib', 'python', 'site-packages')
    
    print(f"📦 [Search Server] Using isolated packages from: {user_site_packages}")
    
    # Add our isolated packages to the FRONT of sys.path (highest priority)
    if user_site_packages not in sys.path:
        sys.path.insert(0, user_site_packages)
        print(f"✅ [Search Server] Added {user_site_packages} to sys.path[0]")
    
    # Also add to PYTHONPATH for child processes
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if user_site_packages not in current_pythonpath:
        new_pythonpath = user_site_packages + (os.pathsep + current_pythonpath if current_pythonpath else '')
        os.environ['PYTHONPATH'] = new_pythonpath
        print(f"✅ [Search Server] Updated PYTHONPATH")
    
    # Set user base for pip installations
    os.environ['PYTHONUSERBASE'] = user_base
    os.environ['PYTHONNOUSERSITE'] = '0'  # Allow user site packages
    
    print(f"🔧 [Search Server] Python path priority: {sys.path[:3]}...")
    
    # Test if we can import numpy from the correct location
    try:
        import numpy as np
        print(f"✅ [Search Server] Using NumPy {np.__version__} from: {np.__file__}")
        
        # Verify it's NumPy 1.x
        major_version = int(np.__version__.split('.')[0])
        if major_version >= 2:
            print(f"⚠️ [Search Server] WARNING: Found NumPy {np.__version__} (2.x) - may cause OpenCV issues")
        else:
            print(f"✅ [Search Server] NumPy {np.__version__} (1.x) - compatible with sentence-transformers")
            
    except ImportError as e:
        print(f"❌ [Search Server] Could not import NumPy: {e}")
        print(f"💡 [Search Server] Dependencies may need to be installed to: {user_site_packages}")

# CRITICAL: Run environment setup before any other imports
try:
    setup_isolated_environment()
except Exception as e:
    print(f"⚠️ [Search Server] Isolated environment setup failed: {e}")
    print("🔄 [Search Server] Falling back to system packages...")
    # Continue with system packages - no sys.path modification

# STEP 2: Now safe to import other modules
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

# Critical imports with error handling
try:
    from sentence_transformers import SentenceTransformer, util
    print("✅ [Search Server] Successfully imported sentence_transformers")
except ImportError as e:
    print(f"❌ [Search Server] Failed to import sentence_transformers: {e}")
    print("💡 [Search Server] This may indicate NumPy/huggingface_hub compatibility issues")
    # Let the error propagate - search server cannot function without this
    raise

import logging
from rank_bm25 import BM25Okapi
import re

# Import enhanced tokenizer
try:
    from enhanced_tokenizer import get_enhanced_tokenizer
    ENHANCED_TOKENIZER_AVAILABLE = True
except ImportError:
    try:
        # Try relative import for packaged environment
        from .enhanced_tokenizer import get_enhanced_tokenizer
        ENHANCED_TOKENIZER_AVAILABLE = True
    except ImportError:
        try:
            # Try absolute path import
            import sys
            import os
            current_dir = os.path.dirname(__file__)
            tokenizer_path = os.path.join(current_dir, 'enhanced_tokenizer.py')
            if os.path.exists(tokenizer_path):
                sys.path.append(current_dir)
                from enhanced_tokenizer import get_enhanced_tokenizer
                ENHANCED_TOKENIZER_AVAILABLE = True
            else:
                ENHANCED_TOKENIZER_AVAILABLE = False
        except ImportError:
            ENHANCED_TOKENIZER_AVAILABLE = False

# --- Add backend/processing to Python path ---
# Use current working directory since we set cwd to Resources in main.js
current_dir = os.getcwd()
print(f"💻 [Search Server] Working directory: {current_dir}")

# Add backend/processing to path
backend_processing_dir = os.path.join(current_dir, 'backend', 'processing')
if os.path.exists(backend_processing_dir):
    sys.path.append(backend_processing_dir)
    print(f"📁 [Search Server] Added to path: {backend_processing_dir}")
else:
    # Fallback for development
    fallback_dir = os.path.join(os.path.dirname(__file__), '..', 'processing')
    sys.path.append(fallback_dir)
    print(f"📁 [Search Server] Fallback path: {fallback_dir}")

print(f"🐍 [Search Server] Python path: {sys.path[:3]}...")

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
    """Kill any existing processes running on the specified port (except this process)."""
    try:
        print(f"🔍 Checking for existing processes on port {port}...")
        
        # Use lsof to find processes using the port
        result = subprocess.run(['lsof', '-ti', f':{port}'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            current_pid = os.getpid()
            print(f"🔄 Found {len(pids)} process(es) on port {port}: {', '.join(pids)}")
            print(f"🔍 Current process PID: {current_pid}")
            
            killed_any = False
            for pid in pids:
                try:
                    pid_int = int(pid.strip())
                    
                    # Don't kill our own process!
                    if pid_int == current_pid:
                        print(f"⚠️ Skipping current process {pid_int} (this search server)")
                        continue
                    
                    print(f"🛑 Killing process {pid_int}...")
                    os.kill(pid_int, signal.SIGTERM)
                    print(f"✅ Killed process {pid_int}")
                    killed_any = True
                except (ValueError, ProcessLookupError, PermissionError) as e:
                    print(f"⚠️ Could not kill process {pid}: {e}")
            
            if killed_any:
                # Wait a moment for processes to terminate
                time.sleep(1)
                print(f"✅ Port {port} cleanup completed")
            else:
                print(f"✅ No other processes to kill on port {port}")
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
        
        # BM25 models for keyword search
        self.bm25_models = {
            'video': None,
            'text': None,
            'audio': None,
            'image': None
        }
        self.bm25_documents = {
            'video': [],
            'text': [],
            'audio': [],
            'image': []
        }
        self.bm25_file_paths = {
            'video': [],
            'text': [],
            'audio': [],
            'image': []
        }
        
        # Search result cache for ultra-fast repeated queries
        self.search_cache = {}
        self.max_cache_size = 100
        
        # Enhanced tokenizer for better BM25 search
        self.tokenizer = None
        self.tokenizer_capabilities = {}
        
        self.embeddings_cache_dir = get_embeddings_cache_dir()
        self.model_cache_dir = get_models_cache_dir()
        
        # Ensure directories exist
        Path(self.embeddings_cache_dir).mkdir(exist_ok=True)
        Path(self.model_cache_dir).mkdir(exist_ok=True)
        
        # Setup routes
        self._setup_routes()
        
        print("🚀 Initializing ContentCache Search Server...")
        self._initialize_tokenizer()
        self._load_models()
        self._load_embeddings_and_metadata()
        self._build_bm25_indexes()
        print("✅ Search server ready!")

    def _initialize_tokenizer(self):
        """Initialize the enhanced tokenizer for better search quality."""
        print("🔧 Initializing enhanced tokenizer...")
        
        if ENHANCED_TOKENIZER_AVAILABLE:
            try:
                self.tokenizer = get_enhanced_tokenizer()
                self.tokenizer_capabilities = self.tokenizer.get_capabilities()
                
                # Log capabilities
                capabilities_status = []
                for feature, available in self.tokenizer_capabilities.items():
                    status = "✅" if available else "⚠️"
                    capabilities_status.append(f"{status} {feature.replace('_', ' ').title()}")
                
                print("📊 Tokenizer capabilities:")
                for status in capabilities_status:
                    print(f"  {status}")
                
                print("✅ Enhanced tokenizer ready!")
                
            except Exception as e:
                print(f"⚠️ Enhanced tokenizer initialization failed: {e}")
                print("🔄 Falling back to basic tokenization")
                self.tokenizer = None
        else:
            print("⚠️ Enhanced tokenizer not available, using basic regex tokenization")
            self.tokenizer = None

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
                    search_response = self._perform_search(query, content_type, top_k)
                    if isinstance(search_response, dict) and 'results' in search_response:
                        results = search_response  # Enhanced search returns dict with metadata
                    else:
                        results = {'results': search_response}  # Fallback for basic search
                
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

        @self.app.route('/health', methods=['GET'])
        def health():
            """Health check endpoint for the search server."""
            return jsonify({
                'status': 'healthy',
                'server': 'contentcache-search',
                'timestamp': time.time()
            })

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

        @self.app.route('/suggest', methods=['GET', 'POST'])
        def suggest_query():
            """Get spell-corrected suggestions for a search query."""
            try:
                if request.method == 'GET':
                    query = request.args.get('q', '')
                else:
                    data = request.get_json() or {}
                    query = data.get('query', '')
                
                if not query:
                    return jsonify({'error': 'No query provided'}), 400
                
                suggestions = []
                corrections = {}
                
                if self.tokenizer:
                    # Get query suggestions
                    suggestions = self.tokenizer.get_query_suggestions(query)
                    
                    # Get detailed corrections
                    _, corrections = self.tokenizer.tokenize_and_process(
                        query, 
                        apply_spell_check=True,
                        apply_stemming=False,  # Don't stem for suggestions
                        remove_stopwords=False
                    )
                else:
                    suggestions = [query]
                
                return jsonify({
                    'original_query': query,
                    'suggestions': suggestions,
                    'corrections': corrections,
                    'tokenizer_available': self.tokenizer is not None,
                    'capabilities': self.tokenizer_capabilities
                })
                
            except Exception as e:
                logger.error(f"Query suggestion error: {e}")
                return jsonify({
                    'error': f'Failed to generate suggestions: {str(e)}',
                    'original_query': query,
                    'suggestions': [query]
                }), 500

        @self.app.route('/refresh', methods=['POST'])
        def refresh_embeddings():
            """Manually refresh embeddings and metadata from disk."""
            try:
                print("🔄 Manual refresh requested...")
                
                # Clear current data
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
                
                # Clear BM25 indexes
                self.bm25_models = {
                    'video': None,
                    'text': None,
                    'audio': None,
                    'image': None
                }
                self.bm25_documents = {
                    'video': [],
                    'text': [],
                    'audio': [],
                    'image': []
                }
                self.bm25_file_paths = {
                    'video': [],
                    'text': [],
                    'audio': [],
                    'image': []
                }
                
                # Reload everything
                self._load_embeddings_and_metadata()
                self._build_bm25_indexes()  # Rebuild BM25 indexes with new data
                
                # Get new counts
                total_metadata = sum(len(metadata) for metadata in self.content_metadata.values())
                total_embeddings = sum(len(embeddings) for embeddings in self.content_embeddings.values())
                
                print("✅ Manual refresh complete!")
                
                return jsonify({
                    'status': 'success',
                    'message': 'Embeddings and metadata refreshed successfully',
                    'stats': {
                        'video': len(self.content_metadata['video']),
                        'text': len(self.content_metadata['text']),
                        'audio': len(self.content_metadata['audio']),
                        'image': len(self.content_metadata['image'])
                    },
                    'totals': {
                        'metadata': total_metadata,
                        'embeddings': total_embeddings
                    }
                })
                
            except Exception as e:
                logger.error(f"Refresh error: {e}")
                return jsonify({
                    'status': 'error',
                    'message': f'Failed to refresh embeddings: {str(e)}'
                }), 500

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

    def _build_bm25_indexes(self):
        """Build BM25 indexes for keyword search from metadata."""
        print("🔧 Building BM25 indexes for keyword search...")
        
        for content_type in ['video', 'text', 'audio', 'image']:
            documents = []
            file_paths = []
            
            print(f"📋 Processing {content_type} content for BM25 indexing...")
            processed_items = 0
            
            for file_path, metadata in self.content_metadata[content_type].items():
                try:
                    # Extract searchable text from metadata
                    text_parts = []
                    
                    if content_type == 'video':
                        # COMPREHENSIVE video indexing
                        # 1. Video summary
                        summary = metadata.get('video_summary', '')
                        if summary and summary != "Video analysis unavailable due to API server error.":
                            text_parts.append(summary)
                        
                        # 2. All tag categories
                        tags = metadata.get('tags', {})
                        for tag_category, tag_list in tags.items():
                            if isinstance(tag_list, list):
                                text_parts.extend([tag for tag in tag_list if tag])  # Filter empty strings
                            elif isinstance(tag_list, str) and tag_list:
                                text_parts.append(tag_list)
                        
                        # 3. Frame captions (if available)
                        frame_captions = metadata.get('frame_captions', [])
                        if isinstance(frame_captions, list):
                            text_parts.extend(frame_captions)
                        elif isinstance(frame_captions, str):
                            text_parts.append(frame_captions)
                        
                        # 4. Audio summary/transcript
                        audio_summary = metadata.get('audio_summary', '')
                        if audio_summary:
                            if isinstance(audio_summary, dict):
                                # Extract text from audio analysis dict
                                audio_text = audio_summary.get('text', '') or audio_summary.get('transcript', '') or audio_summary.get('summary', '')
                                if audio_text:
                                    text_parts.append(audio_text)
                            elif isinstance(audio_summary, str):
                                text_parts.append(audio_summary)
                        
                        # 5. OCR text from frames (prominent text)
                        text_data = metadata.get('text_data', {})
                        if text_data:
                            prominent_text = text_data.get('prominent_text', [])
                            if isinstance(prominent_text, list):
                                text_parts.extend(prominent_text)
                            elif isinstance(prominent_text, str):
                                text_parts.append(prominent_text)
                        
                        # 6. Enhanced vision analysis
                        vision_analysis = metadata.get('vision_analysis', '')
                        if vision_analysis:
                            text_parts.append(vision_analysis)
                        
                        # 7. Location information
                        location_info = metadata.get('metadata', {}).get('location', '')
                        if location_info and location_info.lower() not in ['none', 'null', '']:
                            text_parts.append(location_info)
                        
                        # 8. Any additional description metadata
                        included_description = metadata.get('metadata', {}).get('included_description', '')
                        if included_description and included_description.lower() not in ['none', 'null', '']:
                            text_parts.append(included_description)
                    
                    elif content_type == 'text':
                        # ENHANCED text indexing - include actual document content!
                        analysis = metadata.get('analysis', {})
                        summary = analysis.get('summary', '')
                        if summary:
                            text_parts.append(summary)
                        
                        # Include key topics extracted by AI
                        key_topics = analysis.get('key_topics', [])
                        if isinstance(key_topics, list):
                            text_parts.extend(key_topics)
                        
                        # Include actual document content preview (HUGE for search!)
                        content_preview = metadata.get('content_preview', '')
                        if content_preview:
                            text_parts.append(content_preview)
                        
                        # Legacy content field
                        content = metadata.get('content', '')
                        if content:
                            text_parts.append(content)
                    
                    elif content_type == 'image':
                        # COMPREHENSIVE image indexing
                        # 1. Image summary
                        image_summary = metadata.get('image_summary', '')
                        if image_summary:
                            text_parts.append(image_summary)
                        
                        # 2. All tag categories
                        tags = metadata.get('tags', {})
                        for tag_category, tag_list in tags.items():
                            if isinstance(tag_list, list):
                                text_parts.extend([tag for tag in tag_list if tag])  # Filter empty strings
                            elif isinstance(tag_list, str) and tag_list:
                                text_parts.append(tag_list)
                        
                        # 3. OCR text (if available)
                        ocr_text = metadata.get('ocr_text', '')
                        if ocr_text:
                            text_parts.append(ocr_text)
                        
                        # 4. Location information
                        location_info = metadata.get('metadata', {}).get('location', '')
                        if location_info and location_info.lower() not in ['none', 'null', '']:
                            text_parts.append(location_info)
                        
                        # 5. Included description
                        included_description = metadata.get('metadata', {}).get('included_description', '')
                        if included_description and included_description.lower() not in ['none', 'null', '']:
                            text_parts.append(included_description)
                        
                        # Also check for analysis field (backward compatibility)
                        analysis = metadata.get('analysis', '')
                        if analysis:
                            text_parts.append(analysis)
                            
                    elif content_type == 'audio':
                        # For audio, use analysis content
                        analysis = metadata.get('analysis', '')
                        if analysis:
                            text_parts.append(analysis)
                        
                        # Also check for transcript
                        transcript = metadata.get('transcript', '')
                        if transcript:
                            text_parts.append(transcript)
                    
                    # Add filename as searchable text
                    filename = os.path.basename(file_path)
                    text_parts.append(filename)
                    
                    # Combine all text and tokenize
                    full_text = ' '.join(filter(None, text_parts))  # Filter out empty strings
                    if full_text.strip():
                        # Use enhanced tokenizer if available, fallback to basic regex
                        if self.tokenizer:
                            try:
                                tokens, corrections = self.tokenizer.tokenize_and_process(
                                    full_text,
                                    apply_spell_check=False,  # Don't spell-check content, only queries
                                    apply_stemming=True,
                                    remove_stopwords=True,
                                    min_length=2
                                )
                                if corrections:
                                    print(f"🔧 Applied corrections during indexing: {corrections}")
                            except Exception as e:
                                print(f"⚠️ Enhanced tokenizer failed for {file_path}: {e}")
                                # Fallback to basic tokenization
                                tokens = re.findall(r'\b\w+\b', full_text.lower())
                        else:
                            # Fallback to basic tokenization
                            tokens = re.findall(r'\b\w+\b', full_text.lower())
                        
                        if tokens:  # Only add if we got tokens
                            documents.append(tokens)
                            file_paths.append(file_path)
                            processed_items += 1
                            
                            # Debug logging for specific terms
                            if any(term in tokens for term in ['passport', 'identification']):
                                print(f"🔍 Found search term in {content_type}: {os.path.basename(file_path)} - tokens: {[t for t in tokens if t in ['passport', 'identification']]}")
                    
                except Exception as e:
                    print(f"⚠️ Error processing {file_path} for BM25 indexing: {e}")
                    continue
            
            print(f"📊 {content_type.title()}: processed {processed_items} items from {len(self.content_metadata[content_type])} total")
            
            if documents:
                try:
                    self.bm25_models[content_type] = BM25Okapi(documents)
                    self.bm25_documents[content_type] = documents
                    self.bm25_file_paths[content_type] = file_paths
                    print(f"✅ Built BM25 index for {content_type}: {len(documents)} documents")
                except Exception as e:
                    print(f"⚠️ Failed to build BM25 index for {content_type}: {e}")
            else:
                print(f"⚠️ No documents found for {content_type} BM25 index")
        
        print("✅ BM25 indexes ready!")
        
        # Test search for common terms
        test_terms = ['passport', 'identification', 'document']
        print("🧪 Testing BM25 indexes with common search terms...")
        for term in test_terms:
            total_matches = 0
            for content_type in ['video', 'text', 'audio', 'image']:
                if self.bm25_models[content_type] is not None:
                    try:
                        scores = self.bm25_models[content_type].get_scores([term])
                        matches = sum(1 for score in scores if score > 0)
                        total_matches += matches
                        if matches > 0:
                            print(f"  📋 '{term}' found in {matches} {content_type} items")
                    except Exception as e:
                        print(f"  ⚠️ Error testing '{term}' in {content_type}: {e}")
            if total_matches == 0:
                print(f"  ❌ '{term}' not found in any indexed content!")
            else:
                print(f"  ✅ '{term}' found in {total_matches} total items")

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
            # Path already added at top of file, no need to add again
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
        """Perform hybrid search combining keyword (BM25) and semantic similarity with OR logic."""
        if not self.sentence_model:
            raise RuntimeError("SentenceTransformer model not loaded")
        
        # ULTRA-FAST: Check cache first
        cache_key = f"{query.lower()}:{content_type}:{top_k}"
        if cache_key in self.search_cache:
            print("⚡ Cache hit: Returning cached results (sub-millisecond)")
            return self.search_cache[cache_key]
        
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
        
        search_types = ['video', 'text', 'audio', 'image'] if content_type == 'all' else [content_type]
        
        # Step 1: Perform keyword search (BM25) - FASTEST
        keyword_results = {}  # {file_path: score}
        spell_corrections = {}
        
        # Enhanced query tokenization with spell checking
        if self.tokenizer:
            query_tokens, spell_corrections = self.tokenizer.tokenize_and_process(
                core_query,
                apply_spell_check=True,   # Apply spell checking to queries
                apply_stemming=True,
                remove_stopwords=True,
                min_length=2
            )
            
            if spell_corrections:
                print(f"🔤 Applied spell corrections: {spell_corrections}")
        else:
            # Fallback to basic tokenization
            query_tokens = re.findall(r'\b\w+\b', core_query.lower())
        
        for ctype in search_types:
            if self.bm25_models[ctype] is not None:
                try:
                    scores = self.bm25_models[ctype].get_scores(query_tokens)
                    file_paths = self.bm25_file_paths[ctype]
                    
                    for i, score in enumerate(scores):
                        if i < len(file_paths):
                            normalized_score = min(score / 10.0, 1.0)  # Normalize BM25 scores to 0-1 range
                            if normalized_score >= 0.3:  # 30% threshold for keyword
                                keyword_results[file_paths[i]] = {
                                    'score': normalized_score,
                                    'type': ctype,
                                    'search_type': 'keyword'
                                }
                except Exception as e:
                    print(f"⚠️ BM25 search failed for {ctype}: {e}")
        
        print(f"🔍 Keyword search: {len(keyword_results)} results above 30% threshold")
        
        # SPEED OPTIMIZATION: If BM25 found enough results, skip semantic search for common queries
        if len(keyword_results) >= top_k and len(query_tokens) <= 3:  # Short queries with good BM25 results
            print("⚡ Fast path: Using keyword-only results (sufficient matches found)")
            
            # Format and return BM25-only results
            formatted_results = []
            for file_path, search_data in keyword_results.items():
                ctype = search_data['type']
                
                result = {
                    'type': ctype,
                    'content_type': ctype,
                    'file_path': file_path,
                    'filename': os.path.basename(file_path),
                    'score': round(search_data['score'], 4),
                    'similarity_score': round(search_data['score'], 4),
                    'search_type': search_data['search_type']
                }
                
                # Add metadata
                if file_path in self.content_metadata[ctype]:
                    metadata = self.content_metadata[ctype][file_path]
                    
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
                
                formatted_results.append(result)
            
            # Sort by score and apply filters
            formatted_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Apply location filtering if specified
            if location_text:
                formatted_results = self._apply_location_filter(formatted_results, location_text, search_radius)
            
            # Apply date filtering if specified  
            if date_filter:
                formatted_results = self._apply_date_filter_with_buckets(formatted_results, date_filter)
            
            return formatted_results[:top_k]
        
        # Step 2: Perform semantic search (slower but more comprehensive)
        semantic_results = {}  # {file_path: score}
        all_embeddings = {}
        for ctype in search_types:
            all_embeddings.update({
                f"{ctype}:{path}": embedding 
                for path, embedding in self.content_embeddings[ctype].items()
            })
        
        if all_embeddings:
            query_embedding = self.sentence_model.encode([core_query])
            
            for type_path, embedding in all_embeddings.items():
                if embedding is not None:
                    try:
                        if isinstance(embedding, list):
                            embedding = np.array(embedding)
                        
                        similarity = util.pytorch_cos_sim(query_embedding, embedding).item()
                        
                        if similarity >= 0.3:  # 30% threshold for semantic
                            ctype, file_path = type_path.split(':', 1)
                            semantic_results[file_path] = {
                                'score': similarity,
                                'type': ctype,
                                'search_type': 'semantic'
                            }
                    except Exception as e:
                        print(f"⚠️ Semantic search failed for {type_path}: {e}")
        
        print(f"🔍 Semantic search: {len(semantic_results)} results above 30% threshold")
        
        # Step 3: Combine results using OR logic
        all_results = {}
        
        # Add keyword results
        for file_path, data in keyword_results.items():
            all_results[file_path] = data
        
        # Add semantic results (prefer higher scores)
        for file_path, data in semantic_results.items():
            if file_path in all_results:
                # File found by both methods - use the higher score
                if data['score'] > all_results[file_path]['score']:
                    all_results[file_path] = data
                    all_results[file_path]['search_type'] = 'hybrid'
                else:
                    all_results[file_path]['search_type'] = 'hybrid'
            else:
                all_results[file_path] = data
        
        print(f"🎯 Combined search: {len(all_results)} unique results (OR logic)")
        
        # Step 4: Format results with metadata
        formatted_results = []
        for file_path, search_data in all_results.items():
            ctype = search_data['type']
            
            result = {
                'type': ctype,
                'content_type': ctype,
                'file_path': file_path,
                'filename': os.path.basename(file_path),
                'score': round(search_data['score'], 4),
                'similarity_score': round(search_data['score'], 4),
                'search_type': search_data['search_type']
            }
            
            # Add metadata
            if file_path in self.content_metadata[ctype]:
                metadata = self.content_metadata[ctype][file_path]
                
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
            
            formatted_results.append(result)
        
        # Step 5: Sort by score and apply filters
        formatted_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply location filtering if specified
        if location_text:
            formatted_results = self._apply_location_filter(formatted_results, location_text, search_radius)
        
        # Apply date filtering if specified
        if date_filter:
            formatted_results = self._apply_date_filter_with_buckets(formatted_results, date_filter)
        
        final_results = formatted_results[:top_k]
        
        # Prepare response with metadata
        response_data = {
            'results': final_results,
            'search_metadata': {
                'original_query': query,
                'processed_query': core_query,
                'spell_corrections': spell_corrections,
                'query_tokens': query_tokens if 'query_tokens' in locals() else [],
                'tokenizer_used': 'enhanced' if self.tokenizer else 'basic',
                'capabilities': self.tokenizer_capabilities
            }
        }
        
        # Cache the results for future queries
        if len(self.search_cache) >= self.max_cache_size:
            # Simple LRU: remove oldest entry
            oldest_key = next(iter(self.search_cache))
            del self.search_cache[oldest_key]
        
        self.search_cache[cache_key] = response_data
        print(f"📊 Cached search results for future queries")
        
        return response_data
    
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
        non_empty_buckets = [k for k, v in buckets.items() if v]
        if non_empty_buckets:
            max_per_bucket = max(8, top_k // len(non_empty_buckets))  # At least 8 per bucket
        else:
            max_per_bucket = 8  # Default when no buckets have results
        
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
        
        # Check if we can connect to the port to see if server is already running
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', self.port))
            sock.close()
            
            if result == 0:
                # Port is already in use - check if it's a search server
                try:
                    import requests
                    response = requests.get(f"http://localhost:{self.port}/status", timeout=2)
                    if response.status_code == 200:
                        print(f"✅ Search server already running on port {self.port}")
                        print("⚠️ Aborting startup to prevent conflicts")
                        return
                except:
                    pass  # Not a search server, proceed with cleanup
        except:
            pass  # Can't check, proceed normally
        
        # Kill any existing processes on the port before starting (but not ourselves)
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