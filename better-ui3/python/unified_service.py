"""
ContentCache Unified Service

This module provides a single, unified interface for all content processing operations,
making it easy to integrate with UI frameworks (Electron, web, mobile, etc.).

Features:
- Single entry point for all content types (video, image, text, audio)
- Automatic file type detection
- Progress callbacks for UI updates
- Delegates to existing processing modules
"""

import os
import sys
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
from datetime import datetime
import requests

# Add the backend processing directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_processing_dir = os.path.join(current_dir, '..', '..', 'backend', 'processing')
sys.path.append(backend_processing_dir)

# Import existing processing functions
try:
    from videotagger import tag_video_smart_conflict_resolution
    from imageprocessor import tag_image
    from textprocessor import TextProcessor
    from audioanalyzer import analyze_audio_with_openai, save_audio_metadata
    from tagdirectory import batch_process_files
    from config import get_audio_metadata_path
except ImportError as e:
    print(f"Warning: Could not import backend modules: {e}")
    print(f"Please ensure the backend/processing directory exists and modules are available")
    # Provide fallback empty functions
    def tag_video_smart_conflict_resolution(file_path, **kwargs): return {"error": "Backend not available"}
    def tag_image(file_path): return {"error": "Backend not available"}
    class TextProcessor:
        def process_file(self, file_path): return {"error": "Backend not available"}
    def analyze_audio_with_openai(file_path): return {"error": "Backend not available"}
    def save_audio_metadata(file_path, result, metadata_file): pass
    def batch_process_files(directory_path): return {"error": "Backend not available"}
    def get_audio_metadata_path(): return "audio_metadata.json"

class ContentCacheService:
    """
    Unified service for all ContentCache operations.
    
    This class provides a simple, consistent interface for processing any type of content
    by delegating to the existing specialized processing modules.
    """
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        """
        Initialize the ContentCache service.
        
        Args:
            progress_callback: Optional callback function for progress updates
        """
        self.progress_callback = progress_callback
        self.text_processor = TextProcessor()
        self.search_server_url = "http://localhost:5001"
        
        # Supported file types
        self.video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.heic'}
        self.text_extensions = {'.txt', '.md', '.pdf', '.docx', '.rtf'}
        self.audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg'}
    
    def _emit_progress(self, stage: str, progress: float, message: str):
        """Emit progress update if callback is provided."""
        if self.progress_callback:
            self.progress_callback(stage, progress, message)
    
    def detect_file_type(self, file_path: str) -> str:
        """
        Detect the type of content based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File type: 'video', 'image', 'text', 'audio', or 'unknown'
        """
        ext = Path(file_path).suffix.lower()
        
        if ext in self.video_extensions:
            return 'video'
        elif ext in self.image_extensions:
            return 'image'
        elif ext in self.text_extensions:
            return 'text'
        elif ext in self.audio_extensions:
            return 'audio'
        else:
            return 'unknown'
    
    def process_file(self, file_path: str, **options) -> Dict[str, Any]:
        """
        Process a single file by calling the appropriate existing processing function.
        
        Args:
            file_path: Path to the file to process
            **options: Processing options
            
        Returns:
            Processing result dictionary
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_type = self.detect_file_type(file_path)
        filename = os.path.basename(file_path)
        abs_path = os.path.abspath(file_path)
        
        self._emit_progress("detection", 10, f"Detected {file_type} file: {filename}")
        
        # Check if file is already processed
        if self._is_file_already_processed(abs_path, file_type):
            self._emit_progress("complete", 100, f"File already processed, skipping: {filename}")
            return {
                "type": file_type,
                "file_path": file_path,
                "filename": filename,
                "processed_at": "already_processed",
                "result": "File already exists in metadata",
                "success": True,
                "skipped": True
            }
        
        try:
            if file_type == 'video':
                self._emit_progress("processing", 50, f"Processing video: {filename}")
                result = tag_video_smart_conflict_resolution(file_path)
                
            elif file_type == 'image':
                self._emit_progress("processing", 50, f"Processing image: {filename}")
                result = tag_image(file_path)
                
            elif file_type == 'text':
                self._emit_progress("processing", 50, f"Processing text: {filename}")
                result = self.text_processor.process_file(file_path)
                
            elif file_type == 'audio':
                self._emit_progress("processing", 50, f"Processing audio: {filename}")
                result = analyze_audio_with_openai(file_path)
                # Save metadata to ensure it persists
                save_audio_metadata(file_path, result, get_audio_metadata_path())
                
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            self._emit_progress("complete", 100, f"Processing complete: {filename}")
            
            return {
                "type": file_type,
                "file_path": file_path,
                "filename": filename,
                "processed_at": datetime.now().isoformat(),
                "result": result,
                "success": True
            }
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._emit_progress("error", 0, f"Processing failed: {str(e)}")
            return {
                "type": file_type,
                "file_path": file_path,
                "filename": filename,
                "processed_at": datetime.now().isoformat(),
                "error": str(e),
                "success": False
            }
    
    def _is_file_already_processed(self, abs_path: str, file_type: str) -> bool:
        """
        Check if a file has already been processed by checking the appropriate metadata file.
        
        Args:
            abs_path: Absolute path to the file
            file_type: Type of file (video, image, text, audio)
            
        Returns:
            True if file is already processed, False otherwise
        """
        try:
            # Import path functions as needed
            from config import (
                get_video_metadata_path, get_audio_metadata_path, 
                get_text_metadata_path, get_image_metadata_path
            )
            
            metadata_file = None
            if file_type == 'video':
                metadata_file = get_video_metadata_path()
            elif file_type == 'audio':
                metadata_file = get_audio_metadata_path()
            elif file_type == 'text':
                metadata_file = get_text_metadata_path()
            elif file_type == 'image':
                metadata_file = get_image_metadata_path()
            
            if metadata_file and os.path.exists(metadata_file):
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                return abs_path in metadata
            
            return False
            
        except Exception as e:
            # If we can't check metadata, proceed with processing
            print(f"Warning: Could not check metadata for {abs_path}: {e}")
            return False
    
    def process_directory(self, directory_path: str, recursive: bool = True, **options) -> Dict[str, Any]:
        """
        Process all supported files in a directory by calling the existing tagdirectory function.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to process subdirectories
            **options: Processing options
            
        Returns:
            Processing results summary
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        self._emit_progress("directory_processing", 0, f"Starting directory processing: {directory_path}")
        
        try:
            # Use the existing tagdirectory function which already has all the logic
            results = batch_process_files(directory_path)
            
            self._emit_progress("directory_processing", 100, "Directory processing complete")
            
            return {
                "directory_path": directory_path,
                "recursive": recursive,
                "processed_at": datetime.now().isoformat(),
                "results": results,
                "success": True
            }
            
        except Exception as e:
            self._emit_progress("error", 0, f"Directory processing failed: {str(e)}")
            return {
                "directory_path": directory_path,
                "recursive": recursive,
                "processed_at": datetime.now().isoformat(),
                "error": str(e),
                "success": False
            }
    
    def search_content(self, query: str, content_types: Optional[List[str]] = None, top_k: int = 20, 
                      date_filter: str = '', location_filter: str = '') -> Dict[str, Any]:
        """
        Search processed content by calling the search server.
        
        Args:
            query: Search query
            content_types: Optional list of content types to search
            top_k: Number of results to return
            date_filter: Optional date filter string (e.g., "2023-12-25" or "2023-12")
            location_filter: Optional location filter string (e.g., "San Francisco")
            
        Returns:
            Search results from search server (either flat results or bucketed)
        """
        try:
            # Prepare search request
            search_data = {
                "query": query,
                "top_k": top_k,
                "date_filter": date_filter,
                "location_filter": location_filter
            }
            
            # Handle content type filtering
            if content_types and len(content_types) == 1:
                search_data["type"] = content_types[0]
            else:
                search_data["type"] = "all"
            
            # Make request to search server (which has all the search logic)
            response = requests.post(
                f"{self.search_server_url}/search",
                json=search_data,
                timeout=30
            )
            
            if response.status_code == 200:
                search_results = response.json()
                
                # Check if server returned buckets or flat results
                if search_results.get('has_buckets') and 'buckets' in search_results:
                    # Server returned bucketed results
                    return {
                        "query": query,
                        "buckets": search_results.get('buckets', {}),
                        "has_buckets": True,
                        "date_filter": date_filter,
                        "location_filter": location_filter,
                        "total_found": search_results.get('total_results', 0),
                        "success": True
                    }
                else:
                    # Server returned flat results
                    return {
                        "query": query,
                        "results": search_results.get('results', []),
                        "has_buckets": False,
                        "total_found": len(search_results.get('results', [])),
                        "success": True
                    }
            else:
                return {
                    "query": query,
                    "results": [],
                    "error": f"Search server error: {response.status_code}",
                    "success": False
                }
                
        except requests.exceptions.ConnectionError:
            return {
                "query": query,
                "results": [],
                "error": "Could not connect to search server. Please ensure it's running.",
                "success": False
            }
        except Exception as e:
            return {
                "query": query,
                "results": [],
                "error": f"Search failed: {str(e)}",
                "success": False
            }
    
    def get_search_server_status(self) -> Dict[str, Any]:
        """Get the status of the search server."""
        try:
            response = requests.get(f"{self.search_server_url}/status", timeout=5)
            if response.status_code == 200:
                status_data = response.json()
                status_data['running'] = True
                return status_data
            else:
                return {'running': False, 'error': f'Server returned status {response.status_code}'}
        except requests.exceptions.ConnectionError:
            return {'running': False, 'error': 'Connection refused - server may not be running'}
        except Exception as e:
            return {'running': False, 'error': str(e)}


# Convenience functions for easy integration
def create_service(progress_callback: Optional[Callable] = None) -> ContentCacheService:
    """Create a new ContentCache service instance."""
    return ContentCacheService(progress_callback)

def process_file(file_path: str, progress_callback: Optional[Callable] = None, **options) -> Dict[str, Any]:
    """Quick function to process a single file."""
    service = ContentCacheService(progress_callback)
    return service.process_file(file_path, **options)

def process_directory(directory_path: str, progress_callback: Optional[Callable] = None, 
                     recursive: bool = True, **options) -> Dict[str, Any]:
    """Quick function to process a directory."""
    service = ContentCacheService(progress_callback)
    return service.process_directory(directory_path, recursive, **options)


# Example usage and testing
if __name__ == "__main__":
    def example_progress_callback(stage: str, progress: float, message: str):
        """Example progress callback for testing."""
        print(f"[{stage.upper()}] {progress:.1f}% - {message}")
    
    # Create service with progress callback
    service = ContentCacheService(example_progress_callback)
    
    # Test with command line arguments
    if len(sys.argv) > 1:
        target = sys.argv[1]
        
        if os.path.isfile(target):
            print(f"🔄 Processing file: {target}")
            result = service.process_file(target)
            print(f"✅ Result: {result}")
            
        elif os.path.isdir(target):
            print(f"🔄 Processing directory: {target}")
            result = service.process_directory(target)
            print(f"✅ Result: {result}")
            
        else:
            print(f"❌ Path not found: {target}")
    else:
        print("📋 ContentCache Unified Service")
        print("Usage: python unified_service.py <file_or_directory>")
        print("\nThis is a thin wrapper that delegates to existing processing modules:")
        print("  • videotagger.py for videos")
        print("  • imageprocessor.py for images") 
        print("  • textprocessor.py for text files")
        print("  • audioanalyzer.py for audio files")
        print("  • tagdirectory.py for batch directory processing")
        print("  • search_server.py for search functionality") 