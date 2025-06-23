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
import subprocess
import tempfile
import json
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
from datetime import datetime
import requests

# Add current directory to sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import API client for connectivity testing
try:
    from api_client import get_api_client
    API_CLIENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: API client not available: {e}")
    API_CLIENT_AVAILABLE = False

# Import processing modules
try:
    from videotagger import tag_video
    from audioprocessor import process_audio  
    from imageprocessor import process_image
    from textprocessor import process_text_file
except ImportError as e:
    print(f"Warning: Some processing modules not available: {e}")

# Import existing processing functions
from tagdirectory import batch_process_files

class ContentCacheService:
    """
    Unified service for all ContentCache operations.
    
    This class provides a simple, consistent interface for processing any type of content
    by delegating to the existing specialized processing modules.
    """
    
    def __init__(self, progress_callback: Optional[Callable] = None, stop_flag: Optional[Callable] = None):
        """
        Initialize the ContentCache service.
        
        Args:
            progress_callback: Optional callback function for progress updates
            stop_flag: Optional callable that returns True when processing should stop
        """
        self.progress_callback = progress_callback
        self.stop_flag = stop_flag
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
        self._emit_progress("checking", 20, f"Checking if file is already processed...")
        if self._is_file_already_processed(abs_path, file_type):
            self._emit_progress("skipped", 100, f"File already processed: {filename}")
            return {
                "status": "skipped",
                "message": f"File {filename} is already processed",
                "file_path": abs_path,
                "file_type": file_type
            }
        
        self._emit_progress("starting", 30, f"Starting {file_type} processing...")
        
        try:
            if file_type == "video":
                self._emit_progress("video_processing", 40, f"Processing video with AI analysis...")
                result = tag_video(abs_path)
                
            elif file_type == "audio":
                self._emit_progress("audio_processing", 40, f"Processing audio with transcription...")
                result = process_audio(abs_path)
                
            elif file_type == "image":
                self._emit_progress("image_processing", 40, f"Processing image with vision analysis...")
                result = process_image(abs_path)
                
            elif file_type == "text":
                self._emit_progress("text_processing", 40, f"Processing text document...")
                result = process_text_file(abs_path)
                
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Log the result for debugging
            if result:
                self._emit_progress("api_success", 90, f"API processing completed successfully")
                self._emit_progress("complete", 100, f"Processing completed: {filename}")
                self._refresh_search_server()
                return {
                    "status": "success",
                    "message": f"Successfully processed {filename}",
                    "file_path": abs_path,
                    "file_type": file_type,
                    "result": result
                }
            else:
                self._emit_progress("api_failed", 100, f"API processing returned empty result")
                return {
                    "status": "error",
                    "message": f"Processing returned empty result for {filename}",
                    "file_path": abs_path,
                    "file_type": file_type
                }
                
        except Exception as e:
            error_msg = f"Error processing {filename}: {str(e)}"
            self._emit_progress("error", 100, error_msg)
            # Log full error details
            print(f"🚨 PROCESSING ERROR: {error_msg}")
            print(f"🚨 ERROR TYPE: {type(e).__name__}")
            print(f"🚨 ERROR DETAILS: {e}")
            return {
                "status": "error",
                "message": error_msg,
                "file_path": abs_path,
                "file_type": file_type,
                "error": str(e)
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
            
            print(f"🔍 DEBUG: Checking if {abs_path} is already processed")
            print(f"🔍 DEBUG: File type: {file_type}")
            print(f"🔍 DEBUG: Metadata file: {metadata_file}")
            
            if metadata_file and os.path.exists(metadata_file):
                print(f"🔍 DEBUG: Metadata file exists, loading...")
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                file_exists = abs_path in metadata
                print(f"🔍 DEBUG: File exists in metadata: {file_exists}")
                print(f"🔍 DEBUG: Total files in metadata: {len(metadata)}")
                
                if file_exists:
                    print(f"🔍 DEBUG: File found in metadata, skipping processing")
                else:
                    print(f"🔍 DEBUG: File not found in metadata, will process")
                
                return file_exists
            else:
                print(f"🔍 DEBUG: Metadata file does not exist, will process")
                return False
            
        except Exception as e:
            # If we can't check metadata, proceed with processing
            print(f"⚠️ WARNING: Could not check metadata for {abs_path}: {e}")
            print(f"🔍 DEBUG: Proceeding with processing due to error")
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
        
        try:
            # Use the existing tagdirectory function which now handles progress callbacks
            results = batch_process_files(directory_path, progress_callback=self._emit_progress, stop_flag=self.stop_flag)
            
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
    
    def search_content(self, query: str, content_types: Optional[List[str]] = None, top_k: int = 20) -> Dict[str, Any]:
        """
        Search processed content by calling the search server.
        
        Args:
            query: Search query
            content_types: Optional list of content types to search
            top_k: Number of results to return
            
        Returns:
            Search results from search server
        """
        try:
            # Prepare search request
            search_data = {
                "query": query,
                "top_k": top_k
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
                return response.json()
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
    
    def test_api_connectivity(self) -> Dict[str, Any]:
        """
        Test connectivity to the ContentCache API server.
        
        Returns:
            Dict with test results
        """
        if not API_CLIENT_AVAILABLE:
            return {
                "status": "error",
                "message": "API client not available - import failed"
            }
        
        try:
            self._emit_progress("api_test", 10, "Testing API connectivity...")
            
            # Get API client
            api_client = get_api_client()
            self._emit_progress("api_test", 30, f"Testing connection to {api_client.base_url}...")
            
            # Test health endpoint
            health_result = api_client.health_check()
            self._emit_progress("api_test", 60, "Health check passed!")
            
            # Test a simple OpenAI call
            test_messages = [{"role": "user", "content": "Hello, respond with 'API test successful'"}]
            chat_result = api_client.openai_chat(
                messages=test_messages, 
                model="gpt-4o-mini",
                max_tokens=20
            )
            
            self._emit_progress("api_test", 100, "API connectivity test completed successfully!")
            
            return {
                "status": "success",
                "message": "API connectivity test passed",
                "health": health_result,
                "chat_test": chat_result,
                "api_url": api_client.base_url
            }
            
        except Exception as e:
            error_msg = f"API connectivity test failed: {str(e)}"
            self._emit_progress("api_test", 100, error_msg)
            print(f"🚨 API TEST ERROR: {error_msg}")
            return {
                "status": "error", 
                "message": error_msg,
                "error_details": str(e)
            }
    
    def _refresh_search_server(self):
        """Trigger search server to refresh embeddings after new content is processed."""
        try:
            response = requests.post(f"{self.search_server_url}/refresh", timeout=10)
            if response.status_code == 200:
                print("✅ Search server refreshed with new embeddings")
                return True
            else:
                print(f"⚠️ Search server refresh failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"⚠️ Could not refresh search server: {e}")
            return False

    def stop_processing(self):
        """Stop any running directory processing by calling tagdirectory's stop function."""
        try:
            from tagdirectory import stop_running_instance
            print("🛑 Calling tagdirectory stop_running_instance...")
            result = stop_running_instance()
            if result:
                print("✅ Successfully stopped tagdirectory processing")
                return {"status": "success", "message": "Processing stopped successfully"}
            else:
                print("⚠️ No running process found or failed to stop")
                return {"status": "warning", "message": "No running process found or failed to stop"}
        except Exception as e:
            error_msg = f"Failed to stop processing: {str(e)}"
            print(f"❌ {error_msg}")
            return {"status": "error", "message": error_msg}


# Convenience functions for easy integration
def create_service(progress_callback: Optional[Callable] = None, stop_flag: Optional[Callable] = None) -> ContentCacheService:
    """Create a new ContentCache service instance."""
    return ContentCacheService(progress_callback, stop_flag)

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