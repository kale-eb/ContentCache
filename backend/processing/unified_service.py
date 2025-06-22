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

# Add the processing directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing processing functions
from videotagger import tag_video
from imageprocessor import tag_image
from textprocessor import TextProcessor
from audioanalyzer import analyze_audio_with_openai
from tagdirectory import batch_process_files

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
        
        self._emit_progress("detection", 10, f"Detected {file_type} file: {filename}")
        
        try:
            if file_type == 'video':
                self._emit_progress("processing", 50, f"Processing video: {filename}")
                result = tag_video(file_path)
                
            elif file_type == 'image':
                self._emit_progress("processing", 50, f"Processing image: {filename}")
                result = tag_image(file_path)
                
            elif file_type == 'text':
                self._emit_progress("processing", 50, f"Processing text: {filename}")
                result = self.text_processor.process_file(file_path)
                
            elif file_type == 'audio':
                self._emit_progress("processing", 50, f"Processing audio: {filename}")
                result = analyze_audio_with_openai(file_path)
                
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
            self._emit_progress("error", 0, f"Processing failed: {str(e)}")
            return {
                "type": file_type,
                "file_path": file_path,
                "filename": filename,
                "processed_at": datetime.now().isoformat(),
                "error": str(e),
                "success": False
            }
    
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