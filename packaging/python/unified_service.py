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

print("🔧 Loading ContentCache Unified Service...")

import os
import sys
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
from datetime import datetime
import requests
import subprocess
import time

# Add the backend processing directory to path for imports
# Use current working directory since we set cwd to Resources in main.js
current_dir = os.getcwd()
print(f"💻 Working directory: {current_dir}")

# Try multiple possible locations for backend modules
possible_backend_dirs = [
    # Packaged app location (Resources/backend/processing)
    os.path.join(current_dir, 'backend', 'processing'),
    # Alternative packaged location
    os.path.join(current_dir, 'python-dist', 'backend', 'processing'),
    # Development location (fallback)
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend', 'processing'),
]

backend_processing_dir = None
for possible_dir in possible_backend_dirs:
    if os.path.exists(possible_dir):
        backend_processing_dir = possible_dir
        break

if backend_processing_dir:
    sys.path.append(backend_processing_dir)
    print(f"📁 Found backend processing directory: {backend_processing_dir}")
else:
    print("❌ Backend processing directory not found in any expected location")
    print(f"📁 Searched locations:")
    for i, dir_path in enumerate(possible_backend_dirs, 1):
        print(f"   {i}. {dir_path} - {'✅ exists' if os.path.exists(dir_path) else '❌ not found'}")

print(f"📁 Current directory: {current_dir}")
print(f"📁 Backend processing directory: {backend_processing_dir}")
print(f"📁 Backend directory exists: {os.path.exists(backend_processing_dir) if backend_processing_dir else False}")
if backend_processing_dir and os.path.exists(backend_processing_dir):
    print(f"📋 Backend directory contents: {os.listdir(backend_processing_dir)}")
print(f"🐍 Python path: {sys.path[:3]}...")

print("🔧 Attempting to import backend modules...")

# Import existing processing functions
try:
    print("🔧 Attempting individual module imports...")
    
    print("  Importing videotagger...")
    from videotagger import tag_video_smart_conflict_resolution
    print("  ✅ videotagger imported successfully")
    
    print("  Importing imageprocessor...")
    from imageprocessor import tag_image
    print("  ✅ imageprocessor imported successfully")
    
    print("  Importing textprocessor...")
    from textprocessor import TextProcessor
    print("  ✅ textprocessor imported successfully")
    
    print("  Importing audioanalyzer...")
    from audioanalyzer import analyze_audio_with_openai, save_audio_metadata
    print("  ✅ audioanalyzer imported successfully")
    
    print("  Importing tagdirectory...")
    from tagdirectory import batch_process_files
    print("  ✅ tagdirectory imported successfully")
    
    print("  Importing config...")
    from config import (
        get_audio_metadata_path, get_video_metadata_path, 
        get_text_metadata_path, get_image_metadata_path
    )
    print("  ✅ config imported successfully")
    
    print("✅ Backend modules imported successfully")
    backend_modules_available = True
except ImportError as e:
    print(f"❌ WARNING: Could not import backend modules!")
    print(f"❌ Import error: {str(e)}")
    print(f"❌ Error type: {type(e).__name__}")
    
    # Try to give more specific error information
    import traceback
    print(f"📋 Full traceback:")
    traceback.print_exc()
    
    print(f"📁 Please ensure the backend/processing directory exists and modules are available")
    print(f"🔧 Using fallback functions - processing will not work properly")
    backend_modules_available = False
    
    # Provide fallback empty functions
    def tag_video_smart_conflict_resolution(file_path, **kwargs): 
        print(f"🚫 FALLBACK: tag_video_smart_conflict_resolution called - backend not available")
        return {"error": "Backend not available"}
    def tag_image(file_path): 
        print(f"🚫 FALLBACK: tag_image called - backend not available")
        return {"error": "Backend not available"}
    def analyze_audio_with_openai(file_path): 
        print(f"🚫 FALLBACK: analyze_audio_with_openai called - backend not available")
        return {"error": "Backend not available"}
    def save_audio_metadata(file_path, result, metadata_file): pass
    def batch_process_files(directory_path): 
        print(f"🚫 FALLBACK: batch_process_files called - backend not available")
        return {"error": "Backend not available"}
    def get_audio_metadata_path(): return "audio_metadata.json"
    def get_video_metadata_path(): return "video_metadata.json"
    def get_text_metadata_path(): return "text_metadata.json"
    def get_image_metadata_path(): return "image_metadata.json"
    
    # Define fallback TextProcessor class for import failures
    class TextProcessor:
        def process_file(self, file_path): 
            print(f"🚫 FALLBACK: TextProcessor.process_file called - backend import failed")
            print(f"🔧 This suggests there was an import error in the backend modules")
            return {"error": "Backend TextProcessor not available due to import failure", "file_path": file_path}


# Auto-start search server if not running
def check_and_start_search_server():
    """Check if search server is running, and start it if not."""
    search_server_url = "http://localhost:5001"
    
    try:
        # Test if search server is already running
        response = requests.get(f"{search_server_url}/status", timeout=2)
        if response.status_code == 200:
            print("✅ Search server is already running")
            return True
    except requests.exceptions.ConnectionError:
        print("🔍 Search server not running, attempting to start...")
    except Exception as e:
        print(f"⚠️ Error checking search server: {e}")
    
    # Try to find and start the search server
    possible_search_dirs = [
        # Packaged app location (Resources/backend/search)
        os.path.join(current_dir, 'backend', 'search'),
        # Alternative packaged location
        os.path.join(current_dir, 'python-dist', 'backend', 'search'),
        # Development location (fallback)
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend', 'search'),
    ]
    
    search_server_script = None
    search_working_dir = None
    
    for possible_dir in possible_search_dirs:
        search_script_path = os.path.join(possible_dir, 'search_server.py')
        if os.path.exists(search_script_path):
            search_server_script = search_script_path
            search_working_dir = possible_dir
            break
    
    if not search_server_script:
        print("❌ Search server script not found in any expected location")
        print("📁 Searched locations:")
        for i, dir_path in enumerate(possible_search_dirs, 1):
            script_path = os.path.join(dir_path, 'search_server.py')
            print(f"   {i}. {script_path} - {'✅ exists' if os.path.exists(script_path) else '❌ not found'}")
        return False
    
    print(f"🚀 Starting search server: {search_server_script}")
    print(f"📁 Working directory: {search_working_dir}")
    
    try:
        # Start the search server in the background
        search_process = subprocess.Popen(
            [sys.executable, 'search_server.py'],
            cwd=search_working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Detach the process so it continues running independently
            start_new_session=True
        )
        
        print(f"✅ Search server process started with PID: {search_process.pid}")
        
        # Wait a few seconds and test if it's working
        print("⏳ Waiting 5 seconds for search server to start...")
        time.sleep(5)
        
        try:
            response = requests.get(f"{search_server_url}/status", timeout=10)
            if response.status_code == 200:
                print("✅ Search server started successfully!")
                return True
            else:
                print(f"⚠️ Search server responded with status {response.status_code}")
                return False
        except Exception as e:
            print(f"⚠️ Search server may not be fully ready yet: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to start search server: {e}")
        return False

# Try to start search server if needed
print("🔍 Checking search server status...")
# Disabled: Let Electron main process handle search server startup
# check_and_start_search_server()

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
        self._text_processor = None  # Lazy initialization
        self.search_server_url = "http://localhost:5001"
        
        # Supported file types
        self.video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.heic'}
        self.text_extensions = {'.txt', '.md', '.pdf', '.docx', '.rtf'}
        self.audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg'}
    
    @property
    def text_processor(self):
        """Lazy initialization of TextProcessor"""
        if self._text_processor is None:
            try:
                # Access TextProcessor from global scope explicitly
                TextProcessorClass = globals().get('TextProcessor')
                if TextProcessorClass is None:
                    raise NameError("TextProcessor not found in global scope")
                self._text_processor = TextProcessorClass()
                print("✅ TextProcessor initialized successfully")
            except Exception as e:
                print(f"❌ Error initializing TextProcessor: {e}")
                print(f"🔧 Available globals: {list(globals().keys())[:10]}...")  # Debug info
                # Fallback class
                class FallbackTextProcessor:
                    def process_file(self, file_path): 
                        print(f"🚫 EMERGENCY FALLBACK: TextProcessor init failed - {str(e)}")
                        return {"error": f"TextProcessor initialization failed: {str(e)}", "file_path": file_path}
                self._text_processor = FallbackTextProcessor()
        return self._text_processor
    
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
        
        # Check stop flag at the beginning
        if self.stop_flag and callable(self.stop_flag) and self.stop_flag():
            return {
                "file_path": file_path,
                "filename": os.path.basename(file_path),
                "processed_at": datetime.now().isoformat(),
                "result": "Processing stopped by user",
                "success": False,
                "stopped": True
            }

        file_type = self.detect_file_type(file_path)
        filename = os.path.basename(file_path)
        abs_path = os.path.abspath(file_path)
        
        self._emit_progress("detection", 10, f"Detected {file_type} file: {filename}")
        
        # Check stop flag again before metadata check
        if self.stop_flag and callable(self.stop_flag) and self.stop_flag():
            return {
                "file_path": file_path,
                "filename": filename,
                "processed_at": datetime.now().isoformat(),
                "result": "Processing stopped by user",
                "success": False,
                "stopped": True
            }
        
        # Check if file is already processed
        is_already_processed = self._is_file_already_processed(abs_path, file_type)
        print(f"🔍 Checking if file already processed: {abs_path}")
        print(f"📋 Metadata check result: {'Already processed' if is_already_processed else 'New file'}")
        
        if is_already_processed:
            self._emit_progress("complete", 100, f"File already processed, skipping: {filename}")
            print(f"⏭️ Skipping already processed file: {filename}")
            return {
                "type": file_type,
                "file_path": file_path,
                "filename": filename,
                "processed_at": "already_processed",
                "result": "File already exists in metadata",
                "success": True,
                "skipped": True
            }
        
        # Check stop flag before starting processing
        if self.stop_flag and callable(self.stop_flag) and self.stop_flag():
            return {
                "file_path": file_path,
                "filename": filename,
                "processed_at": datetime.now().isoformat(),
                "result": "Processing stopped by user",
                "success": False,
                "stopped": True
            }
        
        print(f"🚀 Processing new file: {filename}")
        
        try:
            if file_type == 'video':
                self._emit_progress("processing", 50, f"Processing video: {filename}")
                print(f" Calling tag_video_smart_conflict_resolution for: {file_path}")
                result = tag_video_smart_conflict_resolution(file_path, stop_flag=self.stop_flag)
                print(f"🎥 Video processing result: {type(result)} - {str(result)[:200]}...")
                
            elif file_type == 'image':
                self._emit_progress("processing", 50, f"Processing image: {filename}")
                # Check if tag_image accepts stop_flag (it might not, but we'll try)
                try:
                    result = tag_image(file_path, stop_flag=self.stop_flag)
                except TypeError:
                    # If it doesn't accept stop_flag, call without it
                    result = tag_image(file_path)
                
            elif file_type == 'text':
                self._emit_progress("processing", 50, f"Processing text: {filename}")
                result = self.text_processor.process_file(file_path)
                
            elif file_type == 'audio':
                self._emit_progress("processing", 50, f"Processing audio: {filename}")
                # Check if analyze_audio_with_openai accepts stop_flag
                try:
                    result = analyze_audio_with_openai(file_path, stop_flag=self.stop_flag)
                except TypeError:
                    # If it doesn't accept stop_flag, call without it
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
            metadata_file = None
            if file_type == 'video':
                metadata_file = get_video_metadata_path()
            elif file_type == 'audio':
                metadata_file = get_audio_metadata_path()
            elif file_type == 'text':
                metadata_file = get_text_metadata_path()
            elif file_type == 'image':
                metadata_file = get_image_metadata_path()
            
            print(f"📁 Checking metadata file: {metadata_file}")
            print(f"📁 Metadata file exists: {os.path.exists(metadata_file) if metadata_file else False}")
            
            if metadata_file and os.path.exists(metadata_file):
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                file_found = abs_path in metadata
                print(f"📋 Total items in metadata: {len(metadata)}")
                print(f"🔍 Looking for: {abs_path}")
                print(f"✅ File found in metadata: {file_found}")
                
                if not file_found and len(metadata) > 0:
                    print(f"📋 First few paths in metadata:")
                    for i, path in enumerate(list(metadata.keys())[:3]):
                        print(f"    {i+1}. {path}")
                
                return file_found
            
            print(f"📋 No metadata file found, treating as new file")
            return False
            
        except Exception as e:
            # If we can't check metadata, proceed with processing
            print(f"⚠️ Warning: Could not check metadata for {abs_path}: {e}")
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

    def stop_processing(self) -> Dict[str, Any]:
        """Stop any currently running processing by calling tagdirectory stop function."""
        try:
            # Try to import and call the tagdirectory stop function
            from tagdirectory import stop_running_instance
            result = stop_running_instance()
            return {
                "success": True,
                "message": "Stop command sent to tagdirectory",
                "result": result
            }
        except ImportError as e:
            return {
                "success": False,
                "error": f"Could not import tagdirectory stop function: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to stop processing: {str(e)}"
            }


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

# TextProcessor should be available from imports or fallback defined in error handler above
# No need for additional fallback here 