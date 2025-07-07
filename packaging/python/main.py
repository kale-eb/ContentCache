#!/usr/bin/env python3
"""
ContentCache Main Entry Point

CRITICAL: Environment isolation must happen BEFORE any imports
to prevent conflicts with system NumPy 2.x and other incompatible packages.
"""

# STEP 1: Setup isolated environment BEFORE any imports
import sys
import os
from pathlib import Path

def setup_isolated_environment():
    """Setup isolated Python environment to avoid system package conflicts."""
    print("🔧 Setting up isolated Python environment...")
    
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
    
    print(f"📦 Using isolated packages from: {user_site_packages}")
    
    # Add our isolated packages to the FRONT of sys.path (highest priority)
    if user_site_packages not in sys.path:
        sys.path.insert(0, user_site_packages)
        print(f"✅ Added {user_site_packages} to sys.path[0]")
    
    # Also add to PYTHONPATH for child processes
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if user_site_packages not in current_pythonpath:
        new_pythonpath = user_site_packages + (os.pathsep + current_pythonpath if current_pythonpath else '')
        os.environ['PYTHONPATH'] = new_pythonpath
        print(f"✅ Updated PYTHONPATH")
    
    # Set user base for pip installations
    os.environ['PYTHONUSERBASE'] = user_base
    os.environ['PYTHONNOUSERSITE'] = '0'  # Allow user site packages
    
    print(f"🔧 Python path priority: {sys.path[:3]}...")
    
    # Test if we can import numpy from the correct location
    try:
        import numpy as np
        print(f"✅ Using NumPy {np.__version__} from: {np.__file__}")
        
        # Verify it's NumPy 1.x
        major_version = int(np.__version__.split('.')[0])
        if major_version >= 2:
            print(f"⚠️ WARNING: Found NumPy {np.__version__} (2.x) - may cause OpenCV issues")
        else:
            print(f"✅ NumPy {np.__version__} (1.x) - compatible with OpenCV")
            
    except ImportError as e:
        print(f"❌ Could not import NumPy: {e}")
        print(f"💡 Dependencies may need to be installed to: {user_site_packages}")

# CRITICAL: Run environment setup before any other imports
try:
    setup_isolated_environment()
except Exception as e:
    print(f"⚠️ Isolated environment setup failed: {e}")
    print("🔄 Falling back to system packages...")
    # Continue with system packages - no sys.path modification

# STEP 2: Now safe to import other modules
import json
import time
import threading
import queue

# Import the ContentCache service (this will now use isolated environment or system fallback)
try:
    from unified_service import ContentCacheService
    print("✅ Successfully imported ContentCacheService")
except ImportError as e:
    print(f"❌ Failed to import ContentCacheService: {e}")
    print("💡 This may indicate missing dependencies")
    # Let the error propagate - app cannot function without this
    raise

class ElectronBridge:
    """Bridge between Electron frontend and ContentCache unified service"""
    
    def __init__(self):
        # Create the unified service with progress callback and stop flag
        self.service = ContentCacheService(
            progress_callback=self.progress_callback,
            stop_flag=lambda: self.processing_stopped
        )
        self.processing_stopped = False
        
        # Set up API client stop callback if available
        try:
            from api_client import get_api_client
            client = get_api_client()
            client.set_stop_callback(self.trigger_stop_from_api)
            print("✅ API client configured with stop callback in ElectronBridge")
        except Exception as e:
            print(f"⚠️ Could not configure API client stop callback in ElectronBridge: {e}")
            
        self.send_output("ContentCache service initialized")
    
    def progress_callback(self, stage: str, progress: float, message: str):
        """Handle progress updates from the unified service"""
        progress_data = {
            "type": "progress",
            "stage": stage,
            "progress": progress,
            "message": message
        }
        self.send_json_response(progress_data)
    
    def send_output(self, message: str):
        """Send plain text output to Electron"""
        print(f"OUTPUT: {message}", flush=True)
        sys.stdout.flush()
    
    def send_json_response(self, data: dict):
        """Send JSON response to Electron"""
        print(json.dumps(data), flush=True)
        sys.stdout.flush()
    
    def send_ai_response(self, message: str):
        """Send AI chat response to Electron"""
        response = {
            "type": "ai_response",
            "message": message
        }
        self.send_json_response(response)
    
    def process_files(self, file_paths: list):
        """Process multiple files using the unified service"""
        self.send_output(f"DEBUG: process_files called with: {file_paths}, type: {type(file_paths)}")
        
        if not file_paths:
            self.send_output("ERROR: No files provided to process_files")
            return []
            
        results = []
        total_files = len(file_paths)
        self.processing_stopped = False  # Reset stop flag
        
        self.send_output(f"Starting to process {total_files} file(s)")
        
        for i, file_path in enumerate(file_paths):
            self.send_output(f"DEBUG: Processing file {i+1}/{total_files}: {file_path} (type: {type(file_path)})")
            
            # Check if processing should stop
            if self.processing_stopped:
                self.send_output("Processing stopped by user")
                break
                
            try:
                if os.path.isdir(file_path):
                    # Process directory
                    self.send_output(f"Processing directory: {file_path}")
                    result = self.service.process_directory(file_path)
                    
                    # Check if directory processing was stopped
                    if result.get('results', {}).get('status') == 'stopped':
                        self.processing_stopped = True
                        results.append(result)
                        self.send_output("Directory processing was stopped")
                        break
                else:
                    # Process single file
                    self.send_output(f"Processing file: {os.path.basename(file_path)}")
                    result = self.service.process_file(file_path)
                    
                    # Check if file processing was stopped
                    if result.get('stopped', False):
                        self.processing_stopped = True
                        results.append(result)
                        self.send_output("File processing was stopped")
                        break
                
                results.append(result)
                
                # Send progress update with file type info
                progress = ((i + 1) / total_files) * 100
                file_type = self.service.detect_file_type(file_path)
                self.send_json_response({
                    "type": "batch_progress",
                    "progress": progress,
                    "completed": i + 1,
                    "total": total_files,
                    "current_file": os.path.basename(file_path),
                    "current_file_type": file_type,
                    "current_result": result.get('success', False)
                })
                
            except Exception as e:
                error_result = {
                    "file_path": file_path,
                    "error": str(e),
                    "success": False
                }
                results.append(error_result)
                self.send_output(f"Error processing {file_path}: {str(e)}")
        
        # Send final results with detailed statistics
        successful_files = [r for r in results if r.get('success', False)]
        skipped_files = [r for r in results if r.get('skipped', False)]
        failed_files = [r for r in results if not r.get('success', False) and not r.get('skipped', False)]
        
        final_status = "stopped" if self.processing_stopped else "complete"
        self.send_json_response({
            "type": "processing_complete",
            "results": results,
            "total_processed": len(results),
            "successful": len(successful_files),
            "skipped": len(skipped_files),
            "failed": len(failed_files),
            "status": final_status,
            "summary": {
                "total_files": total_files,
                "processed_new": len(successful_files),
                "skipped_existing": len(skipped_files),
                "failed": len(failed_files),
                "stopped_early": self.processing_stopped
            }
        })
        
        return results
    
    def search_videos(self, query: str, content_types: list = None, top_k: int = 20, 
                     date_filter: str = '', location_filter: str = '', offset: int = 0):
        """Search content using the unified service with optional filters and pagination"""
        self.send_output(f"Searching for: '{query}' with content_types={content_types}, top_k={top_k}, date_filter='{date_filter}', location_filter='{location_filter}', offset={offset}")
        
        try:
            # Check search server status first
            self.send_output("Checking search server status...")
            status = self.service.get_search_server_status()
            self.send_output(f"Search server status: {status}")
            
            if not status.get('running', False):
                self.send_json_response({
                    "type": "search_error",
                    "error": "Search server is not running. Please start the search server first.",
                    "suggestion": "Run 'python search_server.py' in a separate terminal"
                })
                return
            
            # Perform search with filters and pagination
            self.send_output("Calling search_content...")
            results = self.service.search_content(query, content_types, top_k, date_filter, location_filter, offset)
            self.send_output(f"Search results received: {results}")
            
            if results.get('success', False):
                # Check if results are bucketed or flat
                if results.get('has_buckets', False) and 'buckets' in results:
                    # Return bucketed results
                    self.send_json_response({
                        "type": "search_results",
                        "query": query,
                        "buckets": results.get('buckets', {}),
                        "has_buckets": True,
                        "date_filter": date_filter,
                        "location_filter": location_filter,
                        "total_found": results.get('total_found', 0)
                    })
                    self.send_output(f"Found {results.get('total_found', 0)} results in {len(results.get('buckets', {}))} buckets")
                else:
                    # Return flat results with pagination info
                    response_data = {
                        "type": "search_results",
                        "query": query,
                        "results": results.get('results', []),
                        "has_buckets": False,
                        "total_found": results.get('total_found', len(results.get('results', [])))
                    }
                    
                    # Add pagination info if available
                    if 'offset' in results:
                        response_data['offset'] = results['offset']
                        response_data['limit'] = results.get('limit', top_k)
                        response_data['has_more'] = results.get('has_more', False)
                        response_data['is_filter_only'] = results.get('is_filter_only', False)
                        response_data['applied_filters'] = results.get('applied_filters', {})
                    
                    self.send_json_response(response_data)
                    
                    if results.get('is_filter_only', False):
                        self.send_output(f"Found {results.get('total_found', 0)} filter-only results (showing {len(results.get('results', []))} from offset {offset})")
                    else:
                        self.send_output(f"Found {len(results.get('results', []))} results")
            else:
                self.send_json_response({
                    "type": "search_error",
                    "error": results.get('error', 'Unknown search error'),
                    "query": query
                })
                
        except Exception as e:
            self.send_json_response({
                "type": "search_error",
                "error": str(e),
                "query": query
            })
            self.send_output(f"Search error: {str(e)}")
    
    def ai_chat(self, message: str):
        """Handle AI chat messages"""
        try:
            # Check if search server is running for context
            status = self.service.get_search_server_status()
            
            if status.get('running', False):
                # If search server is running, we can provide context-aware responses
                if "search" in message.lower() or "find" in message.lower():
                    # Extract potential search terms
                    search_terms = message.lower().replace("search for", "").replace("find", "").strip()
                    if search_terms:
                        # Perform a search to provide context
                        search_results = self.service.search_content(search_terms, top_k=5)
                        if search_results.get('success', False) and search_results.get('results'):
                            response = f"I found {len(search_results['results'])} items related to '{search_terms}':\n\n"
                            for i, result in enumerate(search_results['results'][:3], 1):
                                response += f"{i}. {result.get('filename', 'Unknown file')}\n"
                            response += f"\nWould you like me to search for something more specific?"
                        else:
                            response = f"I searched for '{search_terms}' but didn't find any matching content. Try different keywords or make sure your content has been processed first."
                    else:
                        response = "I can help you search your content! What would you like to find?"
                
                elif "how many" in message.lower() or "count" in message.lower():
                    # Get status information
                    response = f"Based on the search server status, I can help you find information about your content library. What specific counts are you looking for?"
                
                else:
                    # General AI response
                    response = f"I understand you're asking: '{message}'. I can help you search through your processed content, analyze files, and answer questions about your video library. What would you like to know?"
            
            else:
                # Search server not running - limited functionality
                if "search" in message.lower():
                    response = "I'd love to help you search, but the search server isn't running. Please start it with 'python search_server.py' first, then I can search through your content!"
                else:
                    response = f"I understand your question: '{message}'. To provide better answers about your content, please start the search server first. I can still help process new files though!"
            
            self.send_ai_response(response)
            
        except Exception as e:
            self.send_ai_response(f"I encountered an error: {str(e)}. Please try again or check the system logs.")
    
    def get_system_status(self):
        """Get overall system status"""
        try:
            search_status = self.service.get_search_server_status()
            
            status = {
                "type": "system_status",
                "search_server": search_status,
                "unified_service": "running",
                "supported_formats": {
                    "video": list(self.service.video_extensions),
                    "image": list(self.service.image_extensions),
                    "text": list(self.service.text_extensions),
                    "audio": list(self.service.audio_extensions)
                }
            }
            
            self.send_json_response(status)
            
        except Exception as e:
            self.send_json_response({
                "type": "system_status",
                "error": str(e)
            })

    def stop_processing_enhanced(self):
        """Enhanced stop processing that calls both flags and tagdirectory stop"""
        try:
            # Set the flag for current processing loop
            self.processing_stopped = True
            self.send_output("🛑 Stop flag set in ElectronBridge")
            
            # Also call the tagdirectory stop function for any running instances
            result = self.service.stop_processing()
            self.send_output(f"🛑 Tagdirectory stop result: {result}")
            
            # Check if the stop was successful
            if result.get("status") == "success":
                self.send_json_response({
                    "type": "stop_complete",
                    "message": "Processing stop initiated successfully",
                    "tagdirectory_result": result
                })
            else:
                self.send_json_response({
                    "type": "stop_error",
                    "error": result.get("message", "Unknown error stopping processing"),
                    "tagdirectory_result": result
                })
            
        except Exception as e:
            error_msg = f"Failed to stop processing: {str(e)}"
            self.send_output(error_msg)
            self.send_json_response({
                "type": "stop_error",
                "error": error_msg
            })

    def trigger_stop_from_api(self):
        """Called by API client when repeated failures occur"""
        print("🛑 Stop triggered by API failures - setting processing_stopped flag")
        self.processing_stopped = True
        self.send_json_response({
            "type": "api_stop",
            "message": "Processing stopped due to repeated API failures"
            })

def stdin_reader_thread(command_queue):
    """Thread function to read commands from stdin and put them in a queue"""
    try:
        for line in sys.stdin:
            try:
                command = json.loads(line.strip())
                command_queue.put(command)
            except json.JSONDecodeError as e:
                command_queue.put({"action": "error", "message": f"Invalid JSON: {str(e)}"})
            except Exception as e:
                command_queue.put({"action": "error", "message": f"Error reading command: {str(e)}"})
    except Exception as e:
        command_queue.put({"action": "error", "message": f"Stdin reader error: {str(e)}"})

def main():
    """Main loop for handling Electron commands with non-blocking stdin"""
    bridge = ElectronBridge()
    
    # Send initial status
    bridge.get_system_status()
    
    # Create a queue for commands and start stdin reader thread
    command_queue = queue.Queue()
    stdin_thread = threading.Thread(target=stdin_reader_thread, args=(command_queue,), daemon=True)
    stdin_thread.start()
    
    bridge.send_output("🚀 ContentCache service ready - using threaded command processing")
    
    # Main processing loop
    try:
        while True:
            # Check for new commands (non-blocking)
            try:
                # Wait up to 0.1 seconds for a command
                command = command_queue.get(timeout=0.1)
                
                action = command.get('action')
                bridge.send_output(f"📨 Received command: {action}")
                
                if action == 'process':
                    files = command.get('files', [])
                    bridge.send_output(f"DEBUG: Received files: {files}, type: {type(files)}")
                    bridge.process_files(files)
                    
                elif action == 'search':
                    query = command.get('query', '')
                    content_types = command.get('content_types')
                    top_k = command.get('top_k', 20)
                    date_filter = command.get('date_filter', '')
                    location_filter = command.get('location_filter', '')
                    offset = command.get('offset', 0)
                    bridge.send_output(f"Received search command: query='{query}', content_types={content_types}, top_k={top_k}, date_filter='{date_filter}', location_filter='{location_filter}', offset={offset}")
                    bridge.search_videos(query, content_types, top_k, date_filter, location_filter, offset)
                    
                elif action == 'chat':
                    message = command.get('message', '')
                    bridge.ai_chat(message)
                    
                elif action == 'status':
                    bridge.get_system_status()
                    
                elif action == 'stop':
                    bridge.send_output("🛑 Setting stop flag...")
                    bridge.processing_stopped = True
                    bridge.send_output("Processing stop flag set")
                    
                elif action == 'stop_enhanced':
                    bridge.send_output("🛑 Enhanced stop command received...")
                    bridge.stop_processing_enhanced()
                    
                elif action == 'error':
                    bridge.send_output(f"Error: {command.get('message', 'Unknown error')}")
                    
                else:
                    bridge.send_output(f"Unknown action: {action}")
                    
            except queue.Empty:
                # No command received, continue loop
                # This allows the loop to continue and check for stop flags even during processing
                time.sleep(0.05)  # Small sleep to prevent excessive CPU usage
                continue
                
    except KeyboardInterrupt:
        bridge.send_output("🛑 Service interrupted by user")
    except Exception as e:
        bridge.send_output(f"❌ Service error: {str(e)}")
        import traceback
        bridge.send_output(f"Traceback: {traceback.format_exc()}")
    
    bridge.send_output("🔚 ContentCache service shutting down")

if __name__ == "__main__":
    main()
