#!/usr/bin/env python3
import sys
import json
import os
from pathlib import Path

# Import the unified service
from unified_service import ContentCacheService

class ElectronBridge:
    """Bridge between Electron frontend and ContentCache unified service"""
    
    def __init__(self):
        # Create the unified service with progress callback
        self.service = ContentCacheService(progress_callback=self.progress_callback)
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
    
    def send_json_response(self, data: dict):
        """Send JSON response to Electron"""
        print(json.dumps(data), flush=True)
    
    def send_ai_response(self, message: str):
        """Send AI chat response to Electron"""
        response = {
            "type": "ai_response",
            "message": message
        }
        self.send_json_response(response)
    
    def process_files(self, file_paths: list):
        """Process multiple files using the unified service"""
        results = []
        total_files = len(file_paths)
        
        self.send_output(f"Starting to process {total_files} file(s)")
        
        for i, file_path in enumerate(file_paths):
            try:
                if os.path.isdir(file_path):
                    # Process directory
                    self.send_output(f"Processing directory: {file_path}")
                    result = self.service.process_directory(file_path)
                else:
                    # Process single file
                    self.send_output(f"Processing file: {os.path.basename(file_path)}")
                    result = self.service.process_file(file_path)
                
                results.append(result)
                
                # Send progress update
                progress = ((i + 1) / total_files) * 100
                self.send_json_response({
                    "type": "batch_progress",
                    "progress": progress,
                    "completed": i + 1,
                    "total": total_files,
                    "current_file": os.path.basename(file_path)
                })
                
            except Exception as e:
                error_result = {
                    "file_path": file_path,
                    "error": str(e),
                    "success": False
                }
                results.append(error_result)
                self.send_output(f"Error processing {file_path}: {str(e)}")
        
        # Send final results
        self.send_json_response({
            "type": "processing_complete",
            "results": results,
            "total_processed": len(results),
            "successful": len([r for r in results if r.get('success', False)])
        })
        
        return results
    
    def search_videos(self, query: str, content_types: list = None, top_k: int = 20, 
                     date_filter: str = '', location_filter: str = ''):
        """Search content using the unified service with optional filters"""
        self.send_output(f"Searching for: '{query}' with content_types={content_types}, top_k={top_k}, date_filter='{date_filter}', location_filter='{location_filter}'")
        
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
            
            # Perform search with filters
            self.send_output("Calling search_content...")
            results = self.service.search_content(query, content_types, top_k, date_filter, location_filter)
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
                    # Return flat results
                    self.send_json_response({
                        "type": "search_results",
                        "query": query,
                        "results": results.get('results', []),
                            "has_buckets": False,
                        "total_found": len(results.get('results', []))
                    })
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

def main():
    """Main loop for handling Electron commands"""
    bridge = ElectronBridge()
    
    # Send initial status
    bridge.get_system_status()
    
    # Listen for commands from Electron
    for line in sys.stdin:
        try:
            command = json.loads(line.strip())
            action = command.get('action')
            
            if action == 'process':
                files = command.get('files', [])
                bridge.process_files(files)
                
            elif action == 'search':
                query = command.get('query', '')
                content_types = command.get('content_types')
                top_k = command.get('top_k', 20)
                date_filter = command.get('date_filter', '')
                location_filter = command.get('location_filter', '')
                bridge.send_output(f"Received search command: query='{query}', content_types={content_types}, top_k={top_k}, date_filter='{date_filter}', location_filter='{location_filter}'")
                bridge.search_videos(query, content_types, top_k, date_filter, location_filter)
                
            elif action == 'chat':
                message = command.get('message', '')
                bridge.ai_chat(message)
                
            elif action == 'status':
                bridge.get_system_status()
                
            else:
                bridge.send_output(f"Unknown action: {action}")
                
        except json.JSONDecodeError as e:
            bridge.send_output(f"Error: Invalid JSON command - {str(e)}")
        except Exception as e:
            bridge.send_output(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
