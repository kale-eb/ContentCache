"""
ContentCache API Client

This module provides a client interface to communicate with the ContentCache FastAPI server,
replacing direct API calls to OpenAI, Moondream, Google Maps, etc.
"""

import requests
import base64
import json
import os
import time
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

class ContentCacheAPIClient:
    def __init__(self, base_url: str = "https://contentcache-production.up.railway.app"):
        """
        Initialize the API client
        
        Args:
            base_url: Base URL for the ContentCache API server
        """
        self.base_url = base_url.rstrip('/')
        self.logger = logging.getLogger(__name__)
        self.stop_callback = None  # Will be set by the processing system
        
        # Test connection on initialization
        try:
            status = self.health_check()
            print(f"✅ API Client initialized: {status}")
        except Exception as e:
            print(f"⚠️ API Client initialized but server unreachable: {e}")
        
        self.session = requests.Session()
        logger.info(f"ContentCache API Client initialized with base URL: {self.base_url}")
        
        # Set timeout for all requests
        self.session.timeout = 30
    
    def set_stop_callback(self, callback):
        """Set the callback function to call when processing should be stopped due to API failures"""
        self.stop_callback = callback
    
    def _trigger_stop_processing(self, reason: str):
        """Trigger stop processing by calling the stop callback if available"""
        try:
            if self.stop_callback and callable(self.stop_callback):
                print(f"🛑 Calling stop callback: {reason}")
                self.stop_callback()
            else:
                print(f"⚠️ No stop callback available to trigger stop processing: {reason}")
        except Exception as e:
            print(f"❌ Error triggering stop processing: {e}")
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[Any, Any]:
        """Make HTTP request to the API server with retry logic."""
        url = f"{self.base_url}{endpoint}"
        
        print(f"🌐 Making {method} request to: {url}")
        
        max_retries = 2
        retry_delay = 10  # Updated to 10 seconds
        
        for attempt in range(max_retries + 1):  # 0, 1, 2 (3 total attempts)
        try:
            # Add a reasonable timeout if not specified
            if 'timeout' not in kwargs:
                kwargs['timeout'] = 30
                
            response = requests.request(method, url, **kwargs)
            
            print(f"📡 Response status: {response.status_code}")
                print(f"📊 Response size: {len(response.content) if response.content else 0} bytes")
            
                if response.status_code == 200:
            return response.json()
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                    if attempt < max_retries:
                        print(f"⚠️ API request failed (attempt {attempt + 1}/{max_retries + 1}): {error_msg}")
                        print(f"⏱️ Waiting {retry_delay} seconds before retry...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        # Final failure - trigger stop processing
                        print(f"❌ API request failed after {max_retries + 1} attempts: {error_msg}")
                        print(f"🛑 Triggering stop processing due to repeated API failures")
                        self._trigger_stop_processing("API failed after 3 attempts")
                        raise requests.exceptions.RequestException(f"API failed after {max_retries + 1} attempts: {error_msg}")
            
        except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    print(f"⚠️ API request failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
                    print(f"⏱️ Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    continue
                else:
                    # Final failure - trigger stop processing
                    print(f"❌ API request failed after {max_retries + 1} attempts: {str(e)}")
                    print(f"🛑 Triggering stop processing due to repeated API failures")
                    self._trigger_stop_processing("API connection failed after 3 attempts")
                    raise
            except Exception as e:
                if attempt < max_retries:
                    print(f"⚠️ API request failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
                    print(f"⏱️ Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    continue
                else:
                    # Final failure - trigger stop processing
                    print(f"❌ API request failed after {max_retries + 1} attempts: {str(e)}")
                    print(f"🛑 Triggering stop processing due to repeated API failures")
                    self._trigger_stop_processing("Unexpected API error after 3 attempts")
                    raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check API server health and service availability."""
        return self._make_request('GET', '/health')
    
    # ============================================================================
    # OPENAI METHODS
    # ============================================================================
    
    def openai_chat(self, messages: List[Dict[str, Any]], model: str = "gpt-4o-mini", 
                   tools: Optional[List[Dict[str, Any]]] = None, 
                   tool_choice: Optional[Any] = None,
                   max_tokens: Optional[int] = None,
                   temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Make OpenAI chat completion request via API server.
        
        This replaces direct openai.chat.completions.create() calls.
        """
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        return self._make_request('POST', '/api/openai/chat', json=payload)
    
    def openai_vision_frame_analysis(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Analyze video frames using GPT-4o vision with built-in prompt via API server.
        
        Args:
            image_paths: List of file paths to images
            
        Returns:
            Dict with 'analysis' and 'usage' keys
        """
        # Encode images to base64
        images_base64 = []
        for image_path in image_paths:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                encoded = base64.b64encode(image_data).decode('utf-8')
                images_base64.append(encoded)
        
        payload = {
            "images_base64": images_base64,
            "num_frames": len(image_paths)
        }
        
        return self._make_request('POST', '/api/openai/vision-frame-analysis', json=payload)
    
    def openai_video_summary(self, frame_captions: List[str], audio_summary: Any,
                            video_metadata: Dict[str, Any], vision_analysis: Optional[str] = None,
                            text_data: Optional[Dict[str, Any]] = None, 
                            processed_location: Optional[Union[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive video summary using GPT-4o with built-in prompt and function calling.
        
        Args:
            frame_captions: List of frame captions 
            audio_summary: Summarized audio analysis result
            video_metadata: Video metadata from ffprobe
            vision_analysis: Optional enhanced vision analysis
            text_data: Optional OCR text data
            processed_location: Optional pre-processed location info
            
        Returns:
            Dict containing the structured video analysis result
        """
        payload = {
            "frame_captions": frame_captions,
            "audio_summary": audio_summary,
            "video_metadata": video_metadata,
            "vision_analysis": vision_analysis,
            "text_data": text_data,
            "processed_location": processed_location
        }
        
        response = self._make_request("POST", "/api/openai/video-summary", json=payload)
        return response
    
    def openai_transcribe_audio(self, audio_data: bytes, file_format: str = "mp3") -> Dict[str, Any]:
        """
        Transcribe audio using Whisper via API server.
        
        Args:
            audio_data: Raw audio bytes
            file_format: Audio format (mp3, wav, etc.)
            
        Returns:
            Dict with 'transcript' key
        """
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        payload = {
            "audio_base64": audio_base64,
            "file_format": file_format
        }
        
        return self._make_request('POST', '/api/openai/transcribe', json=payload)
    
    def openai_audio_analysis(self, segments: List[Dict[str, Any]], 
                             filename_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze audio segments using GPT-4o with built-in prompt via API server.
        
        Args:
            segments: List of audio segment data
            filename_info: Optional filename context information
            
        Returns:
            Dict with 'result' and 'usage' keys
        """
        payload = {
            "segments": segments,
            "filename_info": filename_info
        }
        
        return self._make_request('POST', '/api/openai/audio-analysis', json=payload)
    
    def openai_image_analysis(self, image_path: str, ocr_text_data: Optional[Dict[str, Any]] = None,
                            processed_location: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze single image using GPT-4o vision with built-in prompt via API server.
        
        Args:
            image_path: Path to image file
            ocr_text_data: Optional OCR text data
            processed_location: Optional location information
            
        Returns:
            Dict with 'analysis' and 'usage' keys
        """
        # Encode image to base64
        with open(image_path, 'rb') as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        payload = {
            "image_base64": image_base64,
            "ocr_text_data": ocr_text_data,
            "processed_location": processed_location
        }
        
        return self._make_request('POST', '/api/openai/image-analysis', json=payload)
    
    def openai_image_summary(self, caption: str, objects: List[str], filename: str,
                           coordinates: Optional[Dict[str, float]] = None,
                           included_description: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive image summary using GPT-4o with built-in prompt via API server.
        
        Args:
            caption: Image caption from Moondream
            objects: List of detected objects from Moondream
            filename: Image filename
            coordinates: Optional GPS coordinates dict with latitude/longitude
            included_description: Optional description from image metadata
            
        Returns:
            Dict with 'result' and 'usage' keys
        """
        payload = {
            "caption": caption,
            "objects": objects,
            "filename": filename,
            "coordinates": coordinates,
            "included_description": included_description
        }
        
        return self._make_request('POST', '/api/openai/image-summary', json=payload)
    
    def openai_text_analysis(self, file_path: str, text_content: str) -> Dict[str, Any]:
        """
        Analyze text documents using GPT-4o with built-in prompt via API server.
        
        Args:
            file_path: Path to text file
            text_content: Extracted text content
            
        Returns:
            Dict with 'result' and 'usage' keys
        """
        payload = {
            "file_path": file_path,
            "text_content": text_content
        }
        
        return self._make_request('POST', '/api/openai/text-analysis', json=payload)
    
    def parse_search_query(self, query: str) -> Dict[str, Any]:
        """
        Parse search query to extract semantic components using OpenAI.
        
        Args:
            query (str): Search query to parse
            
        Returns:
            Dict[str, Any]: Parsed components with search_query, location, and date filters
        """
        payload = {
            'query': query
        }
        
        return self._make_request('POST', '/api/openai/parse-search-query', json=payload)
    
    # ============================================================================
    # MOONDREAM METHODS
    # ============================================================================
    
    def moondream_analysis(self, image_base64: str) -> Dict[str, Any]:
        """
        Analyze image with Moondream to get caption and prominent objects.
        
        Args:
            image_base64: Base64 encoded image
            
        Returns:
            Dict containing caption and objects list
        """
        payload = {"image_base64": image_base64}
        return self._make_request("POST", "/api/moondream/image-analysis", json=payload)
    
    def moondream_caption(self, image_path: str, prompt: str = "Describe this image in detail.") -> Dict[str, Any]:
        """
        Generate caption for single image using Moondream API.
        
        Args:
            image_path: Path to image file
            prompt: Caption prompt
            
        Returns:
            Dict with 'caption' key
        """
        with open(image_path, 'rb') as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        payload = {
            "image_base64": image_base64,
            "prompt": prompt
        }
        
        return self._make_request('POST', '/api/moondream/caption', json=payload)
    
    def moondream_batch_process(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Process multiple images with Moondream API.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Dict with 'results' key containing list of results
        """
        images_base64 = []
        for image_path in image_paths:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                encoded = base64.b64encode(image_data).decode('utf-8')
                images_base64.append(encoded)
        
        payload = {"images_base64": images_base64}
        
        return self._make_request('POST', '/api/moondream/batch', json=payload)
    
    # ============================================================================
    # GOOGLE MAPS METHODS
    # ============================================================================
    
    def google_reverse_geocode(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Reverse geocode coordinates using Google Maps API.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Google Maps API response
        """
        payload = {
            "latitude": latitude,
            "longitude": longitude
        }
        
        return self._make_request('POST', '/api/google/reverse-geocode', json=payload)
    
    def google_nearby_search(self, latitude: float, longitude: float, radius: int = 500) -> Dict[str, Any]:
        """
        Search for nearby places using Google Maps API.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            radius: Search radius in meters
            
        Returns:
            Google Maps API response
        """
        payload = {
            "latitude": latitude,
            "longitude": longitude,
            "radius": radius
        }
        
        return self._make_request('POST', '/api/google/nearby-search', json=payload)
    
    def google_forward_geocode(self, location_text: str) -> Dict[str, Any]:
        """
        Forward geocode location text using Google Maps API.
        
        Args:
            location_text (str): Location text to geocode
            
        Returns:
            Dict[str, Any]: Google Maps API response with coordinates
        """
        payload = {
            'location_text': location_text
        }
        
        return self._make_request('POST', '/api/google/forward-geocode', json=payload)



# ============================================================================
# UTILITY FUNCTIONS FOR EASY MIGRATION
# ============================================================================

# Global client instance for easy migration
_api_client = None

def get_api_client() -> ContentCacheAPIClient:
    """Get or create global API client instance."""
    global _api_client
    if _api_client is None:
        _api_client = ContentCacheAPIClient()
    return _api_client

def use_api_server(base_url: str = None):
    """
    Configure ContentCache to use API server instead of direct API calls.
    
    Args:
        base_url: API server URL. If None, uses environment variable or localhost.
    """
    global _api_client
    
    # Use environment variable or default if base_url is None
    if base_url is None:
        base_url = os.getenv("CONTENTCACHE_API_URL", "https://contentcache-production.up.railway.app")
    
    _api_client = ContentCacheAPIClient(base_url)
    logger.info(f"ContentCache configured to use API server at: {_api_client.base_url}")

# Convenience functions that mirror the original API patterns
def call_openai_chat(**kwargs):
    """Convenience function for OpenAI chat - mirrors original usage."""
    return get_api_client().openai_chat(**kwargs)

def call_gpt4o_vision_frame_analysis(image_paths: List[str], **kwargs):
    """Convenience function for GPT-4o vision frame analysis with built-in prompt."""
    return get_api_client().openai_vision_frame_analysis(image_paths, **kwargs)

def call_gpt4o_video_summary(frame_captions: List[str], audio_summary: Any, 
                           video_metadata: Dict[str, Any], **kwargs):
    """Convenience function for GPT-4o video summary with built-in prompt."""
    return get_api_client().openai_video_summary(frame_captions, audio_summary, video_metadata, **kwargs)

def call_moondream_api(image_path: str, **kwargs):
    """Convenience function for Moondream - mirrors original usage."""
    return get_api_client().moondream_caption(image_path, **kwargs)

def call_google_maps_geocode(lat: float, lon: float):
    """Convenience function for Google Maps geocoding - mirrors original usage."""
    return get_api_client().google_reverse_geocode(lat, lon)

def call_gpt4o_audio_analysis(segments: List[Dict[str, Any]], **kwargs):
    """Convenience function for GPT-4o audio analysis with built-in prompt."""
    return get_api_client().openai_audio_analysis(segments, **kwargs)

def call_gpt4o_image_analysis(image_path: str, **kwargs):
    """Convenience function for GPT-4o image analysis with built-in prompt."""
    return get_api_client().openai_image_analysis(image_path, **kwargs)

def call_gpt4o_image_summary(caption: str, objects: List[str], filename: str, **kwargs):
    """Convenience function for GPT-4o image summary with built-in prompt."""
    return get_api_client().openai_image_summary(caption, objects, filename, **kwargs)

def call_gpt4o_text_analysis(file_path: str, text_content: str, **kwargs):
    """Convenience function for GPT-4o text analysis with built-in prompt."""
    return get_api_client().openai_text_analysis(file_path, text_content, **kwargs) 