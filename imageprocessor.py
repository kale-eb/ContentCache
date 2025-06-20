import os
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple
import logging
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import torch
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification, pipeline
import clip
from rembg import remove
# BLIP imports commented out - using Moondream API only
# from framestagging import processor, model as blip_model
import requests
from pathlib import Path
import base64
import openai
from dotenv import load_dotenv
from datetime import datetime, timezone

import easyocr
from location_utils import process_location_from_metadata
from config import get_image_metadata_path, get_models_cache_dir

# Load environment variables
load_dotenv()

# Set up model cache directory using config
MODEL_CACHE_DIR = get_models_cache_dir()
Path(MODEL_CACHE_DIR).mkdir(exist_ok=True)

# Initialize global OCR readers for standalone functions
try:
    global_easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    print("✅ Global EasyOCR initialized for image analysis")
except Exception as e:
    global_easyocr_reader = None
    logging.warning(f"Could not initialize global EasyOCR: {e}")

# ============================================================================
# STANDALONE OCR FUNCTIONS
# ============================================================================

def extract_text_ocr_standalone(image_path):
    """
    Standalone OCR text extraction function that doesn't require ImageProcessor instance.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Extracted text information
    """
    text_results = {
        'prominent_text': [],
        'all_text': [],
        'text_confidence': [],
        'ocr_method': 'none'
    }
    
    try:
        # Load image
        pil_image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert PIL to numpy array
        image_np = np.array(pil_image)
        
        # Try EasyOCR first (generally more accurate)
        if global_easyocr_reader:
            try:
                results = global_easyocr_reader.readtext(image_np)
                
                all_text = []
                prominent_text = []
                confidences = []
                
                for (bbox, text, confidence) in results:
                    if confidence > 0.3:  # Filter low confidence detections
                        all_text.append(text.strip())
                        confidences.append(confidence)
                        
                        # Consider text "prominent" if confidence > 0.6 and length > 2
                        if confidence > 0.6 and len(text.strip()) > 2:
                            prominent_text.append(text.strip())
                
                text_results.update({
                    'prominent_text': prominent_text,
                    'all_text': all_text,
                    'text_confidence': confidences,
                    'ocr_method': 'easyocr'
                })
                
                print(f"    📖 EasyOCR found {len(all_text)} text elements, {len(prominent_text)} prominent")
                
            except Exception as e:
                logging.warning(f"EasyOCR failed: {e}")
        

        
        return text_results
        
    except Exception as e:
        logging.error(f"Text extraction failed: {e}")
        return text_results

# ============================================================================
# LOCATION PROCESSING FUNCTIONS FOR IMAGES
# ============================================================================

def extract_gps_from_exif(image_path):
    """
    Extract GPS coordinates from image EXIF data.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: GPS coordinates string or None if not found
    """
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            
            if not exif_data:
                return None
            
            # Look for GPS info
            gps_info = None
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name == 'GPSInfo':
                    gps_info = value
                    break
            
            if not gps_info:
                return None
            
            # Extract GPS coordinates
            def get_decimal_from_dms(dms, ref):
                """Convert degrees, minutes, seconds to decimal degrees"""
                degrees = dms[0]
                minutes = dms[1] / 60.0
                seconds = dms[2] / 3600.0
                
                decimal = degrees + minutes + seconds
                if ref in ['S', 'W']:
                    decimal = -decimal
                return decimal
            
            # Get GPS data
            gps_latitude = gps_info.get(2)  # GPSLatitude
            gps_latitude_ref = gps_info.get(1)  # GPSLatitudeRef
            gps_longitude = gps_info.get(4)  # GPSLongitude
            gps_longitude_ref = gps_info.get(3)  # GPSLongitudeRef
            
            if gps_latitude and gps_longitude and gps_latitude_ref and gps_longitude_ref:
                lat = get_decimal_from_dms(gps_latitude, gps_latitude_ref)
                lon = get_decimal_from_dms(gps_longitude, gps_longitude_ref)
                return f"{lat}, {lon}"
            
            return None
            
    except Exception as e:
        logging.debug(f"Could not extract GPS from {image_path}: {e}")
        return None

def convert_location_to_readable(location_text):
    """
    DEPRECATED: Use location_utils.process_location_from_metadata() instead.
    This function is kept for compatibility but now delegates to Google Maps processing.
    """
    print(f"⚠️ Using deprecated convert_location_to_readable() - consider updating to use location_utils directly")
    return process_location_from_metadata(location_text)

def extract_and_store_image_location_coordinates(image_path):
    """
    Extract location coordinates from image EXIF data and store them as raw coordinates.
    No longer converts to readable text - coordinates will be used for proximity search.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Location data with coordinates or None if no location found
    """
    # Extract GPS coordinates from EXIF
    gps_coords = extract_gps_from_exif(image_path)
    
    if gps_coords:
        try:
            # Parse the GPS coordinates string (e.g., "37.7749, -122.4194")
            import re
            coord_pattern = r'([+-]?\d+\.?\d*)[,\s]*([+-]?\d+\.?\d*)'
            match = re.search(coord_pattern, gps_coords)
            
            if match:
                lat, lon = float(match.group(1)), float(match.group(2))
                
                # Validate coordinate ranges
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    location_data = {
                        'type': 'coordinates',
                        'latitude': lat,
                        'longitude': lon,
                        'raw_string': gps_coords
                    }
                    print(f"📍 Image coordinates extracted: {lat}, {lon}")
                    return location_data
                else:
                    print(f"⚠️ Invalid coordinate ranges: lat={lat}, lon={lon}")
                    return None
            else:
                print(f"⚠️ Could not parse GPS coordinates: {gps_coords}")
                return None
                
        except Exception as e:
            print(f"⚠️ Error processing GPS coordinates: {e}")
            return None
    
    return None

# ============================================================================
# ENHANCED VISUAL ANALYSIS WITH GPT-4O MINI VISION API FOR SINGLE IMAGES
# ============================================================================

def encode_image_to_base64(image_path, resize=True, max_dimension=180, quality=50, show_preview=False):
    """
    Encode image to base64 for OpenAI API with compression to reduce token costs.
    
    Args:
        image_path (str): Path to the image file
        resize (bool): Whether to resize the image (default True for cost savings)
        max_dimension (int): Maximum size for the larger dimension (width or height)
        quality (int): JPEG quality (1-100, lower = more compression)
        show_preview (bool): Whether to display the image being sent to API
        
    Returns:
        str: Base64 encoded compressed image
    """
    try:
        import io
        
        # Open and process the image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Store original dimensions for logging
            original_dimensions = img.size
            
            # Resize based on the larger dimension
            if resize:
                width, height = img.size
                if width > height:
                    # Width is larger, scale based on width
                    new_width = max_dimension
                    new_height = int((height * max_dimension) / width)
                else:
                    # Height is larger (or equal), scale based on height
                    new_height = max_dimension
                    new_width = int((width * max_dimension) / height)
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                resize_info = f" resized from {original_dimensions} to {img.size}"
            else:
                resize_info = f" kept at {img.size}"
            
            # Show preview if requested (temporary feature)
            if show_preview:
                try:
                    # Create a copy for display
                    display_img = img.copy()
                    # Resize for display if too large
                    display_img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                    display_img.show(title=f"Image being sent to API: {os.path.basename(image_path)}")
                    print(f"    👁️  Displaying preview of {os.path.basename(image_path)}")
                except Exception as e:
                    print(f"    ⚠️  Could not display preview: {e}")
            
            # Compress to JPEG in memory with optimization
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            buffer.seek(0)
            
            # Encode to base64
            compressed_data = base64.b64encode(buffer.read()).decode('utf-8')
            
            # Log compression info
            original_size = os.path.getsize(image_path)
            compressed_size = len(compressed_data) * 3 // 4  # Approximate size after base64 decoding
            compression_ratio = (1 - compressed_size / original_size) * 100
            
            print(f"    🗜️  Compressed {os.path.basename(image_path)}: {original_size//1024}KB → {compressed_size//1024}KB ({compression_ratio:.1f}% reduction){resize_info}")
            
            return compressed_data
            
    except Exception as e:
        logging.error(f"Error encoding/compressing image {image_path}: {e}")
        return None

def analyze_single_image_with_vision_api(image_path):
    """
    Analyze a single image using GPT-4o mini vision API.
    Provides detailed analysis including text detection and object identification.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Analysis results with summary, objects, text, etc.
    """
    if not os.path.exists(image_path):
        logging.error(f"Image file not found: {image_path}")
        return None
    
    print(f"🔍 Analyzing image with GPT-4o mini vision: {os.path.basename(image_path)}")
    
    # First extract text using OCR at full resolution for better accuracy
    print("📖 Extracting text with OCR before compression...")
    ocr_text_data = extract_text_ocr_standalone(image_path)
    
    # Extract and process location information
    print("📍 Processing location information...")
    processed_location = extract_and_store_image_location_coordinates(image_path)
    if processed_location:
        print(f"✓ Location processed: {processed_location}")
    else:
        print("✓ No location information found in image EXIF")
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Encode image to base64 (with resizing and preview display)
    base64_image = encode_image_to_base64(image_path, resize=True, max_dimension=512, quality=75, show_preview=False)
    if not base64_image:
        return None
    
    # Prepare the adaptive analysis prompt with OCR text if available
    analysis_prompt = (
        "Analyze this image efficiently. Skip categories that don't add value.\n\n"
        "Format (skip irrelevant sections):\n"
        "1. Summary: [1-3 sentence(s) describing the main content and scene]\n"
        "2. Objects: [key items, people] (keywords only)\n"
        "3. Text: [meaningful visible text ONLY - skip if garbled/unclear]\n"
        "4. Setting: [location type, environment] (keywords only)\n"
        "5. Activities: [main actions] (keywords only)\n"
        "6. Mood: [atmosphere keywords] (keywords only)\n"
        "7. Style: [visual characteristics] (keywords only)\n"
        "8. Details: [notable elements] (keywords only)\n\n"
    )
    
    # Add OCR text information if available and meaningful
    if ocr_text_data and ocr_text_data.get('prominent_text'):
        prominent_text = ', '.join(ocr_text_data['prominent_text'])
        # Only include OCR if it contains meaningful words (not just random characters)
        meaningful_words = [word for word in prominent_text.split() if len(word) > 2 and word.isalpha()]
        if len(meaningful_words) > 0:
            analysis_prompt += (
                f"IMPORTANT: OCR detected meaningful text: {prominent_text}\n"
                f"Include this in section 3 if clearly readable. If the OCR text appears garbled or meaningless, skip section 3 entirely.\n\n"
            )
    
    # Add location information if available
    if processed_location:
        analysis_prompt += (
            f"LOCATION: This image was taken at: {processed_location}\n"
            f"Please incorporate this location information into your analysis, especially for section 4 (setting/location).\n\n"
        )
    
    analysis_prompt += "Be thorough and specific in your analysis."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an efficient image analyst focused on extracting meaningful keywords. "
                        "Provide concise, keyword-based analysis that maximizes information density while minimizing token usage. "
                        "Skip verbose descriptions and garbled text. Use comma-separated keywords for categories. "
                        "Only include categories that add genuine value to understanding the image content."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": analysis_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=800
        )
        
        analysis_text = response.choices[0].message.content
        print("✅ Vision analysis complete")
        
        # Parse the analysis into structured format
        return parse_vision_analysis(analysis_text, image_path, processed_location)
        
    except Exception as e:
        logging.error(f"Error in vision API call for {image_path}: {e}")
        return None

def parse_vision_analysis(analysis_text, image_path, processed_location=None):
    """
    Parse the vision analysis text into structured metadata format.
    
    Args:
        analysis_text (str): Raw analysis text from GPT-4o mini
        image_path (str): Path to the analyzed image
        processed_location (str, optional): Pre-processed location from EXIF data
        
    Returns:
        dict: Structured metadata
    """
    try:
        # Get basic file info
        file_stats = os.stat(image_path)
        
        # Try to extract structured information from the analysis
        # This is a simplified parser - could be enhanced with more sophisticated NLP
        lines = analysis_text.split('\n')
        
        summary = ""
        objects = []
        visible_text = []
        activities = []
        mood = []
        location = ""
        
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Identify sections based on keywords
            if any(keyword in line.lower() for keyword in ['summary', 'description', 'shows', 'depicts']):
                current_section = 'summary'
                if ':' in line:
                    summary += line.split(':', 1)[1].strip() + " "
            elif any(keyword in line.lower() for keyword in ['objects', 'items', 'things']):
                current_section = 'objects'
            elif any(keyword in line.lower() for keyword in ['text', 'writing', 'sign', 'words']):
                current_section = 'text'
            elif any(keyword in line.lower() for keyword in ['activities', 'actions', 'doing']):
                current_section = 'activities'
            elif any(keyword in line.lower() for keyword in ['mood', 'atmosphere', 'feeling']):
                current_section = 'mood'
            elif any(keyword in line.lower() for keyword in ['location', 'setting', 'place']):
                current_section = 'location'
                if ':' in line:
                    location = line.split(':', 1)[1].strip()
            else:
                # Add content to current section
                if current_section == 'summary' and not any(char in line for char in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.']):
                    summary += line + " "
        
        # Clean up summary
        summary = summary.strip()
        if not summary:
            summary = analysis_text[:200] + "..." if len(analysis_text) > 200 else analysis_text
        
        # Create structured metadata
        metadata = {
            'summary': summary,
            'detailed_analysis': analysis_text,
            'objects': objects if objects else ['various objects'],  # Fallback
            'visible_text': visible_text,
            'activities': activities,
            'mood': mood if mood else ['neutral'],  # Fallback
            'location': processed_location if processed_location else (location if location else 'None'),
            'file_type': os.path.splitext(image_path)[1].lower(),
            'file_size': file_stats.st_size,
            'last_modified': file_stats.st_mtime,
            'analysis_method': 'gpt4o_vision'
        }
        
        return metadata
        
    except Exception as e:
        logging.error(f"Error parsing vision analysis: {e}")
        return {
            'summary': analysis_text[:200] + "..." if len(analysis_text) > 200 else analysis_text,
            'detailed_analysis': analysis_text,
            'objects': ['unknown'],
            'analysis_method': 'gpt4o_vision',
            'error': str(e)
        }

class ImageProcessor:
    def __init__(self):
        self.metadata_file = 'image_metadata.json'
        self.metadata = self._load_metadata()
        
        # Initialize CLIP model with caching
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device, download_root=MODEL_CACHE_DIR)
        
        # Initialize ViT model for classification with caching
        self.vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224', cache_dir=MODEL_CACHE_DIR)
        self.vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', cache_dir=MODEL_CACHE_DIR)
        self.vit_model.to(self.device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Object detection pipeline with caching
        try:
            # Suppress specific DETR model warnings
            import warnings
            from transformers import logging as transformers_logging
            
            # Temporarily suppress transformers warnings
            transformers_logging.set_verbosity_error()
            
            # Suppress the specific warnings about unused weights and slow processor
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Some weights of the model checkpoint.*were not used")
                warnings.filterwarnings("ignore", message=".*This IS expected if you are initializing.*")
                warnings.filterwarnings("ignore", message=".*This IS NOT expected if you are initializing.*")
                warnings.filterwarnings("ignore", message="Using a slow image processor.*")
                warnings.filterwarnings("ignore", message=".*use_fast.*")
                
                self.object_detector = pipeline(
                    "object-detection", 
                    model="facebook/detr-resnet-50", 
                    cache_dir=MODEL_CACHE_DIR,
                    device=self.device,  # Explicitly set device to avoid MPS warnings
                    # Use fast processor to avoid the slow processor warning
                    feature_extractor=None,  # Let it auto-detect
                    tokenizer=None,  # Not needed for object detection
                )
            
            # Restore normal logging level
            transformers_logging.set_verbosity_warning()
            
        except Exception as e:
            self.object_detector = None
            logging.warning(f"Could not load object detection model: {e}")
        
        # Initialize OCR readers
        try:
            self.easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            print("✅ EasyOCR initialized successfully")
        except Exception as e:
            self.easyocr_reader = None
            logging.warning(f"Could not initialize EasyOCR: {e}")
        
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        self.moondream_api_key = os.getenv("MOONDREAM_API_KEY")
        
        if not self.moondream_api_key:
            print("⚠️ MOONDREAM_API_KEY not found in environment variables")
        else:
            print(f"✅ Moondream API key loaded successfully")

    def _load_metadata(self) -> Dict:
        """Load existing metadata or create new if not exists."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logging.error("Error loading metadata file")
                return {}
        return {}

    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _extract_color_features(self, image: np.ndarray) -> Dict:
        """Extract color-based features from image."""
        try:
            # Convert to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calculate color histograms
            h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            
            # Normalize histograms
            h_hist = cv2.normalize(h_hist, h_hist).flatten()
            s_hist = cv2.normalize(s_hist, s_hist).flatten()
            v_hist = cv2.normalize(v_hist, v_hist).flatten()
            
            # Calculate dominant colors
            pixels = hsv.reshape(-1, 3)
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(pixels)
            dominant_colors = kmeans.cluster_centers_
            
            return {
                'h_histogram': h_hist.tolist(),
                's_histogram': s_hist.tolist(),
                'v_histogram': v_hist.tolist(),
                'dominant_colors': dominant_colors.tolist()
            }
        except Exception as e:
            logging.error(f"Error extracting color features: {str(e)}")
            return {}

    def _extract_texture_features(self, image: np.ndarray) -> Dict:
        """Extract texture-based features from image."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate GLCM (Gray-Level Co-occurrence Matrix)
            from skimage.feature import graycomatrix, graycoprops
            glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
            
            # Calculate texture properties
            contrast = graycoprops(glcm, 'contrast').flatten()
            dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
            homogeneity = graycoprops(glcm, 'homogeneity').flatten()
            energy = graycoprops(glcm, 'energy').flatten()
            correlation = graycoprops(glcm, 'correlation').flatten()
            
            return {
                'contrast': contrast.tolist(),
                'dissimilarity': dissimilarity.tolist(),
                'homogeneity': homogeneity.tolist(),
                'energy': energy.tolist(),
                'correlation': correlation.tolist()
            }
        except Exception as e:
            logging.error(f"Error extracting texture features: {str(e)}")
            return {}

    def _extract_clip_features(self, image: Image.Image) -> np.ndarray:
        """Extract features using CLIP model."""
        try:
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
            return image_features.cpu().numpy()
        except Exception as e:
            logging.error(f"Error extracting CLIP features: {str(e)}")
            return np.array([])

    def _extract_vit_features(self, image: Image.Image) -> Dict:
        """Extract features using ViT model."""
        try:
            inputs = self.vit_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.vit_model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
            # Get top 5 predictions
            top5_prob, top5_indices = torch.topk(probs, 5)
            
            return {
                'predictions': [
                    {
                        'label': self.vit_model.config.id2label[idx.item()],
                        'probability': prob.item()
                    }
                    for prob, idx in zip(top5_prob[0], top5_indices[0])
                ]
            }
        except Exception as e:
            logging.error(f"Error extracting ViT features: {str(e)}")
            return {}

    def _caption_image(self, pil_image: Image.Image) -> str:
        # BLIP functionality disabled - using Moondream API in main pipeline
        return "BLIP disabled - using Moondream API in main pipeline"

    def _detect_objects(self, pil_image: Image.Image) -> List[str]:
        if not self.object_detector:
            return []
        try:
            results = self.object_detector(pil_image)
            # Get unique object labels
            objects = list({r['label'] for r in results})
            return objects
        except Exception as e:
            logging.warning(f"Object detection failed: {e}")
            return []

    def _extract_text_ocr(self, image_path: str, pil_image: Image.Image = None) -> Dict:
        """
        Extract text from image using multiple OCR methods.
        
        Args:
            image_path (str): Path to the image file
            pil_image (Image.Image, optional): PIL image object
            
        Returns:
            dict: Extracted text information
        """
        text_results = {
            'prominent_text': [],
            'all_text': [],
            'text_confidence': [],
            'ocr_method': 'none'
        }
        
        try:
            # Convert PIL image to numpy array for OCR processing
            if pil_image is None:
                pil_image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert PIL to numpy array
            image_np = np.array(pil_image)
            
            # Try EasyOCR first (generally more accurate)
            if self.easyocr_reader:
                try:
                    results = self.easyocr_reader.readtext(image_np)
                    
                    all_text = []
                    prominent_text = []
                    confidences = []
                    
                    for (bbox, text, confidence) in results:
                        if confidence > 0.3:  # Filter low confidence detections
                            all_text.append(text.strip())
                            confidences.append(confidence)
                            
                            # Consider text "prominent" if confidence > 0.6 and length > 2
                            if confidence > 0.6 and len(text.strip()) > 2:
                                prominent_text.append(text.strip())
                    
                    text_results.update({
                        'prominent_text': prominent_text,
                        'all_text': all_text,
                        'text_confidence': confidences,
                        'ocr_method': 'easyocr'
                    })
                    
                    print(f"    📖 EasyOCR found {len(all_text)} text elements, {len(prominent_text)} prominent")
                    
                except Exception as e:
                    logging.warning(f"EasyOCR failed: {e}")
            

            
            return text_results
            
        except Exception as e:
            logging.error(f"Text extraction failed: {e}")
            return text_results

    def _summarize_with_openai(self, caption: str, objects: List[str], text_data: Dict = None, processed_location: str = None) -> Dict:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set.")
        
        prompt = (
            "Given the following image caption and detected objects, "
            "provide a concise summary of the image and a list of objects in the image.\n\n"
            f"Caption: {caption}\nObjects: {objects}\n"
        )
        
        # Add OCR text information if available
        if text_data and text_data.get('prominent_text'):
            prominent_text = ', '.join(text_data['prominent_text'])
            prompt += f"Detected Text (OCR): {prominent_text}\n"
            prompt += "Please incorporate any relevant text information into your summary.\n\n"
        
        # Add location information if available
        if processed_location and processed_location != 'None':
            prompt += f"Location: {processed_location}\n"
            prompt += "Please incorporate this location information into your summary where relevant.\n\n"
        openai_function = {
            "type": "function",
            "function": {
                "name": "summarize_image",
                "description": "Summarize image and extract objects.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "A concise summary of the image."
                        },
                        "objects": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "A list of objects detected in the image."
                        }
                    },
                    "required": ["summary", "objects"]
                }
            }
        }
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "tools": [openai_function],
            "tool_choice": {"type": "function", "function": {"name": "summarize_image"}},
            "max_tokens": 400
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        tool_calls = response.json()["choices"][0]["message"].get("tool_calls", [])
        if tool_calls:
            arguments = tool_calls[0]["function"]["arguments"]
            import json as pyjson
            args = pyjson.loads(arguments)
            return args
        else:
            raise ValueError("No tool call found in OpenAI response")

    def get_moondream_caption(self, image_path):
        """
        Use Moondream SDK to caption an image.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            str: Image caption or None if failed
        """
        try:
            # Initialize Moondream SDK if not already done
            if not hasattr(self, '_moondream_model'):
                try:
                    import moondream as md
                    self._moondream_model = md.vl(api_key=self.moondream_api_key)
                    print("✅ Moondream SDK initialized successfully")
                except ImportError:
                    print("❌ Moondream SDK not installed. Install with: pip install moondream")
                    return None
                except Exception as e:
                    print(f"❌ Error initializing Moondream SDK: {e}")
                    return None
            
            # Load image with PIL
            from PIL import Image
            image = Image.open(image_path)
            
            # Generate caption using Moondream SDK
            result = self._moondream_model.caption(image, length="normal")
            caption = result.get("caption", "")
            
            return caption
                
        except Exception as e:
            print(f"❌ Error calling Moondream SDK: {e}")
            return None

    def _get_file_timestamp(self, file_path):
        """
        Get timestamp for file. Prefers EXIF data for images, falls back to file system.
        
        Args:
            file_path (str): Path to file
            
        Returns:
            str: ISO format timestamp (EXIF date if available, otherwise file creation/modification time)
        """
        try:
            # First try to get EXIF timestamp from image
            try:
                from PIL import Image
                from PIL.ExifTags import TAGS
                
                with Image.open(file_path) as img:
                    exif_data = img._getexif()
                    
                    if exif_data:
                        # Look for date/time fields in EXIF
                        for tag, value in exif_data.items():
                            tag_name = TAGS.get(tag, tag)
                            if tag_name in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
                                # Parse EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
                                try:
                                    dt = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                                    # Assume UTC for EXIF timestamps (could be improved with timezone detection)
                                    dt = dt.replace(tzinfo=timezone.utc)
                                    return dt.isoformat()
                                except ValueError:
                                    continue
            except Exception:
                pass  # Fall back to file system timestamps
            
            # Fallback to file system timestamps
            import platform
            
            # Get file stats
            stat = os.stat(file_path)
            
            # Try to get creation time (works on Windows and macOS)
            if platform.system() == 'Windows':
                # Windows: use creation time
                timestamp = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
            elif platform.system() == 'Darwin':  # macOS
                # macOS: use birth time if available, otherwise creation time
                if hasattr(stat, 'st_birthtime'):
                    timestamp = datetime.fromtimestamp(stat.st_birthtime, tz=timezone.utc)
                else:
                    timestamp = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
            else:
                # Linux/Unix: use modification time (creation time not reliably available)
                timestamp = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            
            return timestamp.isoformat()
            
        except Exception as e:
            print(f"⚠️ Could not extract file timestamp: {e}")
            # Fallback to current time if file timestamp extraction fails
            return datetime.now(timezone.utc).isoformat()

    def _serialize_exif_value(self, value):
        """Convert EXIF values to JSON-serializable format."""
        try:
            # Handle IFDRational objects
            if hasattr(value, 'numerator') and hasattr(value, 'denominator'):
                if value.denominator == 0:
                    return None
                return float(value.numerator) / float(value.denominator)
            
            # Handle tuples (like GPS coordinates)
            elif isinstance(value, tuple):
                return [self._serialize_exif_value(item) for item in value]
            
            # Handle lists
            elif isinstance(value, list):
                return [self._serialize_exif_value(item) for item in value]
            
            # Handle bytes
            elif isinstance(value, bytes):
                try:
                    return value.decode('utf-8', errors='ignore')
                except:
                    return str(value)
            
            # For other types, try to convert to basic Python types
            elif isinstance(value, (int, float, str, bool)) or value is None:
                return value
            else:
                # For anything else, convert to string
                return str(value)
                
        except Exception:
            # If all else fails, return string representation
            return str(value) if value is not None else None

    def get_image_metadata(self, image_path):
        """
        Extract basic metadata from image file.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            dict: Image metadata (JSON-serializable)
        """
        try:
            with Image.open(image_path) as img:
                metadata = {
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "width": img.width,
                    "height": img.height,
                    "has_transparency": img.mode in ('RGBA', 'LA') or 'transparency' in img.info
                }
                
                # Extract EXIF data if available and convert to JSON-serializable format
                if hasattr(img, '_getexif') and img._getexif() is not None:
                    exif = img._getexif()
                    if exif:
                        # Common EXIF tags
                        exif_tags = {
                            256: 'ImageWidth',
                            257: 'ImageLength', 
                            271: 'Make',
                            272: 'Model',
                            274: 'Orientation',
                            306: 'DateTime',
                            33434: 'ExposureTime',
                            33437: 'FNumber',
                            34855: 'ISOSpeedRatings'
                        }
                        
                        metadata['exif'] = {}
                        for tag_id, tag_name in exif_tags.items():
                            if tag_id in exif:
                                # Convert EXIF value to JSON-serializable format
                                metadata['exif'][tag_name] = self._serialize_exif_value(exif[tag_id])
                
                return metadata
                
        except Exception as e:
            print(f"⚠️ Error extracting image metadata: {e}")
            return {}

    def analyze_with_gpt4o_mini(self, image_path, moondream_caption, image_metadata):
        """
        Use GPT-4o mini to create structured analysis of image.
        
        Args:
            image_path (str): Path to image file
            moondream_caption (str): Caption from Moondream API
            image_metadata (dict): Image metadata
            
        Returns:
            dict: Structured analysis results
        """
        try:
            # Prepare the prompt
            filename = os.path.basename(image_path)
            prompt = f"""Analyze this image based on the provided caption and metadata.

IMAGE FILE: {filename}
MOONDREAM CAPTION: {moondream_caption}
METADATA: {json.dumps(image_metadata, indent=2)}

Please provide exactly these 4 fields:
1. summary - Brief summary of image content (1-2 sentences)
2. key_topics - Array of keywords representing main topics/themes
3. key_objects - Array of main objects/elements in the image
4. location - Apparent location or setting (if determinable, otherwise "unknown")

Format your response as clean JSON with exactly these field names."""

            # Define the function schema for structured output
            function_schema = {
                "name": "analyze_image",
                "description": "Analyze an image and provide structured metadata",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Brief summary of image content (1-2 sentences)"
                        },
                        "key_topics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Array of keywords representing main topics/themes"
                        },
                        "key_objects": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Array of main objects/elements in the image"
                        },
                        "location": {
                            "type": "string",
                            "description": "Apparent location or setting (if determinable, otherwise 'unknown')"
                        },
                    },
                    "required": ["summary", "key_topics", "key_objects", "location"]
                }
            }

            # Call GPT-4o mini
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                tools=[{"type": "function", "function": function_schema}],
                tool_choice={"type": "function", "function": {"name": "analyze_image"}},
                max_tokens=1000
            )

            # Extract the function call result
            if response.choices[0].message.tool_calls:
                function_args = response.choices[0].message.tool_calls[0].function.arguments
                analysis = json.loads(function_args)
                # Add file timestamp from metadata
                analysis["timestamp"] = self._get_file_timestamp(image_path)
                return analysis
            else:
                print("⚠️ No function call in GPT-4o mini response")
                return None

        except Exception as e:
            print(f"❌ Error calling GPT-4o mini: {e}")
            return None

    def process_file(self, image_path):
        """
        Process a single image file completely.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            dict: Complete analysis results
        """
        print(f"🖼️ Processing image: {os.path.basename(image_path)}")
            
        # Step 1: Get image metadata
        print("  📊 Extracting metadata...")
        metadata = self.get_image_metadata(image_path)
            
        # Step 2: Get Moondream caption
        print("  🌙 Getting Moondream caption...")
        caption = self.get_moondream_caption(image_path)
        
        if not caption:
            print("  ❌ Failed to get Moondream caption")
            return None
            
        print(f"  ✅ Caption: {caption[:100]}...")
        
        # Step 3: Analyze with GPT-4o mini
        print("  🤖 Analyzing with GPT-4o mini...")
        analysis = self.analyze_with_gpt4o_mini(image_path, caption, metadata)
        
        if not analysis:
            print("  ❌ Failed to get GPT-4o mini analysis")
            return None
            
        # Combine all results
        result = {
            "file_path": os.path.abspath(image_path),
            "filename": os.path.basename(image_path),
            "moondream_caption": caption,
            "metadata": metadata,
            "analysis": analysis,
            "processing_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Save to metadata file
        self.save_metadata(image_path, result)
        
        # Store result before cleanup to prevent memory leaks
        result_copy = result
        
        # Use modular cleanup function
        cleanup_image_processing(
            metadata=metadata,
            caption=caption,
            analysis=analysis,
            result=result
        )
        
        print(f"  ✅ Image processing complete")
        
        return result_copy

    def save_metadata(self, image_path, result, metadata_file=None):
        """Save image analysis results to metadata file using config path."""
        if metadata_file is None:
            metadata_file = get_image_metadata_path()
            
        # Load existing metadata
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
            except Exception:
                metadata = {}
        else:
            metadata = {}
            
        # Save using absolute path as key
        abs_path = os.path.abspath(image_path)
        metadata[abs_path] = result
        
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Image metadata saved for: {abs_path}")

def cleanup_image_processing(metadata=None, caption=None, analysis=None, result=None):
    """
    Cleanup function for image processing to prevent memory leaks.
    
    Args:
        metadata: Image metadata dict
        caption: Moondream caption string
        analysis: GPT-4o mini analysis dict
        result: Final result dict
        
    Returns:
        float: Amount of memory freed in MB
    """
    try:
        import psutil
        import gc
        
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Delete provided variables
        cleanup_vars = {
            'metadata': metadata,
            'caption': caption,
            'analysis': analysis,
            'result': result
        }
        
        deleted_vars = []
        for var_name, var_value in cleanup_vars.items():
            if var_value is not None:
                del var_value
                deleted_vars.append(var_name)
        
        # Light garbage collection for image processing
        gc.collect()
        
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        memory_freed = memory_before - memory_after
        
        if deleted_vars and memory_freed > 0:
            print(f"🧹 Image cleanup: deleted {', '.join(deleted_vars)}, freed {memory_freed:.1f} MB")
        
        return memory_freed
        
    except ImportError:
        # If psutil not available, just do garbage collection
        import gc
        gc.collect()
        return 0.0

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python imageprocessor.py <directory_or_file>")
        return

    processor = ImageProcessor()
    path = sys.argv[1]

    processor.process_file(path)

if __name__ == "__main__":
    main() 