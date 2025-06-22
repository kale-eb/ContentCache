import os
import json
import base64
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pathlib import Path
from api_client import get_api_client
from config import get_image_metadata_path
import warnings

# Suppress PIL warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

def extract_image_metadata(image_path):
    """
    Extract metadata from image file including EXIF data, GPS coordinates, and description.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Extracted metadata including filename, coordinates, and description
    """
    try:
        # Get basic file information
        filename = os.path.basename(image_path)
        
        metadata = {
            'filename': filename,
            'coordinates': None,
            'included_description': None
    }
    
        # Try to extract EXIF data
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            
            if exif_data:
                # Extract GPS coordinates
                gps_info = {}
                for tag, value in exif_data.items():
                    tag_name = TAGS.get(tag, tag)
                    
                    if tag_name == "GPSInfo":
                        for gps_tag, gps_value in value.items():
                            gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                            gps_info[gps_tag_name] = gps_value
                
                # Convert GPS coordinates to decimal degrees
                if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
                    lat = convert_gps_to_decimal(
                        gps_info['GPSLatitude'], 
                        gps_info.get('GPSLatitudeRef', 'N')
                    )
                    lon = convert_gps_to_decimal(
                        gps_info['GPSLongitude'], 
                        gps_info.get('GPSLongitudeRef', 'E')
                    )
                    
                    if lat is not None and lon is not None:
                        metadata['coordinates'] = {
                            'latitude': lat,
                            'longitude': lon
                        }
                
                # Extract description/comment fields
                description_fields = ['ImageDescription', 'UserComment', 'XPComment', 'XPSubject']
                for field in description_fields:
                    if field in exif_data and exif_data[field]:
                        desc = str(exif_data[field]).strip()
                        if desc and desc.lower() not in ['none', 'null', '']:
                            metadata['included_description'] = desc
                            break
        
        return metadata
        
    except Exception as e:
        print(f"⚠️ Error extracting metadata from {image_path}: {e}")
        return {
            'filename': os.path.basename(image_path),
            'coordinates': None,
            'included_description': None
        }

def convert_gps_to_decimal(gps_coord, gps_ref):
    """
    Convert GPS coordinates from degrees/minutes/seconds to decimal degrees.
    
    Args:
        gps_coord: GPS coordinate in DMS format
        gps_ref: GPS reference (N/S for latitude, E/W for longitude)
        
    Returns:
        float: Decimal degree coordinate or None if conversion fails
    """
    try:
        if not gps_coord or len(gps_coord) < 3:
            return None
            
        degrees = float(gps_coord[0])
        minutes = float(gps_coord[1])
        seconds = float(gps_coord[2])
        
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        
        # Apply direction
        if gps_ref in ['S', 'W']:
            decimal = -decimal
            
        return decimal
            
    except (ValueError, TypeError, IndexError):
        return None

def encode_image_to_base64(image_path):
    """
    Encode image to base64 for API transmission.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded image
    """
    try:
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except Exception as e:
        print(f"❌ Error encoding image {image_path}: {e}")
        return None

def process_image(image_path):
    """
    Process a single image using Moondream for captioning/object detection 
    and OpenAI for final structured analysis.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Structured analysis result or None if processing fails
    """
    try:
        print(f"🖼️ Processing image: {os.path.basename(image_path)}")
            
        # Step 1: Extract metadata from image file
        print("  📊 Extracting metadata...")
        metadata = extract_image_metadata(image_path)
            
        # Step 2: Encode image to base64
        print("  🔄 Encoding image...")
        image_base64 = encode_image_to_base64(image_path)
        if not image_base64:
            print("  ❌ Failed to encode image")
            return None
            
        # Step 3: Get caption and objects from Moondream
        print("  🤖 Analyzing with Moondream...")
        client = get_api_client()
        
        moondream_result = client.moondream_analysis(image_base64)
        if not moondream_result:
            print("  ❌ Moondream analysis failed")
            return None
            
        caption = moondream_result.get('caption', '')
        objects = moondream_result.get('objects', [])
        
        print(f"  ✅ Caption: {caption[:100]}{'...' if len(caption) > 100 else ''}")
        print(f"  ✅ Objects: {', '.join(objects[:5])}{'...' if len(objects) > 5 else ''}")
        
        # Step 4: Get structured analysis from OpenAI
        print("  🧠 Generating structured analysis...")
        
        summary_result = client.openai_image_summary(
            caption=caption,
            objects=objects,
            filename=metadata['filename'],
            coordinates=metadata['coordinates'],
            included_description=metadata['included_description']
        )
        
        if not summary_result or 'result' not in summary_result:
            print("  ❌ OpenAI analysis failed")
            return None
        
        # Parse the structured result
        result_json = json.loads(summary_result['result'])
        
        print("  ✅ Image processing complete")
        return result_json
        
    except Exception as e:
        print(f"❌ Error processing image {image_path}: {e}")
        return None

def save_image_metadata(image_path, analysis_result, output_file=None):
    """
    Save image analysis results to metadata file.
    
    Args:
        image_path (str): Path to the processed image
        analysis_result (dict): Structured analysis result
        output_file (str, optional): Output file path (defaults to config path)
    """
    try:
        # Use config path by default
        if output_file is None:
            output_file = get_image_metadata_path()
        
        # Load existing metadata (if any)
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                all_data = json.load(f)
        else:
            all_data = {}
        
        # Use the absolute path to ensure uniqueness
        abs_path = os.path.abspath(image_path)
        
        # Save/overwrite the entry
        all_data[abs_path] = analysis_result
        
        # Write back to the JSON file
        with open(output_file, "w") as f:
            json.dump(all_data, f, indent=2)
        
        print(f"✅ Metadata saved for: {abs_path}")
        
    except Exception as e:
        print(f"❌ Error saving metadata for {image_path}: {e}")

def tag_image(image_path):
    """
    Main image tagging function - processes image and saves metadata.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Analysis result or None if processing fails
    """
    try:
        # Process the image
        result = process_image(image_path)
        
        if result:
            # Save metadata
            save_image_metadata(image_path, result)
            return result
        else:
            print(f"❌ Failed to process image: {image_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error in tag_image for {image_path}: {e}")
        return None

# Example usage and testing
if __name__ == "__main__":
    # Test with a sample image
    test_image = "/path/to/test/image.jpg"
    if os.path.exists(test_image):
        result = tag_image(test_image)
        if result:
            print("\n🎉 Test successful!")
            print(json.dumps(result, indent=2))
        else:
            print("\n❌ Test failed")
    else:
        print(f"Test image not found: {test_image}") 