#!/usr/bin/env python3
"""
Location processing utilities using Google Maps APIs.
Provides accurate reverse geocoding and nearby places search.
"""

import os
import requests
import logging
from typing import Optional, Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def reverse_geocode_google(lat: float, lon: float) -> Optional[Dict]:
    """
    Use Google Maps Geocoding API to reverse geocode coordinates.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        
    Returns:
        dict: Google Maps API response or None if failed
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    
    if not api_key:
        logging.warning("GOOGLE_MAPS_API_KEY environment variable not set. Skipping Google Maps geocoding.")
        return None
    
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        'latlng': f"{lat},{lon}",
        'key': api_key,
        'language': 'en'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data['status'] != 'OK':
            logging.warning(f"Google Maps Geocoding API error: {data['status']}")
            return None
            
        return data
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Google Maps Geocoding request error: {e}")
        return None

def nearby_search_google(lat: float, lon: float, radius: int = 300) -> Optional[Dict]:
    """
    Search for nearby places using Google Places API.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        radius (int): Search radius in meters (default: 300)
        
    Returns:
        dict: Google Places API response or None if failed
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    
    if not api_key:
        return None
    
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        'location': f"{lat},{lon}",
        'radius': radius,
        'key': api_key,
        'language': 'en'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data['status'] not in ['OK', 'ZERO_RESULTS']:
            logging.warning(f"Google Places API error: {data['status']}")
            return None
            
        return data
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Google Places request error: {e}")
        return None

def extract_primary_address(geocode_data: Dict) -> Optional[str]:
    """
    Extract the primary address from Google Maps geocoding response.
    
    Args:
        geocode_data (dict): Google Maps geocoding response
        
    Returns:
        str: Primary formatted address or None
    """
    if not geocode_data or 'results' not in geocode_data or not geocode_data['results']:
        return None
    
    return geocode_data['results'][0].get('formatted_address')

def find_main_venue(places_data: Dict) -> Optional[Dict]:
    """
    Find the main venue (theme park, mall, campus, etc.) from Google Places response.
    Prioritizes broader venue names over specific attractions.
    
    Args:
        places_data (dict): Google Places API response
        
    Returns:
        dict: Main venue information or None
    """
    if not places_data or 'results' not in places_data:
        return None
    
    places = places_data['results']
    
    # Define venue types in order of priority (broader venues first)
    venue_type_priority = [
        # Major venues/complexes
        ['amusement_park', 'theme_park'],
        ['shopping_mall', 'shopping_center'],
        ['university', 'school'],
        ['hospital', 'medical_center'],
        ['airport'],
        ['stadium', 'sports_complex'],
        ['convention_center'],
        ['zoo', 'aquarium'],
        ['museum'],
        ['park'],
        ['casino'],
        ['resort'],
        ['golf_course'],
        # Business complexes
        ['office_park', 'business_park'],
        ['industrial_park'],
        # Transportation hubs
        ['transit_station', 'subway_station', 'train_station'],
        # Tourist attractions (broader ones)
        ['tourist_attraction']
    ]
    
    # Look for main venues by type priority
    for venue_types in venue_type_priority:
        for place in places:
            place_types = place.get('types', [])
            name = place.get('name', '')
            
            # Check if this place matches our venue types
            if any(vtype in place_types for vtype in venue_types):
                # Additional filtering for better venue detection
                user_ratings = place.get('user_ratings_total', 0)
                
                # Skip individual rides/attractions within theme parks
                if any(vtype in place_types for vtype in ['amusement_park', 'theme_park']):
                    # Look for the main park name, not individual rides
                    skip_keywords = [
                        'coaster', 'ride', 'roller', 'swing', 'drop', 'tower', 'wheel',
                        'bumper', 'carousel', 'ferris', 'log', 'flume', 'rapids',
                        'scrambler', 'spinner', 'twister', 'cyclone', 'screamer',
                        'flashback', 'pandemonium', 'houdini', 'chase', 'accelerator'
                    ]
                    
                    if any(keyword in name.lower() for keyword in skip_keywords):
                        continue
                
                # Prefer places with more reviews (indicates main venue vs sub-attraction)
                if venue_types[0] in ['amusement_park', 'theme_park'] and user_ratings < 1000:
                    continue
                
                return {
                    'name': name,
                    'types': place_types,
                    'rating': place.get('rating', 0),
                    'user_ratings': user_ratings,
                    'vicinity': place.get('vicinity', ''),
                    'place_id': place.get('place_id', '')
                }
    
    return None

def find_notable_landmarks(places_data: Dict, limit: int = 3) -> List[Dict]:
    """
    Find notable landmarks from Google Places response.
    
    Args:
        places_data (dict): Google Places API response
        limit (int): Maximum number of landmarks to return
        
    Returns:
        list: List of notable landmarks with name and types
    """
    if not places_data or 'results' not in places_data:
        return []
    
    places = places_data['results']
    notable_places = []
    
    # Sort by rating and prominence
    places.sort(key=lambda x: (
        x.get('user_ratings_total', 0),  # More reviews = more notable
        x.get('rating', 0)  # Higher rating
    ), reverse=True)
    
    for place in places[:limit * 2]:  # Check more places to find notable ones
        name = place.get('name', '')
        types = place.get('types', [])
        rating = place.get('rating', 0)
        user_ratings = place.get('user_ratings_total', 0)
        
        # Filter out boring generic types
        interesting_types = [t for t in types if t not in [
            'establishment', 'point_of_interest', 'geocode'
        ]]
        
        # Prioritize certain types as landmarks
        landmark_types = [
            'shopping_mall', 'office_park', 'business_park', 'university', 
            'hospital', 'school', 'park', 'tourist_attraction', 'museum',
            'library', 'government', 'courthouse', 'city_hall'
        ]
        
        # Check if it's a significant landmark
        is_landmark = (
            any(t in interesting_types for t in landmark_types) or
            user_ratings > 500 or  # Popular place
            (rating > 4.0 and user_ratings > 100)  # Highly rated
        )
        
        if is_landmark and len(notable_places) < limit:
            notable_places.append({
                'name': name,
                'types': interesting_types,
                'rating': rating,
                'user_ratings': user_ratings
            })
    
    return notable_places

def create_contextual_location(address: str, main_venue: Optional[Dict] = None, landmarks: List[Dict] = None) -> str:
    """
    Create a contextual location description prioritizing main venue over landmarks.
    Always includes both venue name and geographic context (city, state, country).
    
    Args:
        address (str): Primary address
        main_venue (dict): Main venue information (theme park, mall, etc.)
        landmarks (list): List of notable landmarks (fallback)
        
    Returns:
        str: Contextual location description with venue and geographic context
    """
    if landmarks is None:
        landmarks = []
    
    def extract_geographic_context(full_address: str) -> str:
        """Extract city, state, country from full address"""
        address_parts = full_address.split(', ')
        
        # For US addresses: typically "Street, City, State ZIP, USA"
        if len(address_parts) >= 3 and 'USA' in address_parts[-1]:
            # Get city and state (skip ZIP code part)
            state_zip = address_parts[-2]
            city = address_parts[-3]
            # Clean state from ZIP code if present
            state = state_zip.split()[0] if ' ' in state_zip else state_zip
            return f"{city}, {state}, USA"
        
        # For international addresses: typically "Street, City, Country"
        elif len(address_parts) >= 2:
            # Take last 2-3 parts as geographic context
            if len(address_parts) >= 3:
                return ', '.join(address_parts[-3:])
            else:
                return ', '.join(address_parts[-2:])
        
        # Fallback: return the full address
        return full_address
    
    # Extract geographic context from the full address
    geographic_context = extract_geographic_context(address)
    
    # Prioritize main venue over landmarks
    if main_venue:
        venue_name = main_venue['name']
        venue_types = main_venue.get('types', [])
        
        # For major venues, lead with venue name + geographic context
        if any(vtype in venue_types for vtype in [
            'amusement_park', 'theme_park', 'shopping_mall', 'university', 
            'hospital', 'airport', 'stadium', 'convention_center', 'zoo', 
            'aquarium', 'museum', 'casino', 'resort'
        ]):
            return f"{venue_name}, {geographic_context}"
        else:
            # For smaller venues, include full address context
            return f"{address}, at {venue_name}"
    
    # Fallback to landmarks if no main venue found
    if landmarks:
        # Find the best landmark (prefer business/office areas)
        business_landmarks = [l for l in landmarks if any(
            t in l['types'] for t in ['shopping_mall', 'office_park', 'business_park']
        )]
        
        if business_landmarks:
            landmark = business_landmarks[0]
        else:
            landmark = landmarks[0]
        
        return f"{address}, near {landmark['name']}"
    
    # Final fallback: just the address
    return address

def convert_coordinates_to_location_google(lat: float, lon: float) -> Optional[str]:
    """
    Convert GPS coordinates to readable location using Google Maps APIs.
    This is the main function to replace OpenAI-based location processing.
    
    Args:
        lat (float): Latitude 
        lon (float): Longitude
        
    Returns:
        str: Contextual location string or None if conversion fails
    """
    try:
        # Step 1: Reverse geocode coordinates
        geocode_result = reverse_geocode_google(lat, lon)
        if not geocode_result:
            return None
        
        address = extract_primary_address(geocode_result)
        if not address:
            return None
        
        # Step 2: Search for nearby places with larger radius for venue detection
        places_result = nearby_search_google(lat, lon, radius=500)
        main_venue = None
        landmarks = []
        
        if places_result:
            # First, try to find the main venue (theme park, mall, etc.)
            main_venue = find_main_venue(places_result)
            
            # If no main venue found, fall back to notable landmarks
            if not main_venue:
                landmarks = find_notable_landmarks(places_result, limit=2)
        
        # Step 3: Create contextual description
        contextual_location = create_contextual_location(address, main_venue, landmarks)
        
        logging.info(f"Google Maps location processing: {lat}, {lon} -> {contextual_location}")
        return contextual_location
        
    except Exception as e:
        logging.error(f"Google Maps location processing failed for {lat}, {lon}: {e}")
        return None

def parse_coordinate_string(coordinate_str: str) -> Optional[tuple]:
    """
    Parse various coordinate string formats to extract lat/lon.
    
    Args:
        coordinate_str (str): Coordinate string in various formats
        
    Returns:
        tuple: (latitude, longitude) or None if parsing fails
    """
    try:
        # Handle ISO 6709 format like "+37.3826-121.9759/"
        if '+' in coordinate_str and '-' in coordinate_str:
            # Remove trailing slash if present
            coord_clean = coordinate_str.rstrip('/')
            
            # Find the second occurrence of + or - (longitude start)
            pos = 1
            while pos < len(coord_clean) and coord_clean[pos] not in ['+', '-']:
                pos += 1
            
            if pos < len(coord_clean):
                lat_str = coord_clean[1:pos]  # Skip first +/-
                lon_str = coord_clean[pos:]
                
                lat = float(lat_str)
                lon = float(lon_str)
                
                # Validate reasonable coordinate ranges
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return (lat, lon)
        
        # Handle comma-separated format like "37.3826,-121.9759"
        if ',' in coordinate_str:
            parts = coordinate_str.split(',')
            if len(parts) == 2:
                lat = float(parts[0].strip())
                lon = float(parts[1].strip())
                
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return (lat, lon)
        
        # Handle space-separated format
        if ' ' in coordinate_str:
            parts = coordinate_str.split()
            if len(parts) == 2:
                lat = float(parts[0])
                lon = float(parts[1])
                
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return (lat, lon)
        
        return None
        
    except (ValueError, IndexError):
        return None

def process_location_from_metadata(location_text: str) -> Optional[str]:
    """
    Process location information from video/image metadata.
    First tries Google Maps, falls back to original text if coordinates can't be parsed.
    
    Args:
        location_text (str): Raw location string from metadata
        
    Returns:
        str: Processed location string or None
    """
    if not location_text or location_text.lower() in ['none', 'null', '', 'unknown']:
        return None
    
    # Try to parse as coordinates first
    coords = parse_coordinate_string(location_text)
    if coords:
        lat, lon = coords
        google_result = convert_coordinates_to_location_google(lat, lon)
        if google_result:
            return google_result
    
    # If not coordinates or Google Maps failed, return cleaned original text
    # Remove obvious coordinate-like patterns but keep readable location names
    cleaned = location_text.strip()
    if any(c in cleaned for c in ['+', '°', "'", '"']) and len(cleaned) < 50:
        # Looks like coordinates but we couldn't parse it - skip it
        return None
    
    return cleaned if len(cleaned) > 3 else None 