#!/usr/bin/env python3
"""
Script for downloading images from DerpiBooru using official API.
Uses CLIP embeddings for similarity-based filtering (optional).
"""

import os
import json
import argparse
import time
import warnings
from urllib.parse import unquote, urlsplit
from tqdm import tqdm
from PIL import Image
import requests


API_BASE_URL = "https://derpibooru.org/api/v1/json"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
ROOT = os.path.dirname(os.path.abspath(__file__))


def load_config(path):
    """Load configuration from config.json."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def search_images(tags, per_page=100):
    """
    Search for images using DerpiBooru API.
    
    Args:
        tags: Space-separated tags to search for (will be joined with +)
        per_page: Maximum number of images to fetch in one request
    
    Returns:
        List of image data objects from the API response
    """
    # Encode tags as API expects (+ instead of spaces, no special chars)
    encoded_tags = '+'.join(tags.split())
    
    url = f"{API_BASE_URL}/search/images"
    
    params = {
        'q': encoded_tags,          # Search query (tags)
        'per_page': per_page,       # Images per page
        'page': 1,                  # Page number
        'sf': 'score',              # Sort field: score, created_at, etc.
        'sd': '-score'              # Sort direction: - for descending, + for ascending
    }
    
    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'application/json',
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        images = data.get('images', [])
        
        print(f"API returned {len(images)} images for query: {encoded_tags}")
        return images
        
    except Exception as e:
        print(f"Error searching images: {e}")
        return []


def get_image_url(image_data):
    """Extract the direct image URL from API response."""
    # API returns images with representations, use full for best quality
    if 'representations' in image_data:
        # Try to get the highest quality representation first
        if 'full' in image_data['representations']:
            return image_data['representations']['full']
        elif 'medium' in image_data['representations']:
            return image_data['representations']['medium']
        elif 'large' in image_data['representations']:
            return image_data['representations']['large']
        elif 'tall' in image_data['representations']:
            return image_data['representations']['tall']
    
    # Fallback to old field names for compatibility (should not be needed)
    if 'original' in image_data:
        return image_data['original']
    elif 'medium' in image_data:
        return image_data['medium']
    
    # Last resort: use thumbnail
    if 'thumbnail' in image_data:
        return image_data['thumbnail']
    
    print(f"  ⚠️ Warning: No valid URL found for image ID {image_data.get('id', '?')}")
    return None


def download_image(image_url, output_dir):
    """Download an image and save it."""
    try:
        headers = {
            'User-Agent': USER_AGENT,
            'Accept': '*/*',
        }
        
        response = requests.get(image_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Проверяем заголовок Content-Type — отбрасываем видео и т.д.
        content_type = response.headers.get('Content-Type', '').lower()
        if 'image/' not in content_type:
            print(f"⚠️ Skipping non-image (Content-Type: {content_type}): {image_url}")
            return False
        
        # Extract filename from URL
        filename = os.path.basename(unquote(urlsplit(image_url).path))
        if not filename:
            filename = f"derpibooru_{abs(hash(image_url))}.jpg"
        
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        # Verify the file is a valid image
        try:
            with Image.open(output_path) as downloaded_image:
                downloaded_image.verify()
        except Exception as e:
            print(f"Warning: Invalid image format saved at {output_path}: {e}")
            os.remove(output_path)
            return False
        
        return True
    
    except Exception as e:
        print(f"Error downloading {image_url}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Download images from DerpiBooru using official API')
    parser.add_argument('tags', type=str, help='Tags to search for (space-separated)')
    parser.add_argument('--config', default=os.path.join(ROOT, 'config.json'), help='Path to config file')
    # Load config first to access nested values
    global_config = load_config(os.path.join(ROOT, 'config.json'))
    configured_limit = global_config.get('api', {}).get('max_limit', 20)
    parser.add_argument('-l', '--limit', type=int, default=None,
                        help=f'Maximum number of images to download (default: {configured_limit})')
    
    args = parser.parse_args()
    global_config = load_config(args.config)
    max_pixels = int(global_config.get('images', {}).get('max_pixels', 200_000_000))
    if max_pixels <= 0:
        parser.error('images.max_pixels must be greater than zero')
    Image.MAX_IMAGE_PIXELS = max_pixels
    warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)
    
    # Get API limits from config if available
    api_limits = global_config.get('api', {})
    max_per_request = args.limit if args.limit is not None else api_limits.get('max_limit', 20)
    if max_per_request <= 0:
        parser.error('--limit must be greater than zero')
    rate_limit_delay = api_limits.get('rate_limit_delay', 2.0)
    
    print(f"Searching for images with tags: {args.tags}")
    print(f"Max images per request: {max_per_request}")
    
    # Search for images (single query for now, can be extended to multiple queries)
    all_images = []
    page = 1
    page_size = min(max_per_request, 50)
    
    while len(all_images) < max_per_request:
        # Build search URL with pagination
        encoded_tags = '+'.join(args.tags.split())
        
        params = {
            'q': encoded_tags,
            'per_page': min(page_size, max_per_request - len(all_images)),
            'page': page,
            'sf': 'created_at',  # Sort by creation date
            'sd': '+created_at'  # Ascending order (oldest first)
        }
        
        headers = {'User-Agent': USER_AGENT, 'Accept': 'application/json'}
        
        try:
            response = requests.get(f"{API_BASE_URL}/search/images", params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            images = data.get('images', [])
            
            if not images:
                print("No more images found for this query.")
                break
            
            all_images.extend(images)
            page += 1
            
            # If we got fewer results than requested, we're done
            if len(images) < min(page_size, max_per_request - (len(all_images) - len(images))):
                print(f"API returned {len(images)} images (less than limit). Stopping pagination.")
                break
                
        except Exception as e:
            print(f"Error fetching page at page {page}: {e}")
            break
    
    # Remove duplicates based on image ID
    seen_ids = set()
    unique_images = []
    for img in all_images:
        if img['id'] not in seen_ids:
            seen_ids.add(img['id'])
            unique_images.append(img)
    
    print(f"Total unique images to download: {len(unique_images)}")
    
    if not unique_images:
        print("No images found!")
        return
    
    # Create output directory
    output_dir = global_config.get('directories', {}).get('temp', 'temp')
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded = 0
    for img in tqdm(unique_images, desc="Downloading"):
        image_url = get_image_url(img)
        
        if not image_url:
            print(f"Skipping image {img['id']}: no valid URL found")
            continue
        
        # Download the image
        success = download_image(image_url, output_dir)
        
        if success:
            downloaded += 1
        
        # Respect rate limits to be nice to the server
        time.sleep(rate_limit_delay)
    
    print(f"\nDownloaded {downloaded} out of {len(unique_images)} images")


if __name__ == '__main__':
    main()
