"""
Photobooth Template System - Core Logic
Enhanced box detection, smart resizing, and texture overlay
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import requests
from io import BytesIO
import hashlib


def find_white_boxes_flexible(template, min_area=1000, debug=False):
    """
    Find white/bright rectangular areas in template and separate into left and right strips.
    Works with any number of boxes - automatically detects the layout.
    Returns boxes as two vertical strips for duplicate image placement.
    Enhanced for better accuracy and edge detection.
    
    Args:
        template: cv2 image (BGR format)
        min_area: minimum area threshold for box detection
        debug: enable debug output
        
    Returns:
        tuple: (left_strip, right_strip) - lists of boxes (x, y, w, h)
    """
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Enhanced threshold for more accurate white detection
    _, th = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    
    # More precise morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)  # Remove noise
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)  # Close gaps

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
            
        # Get more accurate bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Refine box boundaries by checking actual white content
        crop = gray[y:y+h, x:x+w]
        mean_val = cv2.mean(crop)[0]
        
        if mean_val < 230:
            continue
            
        # Refine the box edges by finding exact white boundaries
        crop_thresh = cv2.threshold(crop, 230, 255, cv2.THRESH_BINARY)[1]
        coords = cv2.findNonZero(crop_thresh)
        
        if coords is not None:
            x_coords = coords[:, 0, 0]
            y_coords = coords[:, 0, 1]
            
            # Refine the bounding box to actual white content
            refined_x = int(x + np.min(x_coords))
            refined_y = int(y + np.min(y_coords)) 
            refined_w = int(np.max(x_coords) - np.min(x_coords))
            refined_h = int(np.max(y_coords) - np.min(y_coords))
            
            if refined_w > 10 and refined_h > 10:
                x, y, w, h = refined_x, refined_y, refined_w, refined_h
        
        boxes.append((int(x), int(y), int(w), int(h)))
    
    if debug:
        print(f"Found {len(boxes)} white boxes")
    
    # Sort all boxes by y (top to bottom) then x (left to right)
    boxes.sort(key=lambda r: (r[1], r[0]))
    
    # Separate into left and right columns
    center_x = template.shape[1] // 2
    left_strip = [box for box in boxes if box[0] + box[2]//2 < center_x]
    right_strip = [box for box in boxes if box[0] + box[2]//2 >= center_x]
    
    # Handle single-column templates
    if len(left_strip) == 0 and len(right_strip) > 0:
        left_strip = right_strip.copy()
    elif len(right_strip) == 0 and len(left_strip) > 0:
        right_strip = left_strip.copy()
    
    # Sort each strip top to bottom
    left_strip.sort(key=lambda r: r[1])
    right_strip.sort(key=lambda r: r[1])
    
    if debug:
        print(f"Left strip: {len(left_strip)} boxes, Right strip: {len(right_strip)} boxes")
    
    return left_strip, right_strip


def apply_filters(image_pil):
    """
    Apply subtle color enhancement filters to photo.
    
    Args:
        image_pil: PIL Image
        
    Returns:
        PIL Image with filters applied
    """
    # Enhance contrast slightly
    enhancer = ImageEnhance.Contrast(image_pil)
    image_pil = enhancer.enhance(1.1)
    
    # Enhance color slightly
    enhancer = ImageEnhance.Color(image_pil)
    image_pil = enhancer.enhance(1.05)
    
    return image_pil


def resize_and_place_photo(template, photo_pil, box, mode='cover', bw_filter=False):
    """
    Resize and place a PIL Image photo into a box with improved accuracy.
    
    Args:
        template: cv2 image (BGR format) to place photo into
        photo_pil: PIL Image to place
        box: tuple (x, y, w, h) defining placement area
        mode: 'cover' (fill box, may crop) or 'contain' (fit in box, may pad)
        bw_filter: apply black & white filter
        
    Returns:
        modified template with photo placed
    """
    x, y, w, h = box
    
    # Convert PIL to cv2 format
    photo_cv = cv2.cvtColor(np.array(photo_pil), cv2.COLOR_RGB2BGR)
    src_h, src_w = photo_cv.shape[0:2]

    # Calculate scaling based on mode
    if mode == 'cover':
        # Fill entire box (may crop)
        scale_w = w / src_w
        scale_h = h / src_h
        scale = max(scale_w, scale_h)  # Ensures complete coverage
    else:  # contain
        # Fit within box (may pad)
        scale_w = w / src_w
        scale_h = h / src_h
        scale = min(scale_w, scale_h)  # Ensures entire photo visible

    # Calculate new dimensions
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    
    # High-quality resize
    if scale > 1:
        resized = cv2.resize(photo_cv, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        resized = cv2.resize(photo_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create output array with exact box dimensions
    placed = np.full((h, w, 3), 255, dtype=np.uint8)
    
    if mode == 'cover':
        # Center crop to exact box size
        start_x = (new_w - w) // 2 if new_w > w else 0
        start_y = (new_h - h) // 2 if new_h > h else 0
        
        end_x = min(start_x + w, new_w)
        end_y = min(start_y + h, new_h)
        
        crop_w = end_x - start_x
        crop_h = end_y - start_y
        
        place_x = (w - crop_w) // 2
        place_y = (h - crop_h) // 2
        
        cropped = resized[start_y:end_y, start_x:end_x]
        placed[place_y:place_y + crop_h, place_x:place_x + crop_w] = cropped
    else:  # contain
        # Center the resized image
        offset_x = (w - new_w) // 2
        offset_y = (h - new_h) // 2
        placed[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

    # Convert to PIL and apply filters
    placed_pil = Image.fromarray(cv2.cvtColor(placed, cv2.COLOR_BGR2RGB))
    
    if bw_filter:
        placed_pil = placed_pil.convert('L').convert('RGB')
    else:
        placed_pil = apply_filters(placed_pil)
    
    # Place in template
    placed_final = cv2.cvtColor(np.array(placed_pil), cv2.COLOR_RGB2BGR)
    template[y:y + h, x:x + w] = placed_final
    return template


def overlay_texture(image, texture_path_or_url, opacity=0.1):
    """
    Apply texture overlay with automatic caching for URLs.
    
    Args:
        image: PIL Image to overlay texture on
        texture_path_or_url: local path or URL to texture image
        opacity: texture opacity (0.0 to 1.0)
        
    Returns:
        PIL Image with texture overlaid
    """
    try:
        if os.path.exists(texture_path_or_url):
            # Local file
            texture = Image.open(texture_path_or_url).convert("RGBA")
        else:
            # URL with cache
            cache_dir = os.path.join("static", "texture_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            url_hash = hashlib.md5(texture_path_or_url.encode()).hexdigest()
            cache_path = os.path.join(cache_dir, f"texture_{url_hash}.png")
            
            if os.path.exists(cache_path):
                texture = Image.open(cache_path).convert("RGBA")
            else:
                response = requests.get(texture_path_or_url, timeout=5)
                response.raise_for_status()
                texture = Image.open(BytesIO(response.content)).convert("RGBA")
                texture.save(cache_path)
        
        # Apply texture
        texture = texture.resize(image.size).convert("RGBA")
        mask = texture.split()[3].point(lambda x: int(x * opacity))
        image = image.convert("RGBA")
        image.paste(texture, (0, 0), mask)
        return image.convert("RGB")
    except Exception as e:
        print(f"Warning: Could not apply texture: {e}")
        return image


def create_strip_with_4x6_template(photo_paths, template_path, output_path='photo_strip_4x6_output.jpg', 
                                   bw_filter=False, expected_photo_count=None, debug=False):
    """
    Create photo strip using 4x6 template with duplicate placement in left and right strips.
    
    Args:
        photo_paths: list of paths to photos
        template_path: path to template image (or None for default)
        output_path: where to save output
        bw_filter: apply black & white filter
        expected_photo_count: expected number of photos (for validation)
        debug: enable debug output
    """
    template_cv = None
    
    if template_path and os.path.exists(template_path):
        # Load template
        template_cv = cv2.imread(template_path)
        
        if template_cv is None:
            print(f"Warning: Could not load template from {template_path}")
        else:
            # Auto-detect boxes
            left_strip, right_strip = find_white_boxes_flexible(template_cv, min_area=1000, debug=debug)
            
            # Validate template
            if not left_strip or not right_strip:
                print("Warning: No valid boxes found in template, using default layout")
                template_cv = None  # Fall through to default
            else:
                result = template_cv.copy()
                
                # Place photos in BOTH strips (duplicate)
                photos_to_place = min(len(photo_paths), len(left_strip), len(right_strip))
                
                if debug:
                    print(f"Placing {photos_to_place} photos in template")
                
                for photo_idx in range(photos_to_place):
                    photo_path = photo_paths[photo_idx]
                    
                    if not os.path.exists(photo_path):
                        print(f"Warning: Photo not found: {photo_path}")
                        continue
                    
                    # Load photo
                    photo_pil = Image.open(photo_path).convert('RGB')
                    
                    left_box = left_strip[photo_idx]
                    right_box = right_strip[photo_idx]
                    
                    if debug:
                        print(f"  Photo {photo_idx + 1}: {photo_path}")
                        print(f"    Left box: {left_box}")
                        print(f"    Right box: {right_box}")
                    
                    # Place in BOTH left and right strips
                    result = resize_and_place_photo(result, photo_pil, left_box, mode='cover', bw_filter=bw_filter)
                    result = resize_and_place_photo(result, photo_pil, right_box, mode='cover', bw_filter=bw_filter)
                
                # Convert to PIL
                result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                
                # Apply texture for color
                if not bw_filter:
                    texture_url = "https://i.ibb.co/vJt5HSh/noisy-texture-300x300-o10-d10-c-a82851-t1.png"
                    result_pil = overlay_texture(result_pil, texture_url, opacity=0.2)
                
                # Save with 300 DPI
                result_pil.save(output_path, dpi=(300, 300), quality=100)
                print(f"✓ Photo strip saved to: {output_path}")
                return
    
    # Default 4x6 layout fallback
    print("Using default 4x6 layout")
    strip_width = 1200   # 4 inches @ 300 DPI
    strip_height = 1800  # 6 inches @ 300 DPI
    
    # Calculate layout with margins
    margin = 30
    column_spacing = 20
    row_spacing = 20
    
    available_width = (strip_width - 2*margin - column_spacing) // 2
    available_height = (strip_height - 2*margin - 3*row_spacing) // 4
    
    # Create white strip
    strip = Image.new('RGB', (strip_width, strip_height), 'white')
    
    # Place 4 photos in 2 columns
    for idx in range(min(4, len(photo_paths))):
        if not os.path.exists(photo_paths[idx]):
            print(f"Warning: Photo not found: {photo_paths[idx]}")
            continue
            
        img = Image.open(photo_paths[idx]).convert('RGB')
        img = img.resize((available_width, available_height), Image.Resampling.LANCZOS)
        
        if bw_filter:
            img = img.convert('L').convert('RGB')
        else:
            img = apply_filters(img)
        
        y = margin + idx * (available_height + row_spacing)
        x_left = margin
        x_right = margin + available_width + column_spacing
        
        # Duplicate in both columns
        strip.paste(img, (x_left, y))
        strip.paste(img, (x_right, y))
    
    # Apply texture if not B&W
    if not bw_filter:
        texture_url = "https://i.ibb.co/vJt5HSh/noisy-texture-300x300-o10-d10-c-a82851-t1.png"
        strip = overlay_texture(strip, texture_url, opacity=0.2)
    
    strip.save(output_path, dpi=(300, 300), quality=100)
    print(f"✓ Photo strip saved to: {output_path}")


def main():
    """
    Example usage of the photobooth template system
    """
    # Example: Using template images in current directory
    template_path = "template.png"  # Update with your template path
    
    # Example photos
    photo_paths = [
        "G.png",
        "H.png", 
        "I.png",
        "J.png"
    ]
    
    # Create photo strip
    create_strip_with_4x6_template(
        photo_paths=photo_paths,
        template_path=template_path if os.path.exists(template_path) else None,
        output_path="output_photo_strip.jpg",
        bw_filter=False,
        debug=True
    )


if __name__ == "__main__":
    main()
