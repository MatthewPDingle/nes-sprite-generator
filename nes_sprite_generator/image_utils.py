"""
Utility functions for image processing and rendering pixel art.
"""
import logging
import random
import colorsys
import io
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Set, BinaryIO
import math

try:
    from PIL import Image, ImageDraw
except ImportError:
    logging.error("Pillow (PIL) is required. Install with: pip install pillow")
    raise

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Type definitions
Color = str  # "#RRGGBB" hex string
PixelGrid = List[List[Optional[Color]]]
Palette = List[Color]
RGB = Tuple[int, int, int]
RGBA = Tuple[int, int, int, int]

def remove_white_background(image: Image.Image, tolerance: int = 10) -> Image.Image:
    """
    Remove the white background from an image using a flood-fill algorithm.
    Preserves white pixels that are part of the subject.
    
    Args:
        image: The PIL Image object
        tolerance: Color tolerance for what's considered "white" (0-255)
        
    Returns:
        A new image with transparent background
    """
    logger.info(f"Removing white background with tolerance: {tolerance}")
    
    # Ensure we're working with an RGBA image
    image = image.convert("RGBA")
    width, height = image.size
    
    # Create a mask image to mark background pixels
    mask = Image.new("L", (width, height), 0)
    mask_pixels = mask.load()
    
    # Get the image pixels
    img_pixels = image.load()
    
    # Count white-ish pixels for logging
    white_ish_count = 0
    total_pixels = width * height
    
    for y in range(height):
        for x in range(width):
            r, g, b, a = img_pixels[x, y]
            if r > 255-tolerance and g > 255-tolerance and b > 255-tolerance:
                white_ish_count += 1
    
    white_percentage = (white_ish_count / total_pixels) * 100
    logger.info(f"Image contains {white_ish_count} white-ish pixels out of {total_pixels} total ({white_percentage:.1f}%)")
    
    # Start with all border pixels
    border_points = []
    
    # Add top and bottom border points
    for x in range(width):
        border_points.append((x, 0))         # Top border
        border_points.append((x, height-1))  # Bottom border
    
    # Add left and right border points
    for y in range(height):
        border_points.append((0, y))         # Left border
        border_points.append((width-1, y))   # Right border
    
    # Process each border point that looks like background
    filled_count = 0
    
    for start_point in border_points:
        x, y = start_point
        
        # Skip if already processed
        if mask_pixels[x, y] != 0:
            continue
        
        # Get color at this point
        r, g, b, a = img_pixels[x, y]
        
        # Check if it's white/near-white (using tolerance)
        if r > 255-tolerance and g > 255-tolerance and b > 255-tolerance:
            # Flood fill from this point
            stack = [start_point]
            points_filled = 0
            
            while stack:
                px, py = stack.pop()
                
                # Skip invalid coordinates
                if px < 0 or py < 0 or px >= width or py >= height:
                    continue
                
                # Skip already processed pixels
                if mask_pixels[px, py] != 0:
                    continue
                
                # Get pixel color
                pr, pg, pb, pa = img_pixels[px, py]
                
                # If it's white/near-white, mark as background and add neighbors
                if pr > 255-tolerance and pg > 255-tolerance and pb > 255-tolerance:
                    mask_pixels[px, py] = 255  # Mark as background
                    points_filled += 1
                    
                    # Add 8-connected neighbors for more thorough fill
                    for nx, ny in [(px+1, py), (px-1, py), (px, py+1), (px, py-1),
                                   (px+1, py+1), (px-1, py-1), (px+1, py-1), (px-1, py+1)]:
                        if 0 <= nx < width and 0 <= ny < height:
                            stack.append((nx, ny))
            
            filled_count += points_filled
    
    # Log how many pixels were identified as background
    logger.info(f"Identified {filled_count} pixels as background ({(filled_count / total_pixels) * 100:.1f}% of image)")
    
    # If we didn't fill much, we might have a problem (no clear background)
    if filled_count < 0.2 * total_pixels:
        logger.warning(f"Only identified {filled_count} background pixels. This may indicate there's no clear white background.")
    
    # Create a checker pattern to see the transparency in saved images
    checker = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    checker_size = 8  # Size of checker pattern squares
    
    for y in range(0, height, checker_size):
        for x in range(0, width, checker_size):
            # Alternate between light and dark gray
            color = (200, 200, 200, 255) if ((x // checker_size) + (y // checker_size)) % 2 == 0 else (100, 100, 100, 255)
            
            # Draw the checker square
            for cy in range(min(checker_size, height - y)):
                for cx in range(min(checker_size, width - x)):
                    checker.putpixel((x + cx, y + cy), color)
    
    # Create a new image with transparent background
    result = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    
    # Copy the image pixels, making background pixels transparent
    transparent_count = 0
    
    for y in range(height):
        for x in range(width):
            if mask_pixels[x, y] == 0:  # Not background
                result.putpixel((x, y), img_pixels[x, y])
            else:  # Background
                result.putpixel((x, y), (0, 0, 0, 0))  # Transparent
                transparent_count += 1
    
    logger.info(f"Made {transparent_count} pixels transparent ({(transparent_count / total_pixels) * 100:.1f}% of image)")
    
    return result

def detect_bounding_box(image: Image.Image) -> Tuple[int, int, int, int]:
    """
    Detect the bounding box of the non-transparent content in an image.
    
    Args:
        image: The PIL Image object with transparent background
        
    Returns:
        Tuple of (left, top, right, bottom) coordinates
    """
    # Ensure image has alpha channel
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    
    # Get the alpha band (channel 3 in RGBA)
    alpha = image.split()[3]
    
    # Get bounding box of non-zero (non-transparent) region
    bbox = alpha.getbbox()
    
    if not bbox:
        # If no non-transparent pixels found, return the whole image
        logger.warning("No non-transparent pixels found in the image!")
        full_bbox = (0, 0, image.width, image.height)
        logger.info(f"Using full image bounds: {full_bbox}")
        return full_bbox
    
    # Calculate content dimensions
    content_width = bbox[2] - bbox[0]
    content_height = bbox[3] - bbox[1]
    original_width, original_height = image.size
    
    logger.info(f"Detected content bounding box: {bbox} (size: {content_width}x{content_height})")
    logger.info(f"Content occupies {(content_width * content_height) / (original_width * original_height) * 100:.1f}% of original image")
    
    return bbox

def crop_resize_and_center(
    image: Image.Image, 
    target_width: int, 
    target_height: int
) -> Image.Image:
    """
    Crop to the content bounding box, resize while preserving aspect ratio,
    and center/bottom-align onto a new canvas.
    
    Args:
        image: The PIL Image object with transparent background
        target_width: Target width in pixels
        target_height: Target height in pixels
        
    Returns:
        A new image with the content cropped, resized, and positioned
    """
    # Detect bounding box of content
    bbox = detect_bounding_box(image)
    
    # Crop to bounding box
    content = image.crop(bbox)
    content_width, content_height = content.size
    
    logger.info(f"Cropped content size: {content_width}x{content_height}")
    
    # Calculate scale to preserve aspect ratio
    width_ratio = target_width / content_width
    height_ratio = target_height / content_height
    scale_ratio = min(width_ratio, height_ratio)
    
    # Calculate new dimensions
    new_width = max(1, int(content_width * scale_ratio))
    new_height = max(1, int(content_height * scale_ratio))
    
    logger.info(f"Resized dimensions: {new_width}x{new_height} (scale ratio: {scale_ratio:.3f})")
    
    # Resize using bilinear sampling for smoother results
    resized = content.resize((new_width, new_height), Image.BILINEAR)
    
    # Create a new transparent canvas of the target size
    canvas = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
    
    # Calculate position (center horizontally, bottom-align vertically)
    x_offset = (target_width - new_width) // 2
    y_offset = target_height - new_height  # Bottom align
    
    logger.info(f"Positioning at ({x_offset}, {y_offset})")
    
    # Paste the resized content onto the canvas
    canvas.paste(resized, (x_offset, y_offset), resized)
    
    return canvas

def reduce_colors(image: Image.Image, max_colors: int, transparency_threshold: int = 128) -> Image.Image:
    """
    Reduce the number of colors in an image to the specified maximum.
    Always includes pure black and white in the palette.
    Converts mostly transparent pixels to fully transparent.
    
    Args:
        image: The PIL Image object
        max_colors: Maximum number of colors allowed
        transparency_threshold: Alpha threshold below which pixels become transparent
        
    Returns:
        A new image with reduced color palette
    """
    # Convert to RGBA mode to properly handle transparency
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    
    # First, convert mostly transparent pixels to fully transparent
    pixels = image.load()
    width, height = image.size
    
    # Create a copy to work with
    processed_image = image.copy()
    processed_pixels = processed_image.load()
    
    # Process transparency
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            # If mostly transparent, make fully transparent
            if a < transparency_threshold:
                processed_pixels[x, y] = (0, 0, 0, 0)
    
    # Count unique colors (ignoring transparent pixels)
    unique_colors = set()
    for y in range(height):
        for x in range(width):
            pixel = processed_pixels[x, y]
            if pixel[3] > 0:  # Not transparent
                unique_colors.add(pixel[:3])  # Add RGB (ignore alpha)
    
    current_colors = len(unique_colors)
    logger.info(f"Current unique colors after transparency processing: {current_colors}")
    
    # If we're already under the limit after reserving space for black and white, no need to reduce
    reserved_colors = 2  # Black and white
    available_colors = max_colors - reserved_colors
    
    if current_colors <= available_colors:
        logger.info("No color reduction needed (after reserving black and white)")
        # Still need to ensure black and white are in palette
        return _ensure_black_white_in_palette(processed_image, max_colors)
    
    # Create a mask for non-transparent pixels
    visible_mask = Image.new("L", processed_image.size, 0)
    for y in range(height):
        for x in range(width):
            if processed_pixels[x, y][3] > 0:  # Not transparent
                visible_mask.putpixel((x, y), 255)
    
    # Prepare a new RGB image with non-transparent parts
    rgb_image = Image.new("RGB", processed_image.size, (0, 0, 0))
    for y in range(height):
        for x in range(width):
            if processed_pixels[x, y][3] > 0:  # Not transparent
                rgb_image.putpixel((x, y), processed_pixels[x, y][:3])
    
    # Ensure we have enough room for black and white plus at least one other color
    adjusted_max_colors = min(max_colors, current_colors + reserved_colors)
    logger.info(f"Reducing to {adjusted_max_colors} colors (including reserved colors)")
    
    # Quantize the colors
    quantized = rgb_image.quantize(colors=adjusted_max_colors, method=2, dither=0)  # method=2 is FASTOCTREE
    
    # Convert back to RGBA, restoring transparency
    result = quantized.convert("RGBA")
    
    # Restore transparency
    for y in range(height):
        for x in range(width):
            if processed_pixels[x, y][3] == 0:  # Fully transparent in processed image
                result.putpixel((x, y), (0, 0, 0, 0))
    
    # Ensure black and white are in the palette
    return _ensure_black_white_in_palette(result, max_colors)

def _ensure_black_white_in_palette(image: Image.Image, max_colors: int) -> Image.Image:
    """
    Ensures that pure black and white are in the image's palette.
    If necessary, replaces colors that are close to black or white.
    
    Args:
        image: The PIL Image object
        max_colors: Maximum color limit
        
    Returns:
        Image with black and white in palette
    """
    BLACK = (0, 0, 0, 255)
    WHITE = (255, 255, 255, 255)
    
    # Work with a copy
    result = image.copy()
    pixels = result.load()
    width, height = result.size
    
    # Check if black and white already exist in the image
    has_black = False
    has_white = False
    closest_to_black = None
    closest_to_white = None
    min_black_distance = float('inf')
    min_white_distance = float('inf')
    
    # Count occurrences of each color
    color_count = {}
    for y in range(height):
        for x in range(width):
            pixel = pixels[x, y]
            if pixel[3] == 0:  # Skip transparent pixels
                continue
                
            # Check for exact black or white
            if pixel[:3] == BLACK[:3]:
                has_black = True
            elif pixel[:3] == WHITE[:3]:
                has_white = True
                
            # Track color occurrences
            color_key = pixel[:3]
            if color_key in color_count:
                color_count[color_key] += 1
            else:
                color_count[color_key] = 1
                
            # Calculate distance to black and white
            r, g, b, a = pixel
            # Simple Euclidean distance in RGB space
            black_distance = (r**2 + g**2 + b**2)**0.5
            white_distance = ((255-r)**2 + (255-g)**2 + (255-b)**2)**0.5
            
            # Update closest colors
            if black_distance < min_black_distance:
                min_black_distance = black_distance
                closest_to_black = (r, g, b, a)
                
            if white_distance < min_white_distance:
                min_white_distance = white_distance
                closest_to_white = (r, g, b, a)
    
    logger.info(f"Has black: {has_black}, Has white: {has_white}")
    
    # Add black and white if missing
    # Strategy: Replace dark colors with black and light colors with white
    # Usually preserving occurrence counts - replace the least common close colors
    
    # Sort colors by occurrence (least common first)
    sorted_colors = sorted(color_count.items(), key=lambda x: x[1])
    colors_to_replace = []
    
    if not has_black and len(sorted_colors) > 0:
        # Find darkest color to replace
        darkest = min(sorted_colors, key=lambda x: sum(x[0]))
        # Add to replacement list
        colors_to_replace.append((darkest[0], BLACK[:3]))
        logger.info(f"Adding black by replacing color {darkest[0]}")
        
    if not has_white and len(sorted_colors) > 0:
        # Find lightest color to replace
        lightest = max(sorted_colors, key=lambda x: sum(x[0]))
        # Add to replacement list but avoid replacing the same color twice
        if lightest[0] != colors_to_replace[0][0] if colors_to_replace else True:
            colors_to_replace.append((lightest[0], WHITE[:3]))
            logger.info(f"Adding white by replacing color {lightest[0]}")
    
    # Apply replacements
    for old_color, new_color in colors_to_replace:
        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]
                if (r, g, b) == old_color and a > 0:
                    pixels[x, y] = (*new_color, a)
    
    return result

def process_raw_image(
    image_data: Union[bytes, BinaryIO],
    target_width: int,
    target_height: int,
    max_colors: int,
    white_tolerance: int = 10,
    save_steps: bool = True,
    output_prefix: str = "sprite"
) -> Image.Image:
    """
    Process a raw image according to the NES sprite generator workflow.
    Optionally saves each intermediate step as an image file.
    
    Args:
        image_data: Raw image data as bytes or file-like object
        target_width: Target width in pixels
        target_height: Target height in pixels
        max_colors: Maximum number of colors allowed
        white_tolerance: Tolerance for white detection
        save_steps: Whether to save intermediate step images
        output_prefix: Prefix for saved intermediate files
        
    Returns:
        A processed PIL Image
    """
    import os
    import time
    
    # Create a timestamp for uniqueness
    timestamp = int(time.time())
    
    # Create output directory for intermediate steps if saving
    if save_steps:
        output_dir = f"debug_steps_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving intermediate steps to: {output_dir}/")
    
    # Load the image
    if isinstance(image_data, bytes):
        image = Image.open(io.BytesIO(image_data))
    else:
        image = Image.open(image_data)
    
    logger.info(f"Processing raw image of size: {image.size}x{image.mode}")
    
    # Convert to RGBA
    image = image.convert("RGBA")
    
    # Save original image
    if save_steps:
        image.save(f"{output_dir}/{output_prefix}_{timestamp}_1_original.png")
        logger.info(f"Saved original image: {output_dir}/{output_prefix}_{timestamp}_1_original.png")
    
    # 1. Remove white background
    image_no_bg = remove_white_background(image, tolerance=white_tolerance)
    
    # Save background removal result
    if save_steps:
        image_no_bg.save(f"{output_dir}/{output_prefix}_{timestamp}_2_no_background.png")
        logger.info(f"Saved image with background removed: {output_dir}/{output_prefix}_{timestamp}_2_no_background.png")
    
    # Get bounding box
    bbox = detect_bounding_box(image_no_bg)
    
    # Save bounding box visualization (draw a rectangle around the content)
    if save_steps:
        # Create a copy of the image to draw on
        bbox_viz = image_no_bg.copy()
        draw = ImageDraw.Draw(bbox_viz)
        # Draw red rectangle for bounding box
        draw.rectangle(bbox, outline=(255, 0, 0, 255), width=2)
        bbox_viz.save(f"{output_dir}/{output_prefix}_{timestamp}_3_bounding_box.png")
        logger.info(f"Saved bounding box visualization: {output_dir}/{output_prefix}_{timestamp}_3_bounding_box.png")
        
        # Save cropped content
        cropped = image_no_bg.crop(bbox)
        cropped.save(f"{output_dir}/{output_prefix}_{timestamp}_4_cropped.png")
        logger.info(f"Saved cropped content: {output_dir}/{output_prefix}_{timestamp}_4_cropped.png")
    
    # 2-4. Crop, resize, and position content
    processed = crop_resize_and_center(
        image_no_bg, 
        target_width=target_width, 
        target_height=target_height
    )
    
    # Save resized and positioned content
    if save_steps:
        processed.save(f"{output_dir}/{output_prefix}_{timestamp}_5_resized.png")
        logger.info(f"Saved resized and positioned image: {output_dir}/{output_prefix}_{timestamp}_5_resized.png")
    
    # 5. Reduce colors if needed
    final = reduce_colors(processed, max_colors=max_colors)
    
    # Save final result with reduced colors
    if save_steps:
        final.save(f"{output_dir}/{output_prefix}_{timestamp}_6_final.png")
        logger.info(f"Saved final image with reduced colors: {output_dir}/{output_prefix}_{timestamp}_6_final.png")
        
        # Create a composite image with all steps side by side
        try:
            # Get original dimensions
            original_width, original_height = image.size
            
            # Ensure we have the content dimensions from the bbox
            bbox = detect_bounding_box(image_no_bg)
            content_width = bbox[2] - bbox[0]
            content_height = bbox[3] - bbox[1]
            
            # Make a cropped version for visualization
            cropped = image_no_bg.crop(bbox)
            
            # Create a bbox visualization image
            bbox_viz = image_no_bg.copy()
            draw = ImageDraw.Draw(bbox_viz)
            draw.rectangle(bbox, outline=(255, 0, 0, 255), width=2)
            
            # Choose a scale factor based on original image size
            scale_factor = min(max(1, 800 // (original_width or 1)), 20)
            if scale_factor < 2 and min(original_width, original_height) < 100:
                scale_factor = 2
                
            # Scale up the images for better visibility
            scaled_original = image.resize((original_width * scale_factor, original_height * scale_factor))
            scaled_no_bg = image_no_bg.resize((original_width * scale_factor, original_height * scale_factor))
            scaled_bbox = bbox_viz.resize((original_width * scale_factor, original_height * scale_factor))
            
            # Scale up the final images more since they're smaller
            small_scale = max(scale_factor * 4, 8)
            final_scale = max(small_scale, 40 // min(target_width, target_height))
            
            scaled_cropped = cropped.resize((content_width * scale_factor, content_height * scale_factor))
            scaled_processed = processed.resize((target_width * final_scale, target_height * final_scale))
            scaled_final = final.resize((target_width * final_scale, target_height * final_scale))
            
            # Calculate padding and composite width
            padding = 20
            title_height = 40
            
            # Calculate dimensions for the composite
            col1_width = original_width * scale_factor
            col2_width = original_width * scale_factor
            col3_width = content_width * scale_factor
            col4_width = max(target_width * final_scale, target_width * final_scale)
            
            # Composite width is the sum of all columns plus padding
            comp_width = col1_width + col2_width + col3_width + col4_width + (5 * padding)
            
            # Height is the max of the original height plus title space, processed height, and padding
            comp_height = max(
                (original_height * scale_factor) + title_height + (2 * padding),
                title_height + (2 * padding) + scaled_processed.height + padding//2 + scaled_final.height
            )
            
            # Create the composite image
            comp = Image.new("RGB", (comp_width, comp_height), (240, 240, 240))
            
            # Define columns for each image
            x1 = padding
            x2 = x1 + col1_width + padding
            x3 = x2 + col2_width + padding
            x4 = x3 + col3_width + padding
            
            # Title Y position and image Y position
            title_y = padding
            img_y = title_y + title_height
            
            # Function to add a title to the composite
            def add_title(text, x, y, width):
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(comp)
                try:
                    # Try to load a font, fallback to default if not available
                    font = ImageFont.truetype("Arial.ttf", 16)
                except IOError:
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
                    except IOError:
                        font = ImageFont.load_default()
                        
                # Draw the title centered in its column
                text_width = draw.textlength(text, font=font)
                draw.text((x + (width - text_width) // 2, y), text, fill=(0, 0, 0), font=font)
            
            # Paste the images and add titles
            comp.paste(scaled_original, (x1, img_y))
            add_title("1. Original", x1, title_y, col1_width)
            
            comp.paste(scaled_no_bg, (x2, img_y))
            add_title("2. Background Removed", x2, title_y, col2_width)
            
            comp.paste(scaled_cropped, (x3, img_y))
            add_title("3. Cropped Content", x3, title_y, col3_width)
            
            # For the last column, stack the processed and final images
            stack_y = img_y
            comp.paste(scaled_processed, (x4, stack_y))
            add_title("4. Resized", x4, title_y, col4_width)
            
            # Add the final image below with some spacing
            stack_y += scaled_processed.height + padding//2
            if stack_y + scaled_final.height < comp_height:
                comp.paste(scaled_final, (x4, stack_y))
                add_title("5. Color Reduced", x4, stack_y - title_height//2, col4_width)
            
            # Save the composite
            comp_path = f"{output_dir}/{output_prefix}_{timestamp}_composite.png"
            comp.save(comp_path)
            logger.info(f"Saved composite image: {comp_path}")
            
        except Exception as e:
            logger.error(f"Failed to create composite image: {e}")
    
    return final

def image_to_pixel_grid(image: Image.Image) -> Tuple[PixelGrid, Palette]:
    """
    Convert a PIL Image to a pixel grid and palette.
    
    Args:
        image: PIL Image object
        
    Returns:
        Tuple of (pixel_grid, palette)
    """
    # Ensure we're working with an RGBA image
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    
    # Get image dimensions
    width, height = image.size
    
    # Convert to pixel grid
    pixel_grid = []
    color_to_hex = {}
    palette = []
    
    for y in range(height):
        row = []
        for x in range(width):
            pixel = image.getpixel((x, y))
            
            if pixel[3] == 0:  # Transparent
                row.append(None)
            else:
                # Create hex color from RGB
                hex_color = f"#{pixel[0]:02x}{pixel[1]:02x}{pixel[2]:02x}"
                
                # Add to palette if not seen before
                if hex_color not in color_to_hex:
                    color_to_hex[hex_color] = hex_color
                    palette.append(hex_color)
                
                row.append(hex_color)
        
        pixel_grid.append(row)
    
    return pixel_grid, palette

def render_pixel_grid(pixel_grid: PixelGrid, palette: Palette, scale: int = 1) -> Image.Image:
    """
    Renders a pixel grid to a PIL Image.
    
    Args:
        pixel_grid: 2D list of color values (hex strings or null for transparency)
        palette: List of colors used in the grid
        scale: Scaling factor (1 = 1 pixel per grid cell, 8 = 8x8 pixels per cell)
        
    Returns:
        PIL Image object
    """
    if not pixel_grid or not pixel_grid[0]:
        raise ValueError("Pixel grid cannot be empty")
    
    # Determine dimensions
    height = len(pixel_grid)
    width = len(pixel_grid[0])
    
    # Create a new image with transparency
    img = Image.new('RGBA', (width * scale, height * scale), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw each pixel
    for y in range(height):
        for x in range(width):
            # Get the color value
            color_value = pixel_grid[y][x] if y < len(pixel_grid) and x < len(pixel_grid[y]) else None
            
            # Skip transparent pixels
            if color_value is None:
                continue
            
            # Try to parse the color as a hex value
            try:
                if isinstance(color_value, str):
                    # Handle hex color strings
                    if color_value.startswith('#'):
                        r = int(color_value[1:3], 16)
                        g = int(color_value[3:5], 16)
                        b = int(color_value[5:7], 16)
                        color = (r, g, b, 255)  # Full opacity
                    else:
                        # If no hex but a string, try to use as index
                        idx = int(color_value)
                        if 0 <= idx < len(palette):
                            hex_color = palette[idx]
                            r = int(hex_color[1:3], 16)
                            g = int(hex_color[3:5], 16)
                            b = int(hex_color[5:7], 16)
                            color = (r, g, b, 255)
                        else:
                            # Use a random color if index is out of range
                            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)
                elif isinstance(color_value, int):
                    # Treat as palette index
                    if 0 <= color_value < len(palette):
                        hex_color = palette[color_value]
                        r = int(hex_color[1:3], 16)
                        g = int(hex_color[3:5], 16)
                        b = int(hex_color[5:7], 16)
                        color = (r, g, b, 255)
                    else:
                        # Use a random color if index is out of range
                        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)
                else:
                    # Use a random color for unknown types
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)
            except Exception as e:
                logger.warning(f"Error parsing color at ({x}, {y}): {color_value}, {e}")
                # Fallback to a visible error color
                color = (255, 0, 255, 255)  # Magenta for errors
            
            # Draw the pixel (scaled)
            draw.rectangle(
                [(x * scale, y * scale), ((x + 1) * scale - 1, (y + 1) * scale - 1)], 
                fill=color
            )
    
    return img

def optimize_colors(pixel_grid: PixelGrid, palette: Palette, max_colors: int) -> Tuple[PixelGrid, Palette]:
    """
    Optimize the color palette to have at most max_colors.
    
    Args:
        pixel_grid: 2D list of color values
        palette: Original color palette
        max_colors: Maximum number of colors allowed
        
    Returns:
        Tuple of (optimized pixel grid, optimized palette)
    """
    if len(palette) <= max_colors:
        # No optimization needed
        return pixel_grid, palette
    
    # Count color usage
    color_usage = {}
    for row in pixel_grid:
        for color in row:
            if color is not None:
                if color in color_usage:
                    color_usage[color] += 1
                else:
                    color_usage[color] = 1
    
    # Sort colors by usage (most used first)
    sorted_colors = sorted(color_usage.keys(), key=lambda c: color_usage[c], reverse=True)
    
    # Take the top max_colors colors
    new_palette = sorted_colors[:max_colors]
    
    # Create a mapping for remaining colors to nearest color in new palette
    color_map = {}
    for color in sorted_colors[max_colors:]:
        nearest_color = find_nearest_color(color, new_palette)
        color_map[color] = nearest_color
    
    # Apply the mapping to the pixel grid
    new_grid = []
    for row in pixel_grid:
        new_row = []
        for color in row:
            if color is None:
                new_row.append(None)
            elif color in new_palette:
                new_row.append(color)
            else:
                new_row.append(color_map[color])
        new_grid.append(new_row)
    
    return new_grid, new_palette

def hex_to_rgb(hex_color: str) -> RGB:
    """Convert a hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb: RGB) -> str:
    """Convert an RGB tuple to hex color string."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

def color_distance(color1: str, color2: str) -> float:
    """
    Calculate the perceptual distance between two colors.
    
    This uses a weighted Euclidean distance in RGB space.
    """
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)
    
    # Calculate weighted Euclidean distance
    r_mean = (rgb1[0] + rgb2[0]) / 2
    r_diff = rgb1[0] - rgb2[0]
    g_diff = rgb1[1] - rgb2[1]
    b_diff = rgb1[2] - rgb2[2]
    
    # Weights for human perception
    r_weight = 2 + r_mean / 256
    g_weight = 4.0
    b_weight = 2 + (255 - r_mean) / 256
    
    return math.sqrt(
        r_weight * r_diff**2 +
        g_weight * g_diff**2 +
        b_weight * b_diff**2
    )

def find_nearest_color(color: str, palette: List[str]) -> str:
    """Find the nearest color in the palette to the given color."""
    return min(palette, key=lambda c: color_distance(color, c))

def find_content_boundaries(pixel_grid: PixelGrid) -> tuple[int, int, int, int]:
    """
    Find the boundaries of the actual content in a pixel grid, ignoring transparent pixels.
    
    Args:
        pixel_grid: The pixel grid to analyze
        
    Returns:
        Tuple of (min_x, max_x, min_y, max_y) content boundaries
    """
    height = len(pixel_grid)
    width = len(pixel_grid[0]) if height > 0 else 0
    
    # Initialize bounds to extreme values
    min_x, max_x, min_y, max_y = width, 0, height, 0
    has_content = False
    
    # Scan the grid for non-transparent pixels
    for y in range(height):
        for x in range(width):
            if pixel_grid[y][x] is not None:
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                has_content = True
    
    # If no content found, return full dimensions
    if not has_content:
        return 0, width - 1, 0, height - 1
        
    return min_x, max_x, min_y, max_y

def smart_resize(pixel_grid: PixelGrid, target_width: int, target_height: int, 
               method: str = "nearest") -> PixelGrid:
    """
    Smart resize that:
    1. Detects the actual content boundaries (ignoring transparent pixels)
    2. Crops to that content
    3. Resizes while preserving aspect ratio
    4. Centers horizontally and aligns to bottom
    
    Args:
        pixel_grid: Original pixel grid
        target_width: Desired width
        target_height: Desired height
        method: Resizing method ('nearest', 'bilinear', etc.)
        
    Returns:
        A smartly resized pixel grid
    """
    height = len(pixel_grid)
    width = len(pixel_grid[0]) if height > 0 else 0
    
    # Step 1: Find the content boundaries
    min_x, max_x, min_y, max_y = find_content_boundaries(pixel_grid)
    
    # Calculate content dimensions
    content_width = max_x - min_x + 1
    content_height = max_y - min_y + 1
    
    logging.info(f"Detected content dimensions: {content_width}x{content_height}")
    
    # Step 2: Create a new grid with just the content
    content_grid = []
    for y in range(min_y, max_y + 1):
        row = []
        for x in range(min_x, max_x + 1):
            row.append(pixel_grid[y][x])
        content_grid.append(row)
    
    # Step 3: Calculate scaling factor to preserve aspect ratio
    width_ratio = target_width / content_width
    height_ratio = target_height / content_height
    
    # Use the smaller ratio to ensure the image fits within the canvas
    scale_ratio = min(width_ratio, height_ratio)
    
    # Calculate the new dimensions after scaling
    new_width = max(1, int(content_width * scale_ratio))
    new_height = max(1, int(content_height * scale_ratio))
    
    logging.info(f"Scaling content with ratio {scale_ratio:.2f} to {new_width}x{new_height}")
    
    # Step 4: Resize the content grid
    # Convert to PIL image for resizing
    temp_palette = []
    color_map = {}
    
    # Create a PIL image from the content grid
    content_img = Image.new("RGBA", (content_width, content_height), (0, 0, 0, 0))
    for y in range(content_height):
        for x in range(content_width):
            color = content_grid[y][x]
            if color is not None:
                if color.startswith('#'):
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16)
                    b = int(color[5:7], 16)
                    content_img.putpixel((x, y), (r, g, b, 255))
    
    # Choose the right resampling method
    resampling_methods = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR, 
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS
    }
    resampling = resampling_methods.get(method, Image.Resampling.NEAREST)
    
    # Resize the content
    resized_img = content_img.resize((new_width, new_height), resampling)
    
    # Step 5: Position the resized content in the target grid
    final_img = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
    
    # Determine positioning (center horizontally, bottom aligned)
    x_offset = (target_width - new_width) // 2
    y_offset = target_height - new_height  # Bottom align
    
    # Paste the resized image onto the canvas
    final_img.paste(resized_img, (x_offset, y_offset), resized_img)
    
    # Step 6: Convert back to pixel grid format
    new_grid = []
    for y in range(target_height):
        new_row = []
        for x in range(target_width):
            pixel = final_img.getpixel((x, y))
            if pixel[3] == 0:
                # Transparent
                new_row.append(None)
            else:
                # Convert RGB to hex
                hex_color = f"#{pixel[0]:02x}{pixel[1]:02x}{pixel[2]:02x}"
                new_row.append(hex_color)
        new_grid.append(new_row)
    
    return new_grid

def ensure_dimensions(pixel_grid: PixelGrid, target_width: int, target_height: int, 
                     method: str = "nearest") -> PixelGrid:
    """
    Ensure that the pixel grid has the exact target dimensions.
    
    Args:
        pixel_grid: Original pixel grid
        target_width: Desired width
        target_height: Desired height
        method: Resizing method ('nearest', 'bilinear', etc.)
        
    Returns:
        Adjusted pixel grid
    """
    # Get current dimensions
    current_height = len(pixel_grid)
    current_width = len(pixel_grid[0]) if current_height > 0 else 0
    
    # No change needed if dimensions already match
    if current_width == target_width and current_height == target_height:
        return pixel_grid
    
    # Use smart resize if we need to adjust dimensions and content may not fill the canvas
    if method != "simple":
        # Try using the smart resize that preserves aspect ratio and aligns content
        try:
            return smart_resize(pixel_grid, target_width, target_height, method)
        except Exception as e:
            logging.warning(f"Smart resize failed: {e}. Falling back to basic resize.")
    
    # For very simple cases (adding/removing rows/columns), handle directly
    if method == "nearest" and (
        abs(current_width - target_width) <= 2 and 
        abs(current_height - target_height) <= 2
    ):
        # Simple adjustment
        adjusted_grid = []
        
        # Handle height
        if current_height < target_height:
            # Add rows
            for i in range(target_height):
                if i < current_height:
                    adjusted_grid.append(pixel_grid[i].copy())
                else:
                    # Copy the last row or leave transparent
                    if i > 0:
                        adjusted_grid.append(pixel_grid[-1].copy())
                    else:
                        adjusted_grid.append([None] * current_width)
        else:
            # Remove rows
            adjusted_grid = pixel_grid[:target_height]
        
        # Handle width
        if current_width != target_width:
            for i in range(len(adjusted_grid)):
                row = adjusted_grid[i]
                if current_width < target_width:
                    # Add columns
                    extra_cols = [None] * (target_width - current_width)
                    row.extend(extra_cols)
                else:
                    # Remove columns
                    adjusted_grid[i] = row[:target_width]
        
        return adjusted_grid
    
    # For more complex resizing, use PIL
    # First render to an image
    temp_palette = []
    color_map = {}
    temp_grid = []
    
    # Preprocess to create a palette and indices
    for row in pixel_grid:
        temp_row = []
        for color in row:
            if color is None:
                temp_row.append(None)
            else:
                if color not in color_map:
                    color_map[color] = len(temp_palette)
                    temp_palette.append(color)
                temp_row.append(color_map[color])
        temp_grid.append(temp_row)
    
    # Render to image
    img = render_pixel_grid(pixel_grid, temp_palette)
    
    # Resize the image
    if method == "nearest":
        resampling = Image.Resampling.NEAREST
    elif method == "bilinear":
        resampling = Image.Resampling.BILINEAR
    elif method == "bicubic":
        resampling = Image.Resampling.BICUBIC
    elif method == "lanczos":
        resampling = Image.Resampling.LANCZOS
    else:
        resampling = Image.Resampling.NEAREST
    
    resized_img = img.resize((target_width, target_height), resample=resampling)
    
    # Extract the pixel data
    new_grid = []
    for y in range(target_height):
        new_row = []
        for x in range(target_width):
            pixel = resized_img.getpixel((x, y))
            if len(pixel) == 4 and pixel[3] == 0:
                # Transparent
                new_row.append(None)
            else:
                # Convert RGB to hex
                hex_color = f"#{pixel[0]:02x}{pixel[1]:02x}{pixel[2]:02x}"
                new_row.append(hex_color)
        new_grid.append(new_row)
    
    return new_grid