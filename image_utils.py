#!/usr/bin/env python3
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any

from PIL import Image
import numpy as np

# Filter scikit-learn warnings about CPU cores
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not installed. Color palette reduction will be unavailable.")

logger = logging.getLogger(__name__)

def analyze_content_dimensions(pixel_grid: List[List[List[int]]]) -> Tuple[int, int]:
    """
    Analyze the pixel grid to determine the actual dimensions of non-transparent content.
    
    Args:
        pixel_grid: The pixel grid data (list of rows, each with a list of RGBA pixels)
        
    Returns:
        Tuple of (content_width, content_height) excluding transparent pixels
    """
    if not pixel_grid:
        return 0, 0
    
    height = len(pixel_grid)
    width = len(pixel_grid[0]) if height > 0 else 0
    
    # Find the boundaries of non-transparent content
    min_x, max_x, min_y, max_y = width, 0, height, 0
    
    for y in range(height):
        for x in range(width):
            # If pixel has any opacity (not fully transparent)
            if pixel_grid[y][x][3] > 0:
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
    
    # If we didn't find any non-transparent pixels
    if min_x > max_x or min_y > max_y:
        return 0, 0
    
    # Calculate content dimensions
    content_width = max_x - min_x + 1
    content_height = max_y - min_y + 1
    
    return content_width, content_height

def fix_dimensions(pixel_data: Dict[str, Any], target_width: int, target_height: int) -> Dict[str, Any]:
    """
    Strictly enforce the dimensions of a pixel grid to match the target dimensions.
    Always return a grid exactly matching the requested size, trimming or padding as needed.
    
    Args:
        pixel_data: Dictionary with pixel_grid and other data
        target_width: Desired width of the output grid
        target_height: Desired height of the output grid
        
    Returns:
        Updated pixel_data with corrected dimensions
    """
    grid = pixel_data["pixel_grid"]
    grid_height = len(grid)
    grid_width = len(grid[0]) if grid_height > 0 else 0
    
    if grid_height == target_height and grid_width == target_width:
        # Dimensions are already correct
        return pixel_data
    
    logger.warning(f"Enforcing dimensions: {grid_width}x{grid_height} → {target_width}x{target_height}")
    
    # Create a new grid with the correct dimensions
    fixed_grid = []
    for y in range(target_height):
        row = []
        for x in range(target_width):
            # Use existing pixel if available, otherwise transparent
            if y < grid_height and x < grid_width:
                try:
                    row.append(grid[y][x])
                except IndexError:
                    # Handle potential index errors
                    row.append([0, 0, 0, 0])  # Transparent pixel
            else:
                row.append([0, 0, 0, 0])  # Transparent pixel
        fixed_grid.append(row)
    
    # Update the pixel_data with the corrected grid
    fixed_data = pixel_data.copy()
    fixed_data["pixel_grid"] = fixed_grid
    
    # Verify the dimensions are now correct
    new_height = len(fixed_grid)
    new_width = len(fixed_grid[0]) if new_height > 0 else 0
    logger.info(f"Fixed dimensions: now {new_width}x{new_height}")
    
    return fixed_data

def render_pixel_grid(pixel_data: Dict[str, Any], output_file: str) -> str:
    """
    Render the pixel grid to an image file.
    
    Args:
        pixel_data: Dictionary with pixel_grid and palette
        output_file: Path to save the output image
        
    Returns:
        Path to the saved image file
    """
    pixel_grid = pixel_data["pixel_grid"]
    height = len(pixel_grid)
    width = len(pixel_grid[0]) if height > 0 else 0
    
    # Create a new image with RGBA mode
    img = Image.new("RGBA", (width, height))
    
    # Place each pixel
    for y in range(height):
        for x in range(width):
            # Get the color value and ensure it's properly formatted
            pixel_value = pixel_grid[y][x]
            
            # Ensure pixel_value is a list or tuple of exactly 4 integers
            if isinstance(pixel_value, (list, tuple)) and len(pixel_value) == 4:
                try:
                    # Convert each value to integer and create a proper RGBA tuple
                    r = int(pixel_value[0])
                    g = int(pixel_value[1])
                    b = int(pixel_value[2])
                    a = int(pixel_value[3])
                    rgba = (r, g, b, a)
                    img.putpixel((x, y), rgba)
                except (ValueError, TypeError):
                    # If conversion fails, use transparent black
                    logger.warning(f"Invalid pixel value at ({x}, {y}): {pixel_value}. Using transparent pixel.")
                    img.putpixel((x, y), (0, 0, 0, 0))
            else:
                # If format is incorrect, use transparent black
                logger.warning(f"Invalid pixel format at ({x}, {y}): {pixel_value}. Using transparent pixel.")
                img.putpixel((x, y), (0, 0, 0, 0))
    
    # Save the image
    img.save(output_file)
    logger.info(f"Saved pixel art to {output_file}")
    
    return output_file

def post_process_image(pixel_data: Dict[str, Any], 
                 target_width: int, 
                 target_height: int,
                 resize_method: str = "nearest",
                 max_colors: int = 16) -> Dict[str, Any]:
    """
    Post-process the generated pixel art to fit the target dimensions while preserving aspect ratio.
    
    This function:
    1. Analyzes the image to find the actual content area (ignoring transparent borders)
    2. Crops to that content area
    3. Resizes while preserving aspect ratio to fit either width or height
    4. Positions the result bottom-aligned and horizontally centered
    5. Reduces the color palette to match the maximum allowed colors
    6. Ensures the final output is exactly the requested dimensions
    
    Args:
        pixel_data: Dictionary with pixel_grid and palette
        target_width: Desired width of the output image
        target_height: Desired height of the output image
        resize_method: Method to use for resizing ('nearest', 'bilinear', 'bicubic', 'lanczos')
        max_colors: Maximum number of colors allowed in the final image
        
    Returns:
        Dictionary with the processed pixel grid
    """
    pixel_grid = pixel_data["pixel_grid"]
    height = len(pixel_grid)
    width = len(pixel_grid[0]) if height > 0 else 0
    
    # Ensure input dimensions match target dimensions
    if height != target_height or width != target_width:
        logger.warning(f"Input grid dimensions ({width}x{height}) don't match target ({target_width}x{target_height})")
        # This shouldn't happen if fix_dimensions was called first, but we'll handle it just in case
        pixel_data = fix_dimensions(pixel_data, target_width, target_height)
        pixel_grid = pixel_data["pixel_grid"]
        height = len(pixel_grid)
        width = len(pixel_grid[0]) if height > 0 else 0
    
    # Step 1: Find the content boundaries (ignore transparent pixels)
    min_x, max_x, min_y, max_y = width, 0, height, 0
    has_content = False
    
    for y in range(height):
        for x in range(width):
            # If pixel has any opacity (not fully transparent)
            if pixel_grid[y][x][3] > 0:
                has_content = True
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
    
    # Check if we found any non-transparent pixels
    if not has_content:
        logger.warning("No non-transparent pixels found in the image!")
        return pixel_data  # Return original data if no content found
    
    # Calculate the content dimensions
    content_width = max_x - min_x + 1
    content_height = max_y - min_y + 1
    
    logger.info(f"Detected content dimensions: {content_width}x{content_height} at position ({min_x},{min_y})")
    
    # If content already fills most of the canvas and has correct aspect ratio, skip intensive processing
    content_area = content_width * content_height
    canvas_area = width * height
    content_ratio = content_area / canvas_area
    
    # Only do the full processing if the content doesn't fill most of the canvas or needs repositioning
    if content_ratio < 0.9 or min_y > 0 or min_x > 0:
        # Step 2: Create a PIL image from the content area
        content_img = Image.new("RGBA", (content_width, content_height))
        
        # Copy only the content area
        for y in range(content_height):
            for x in range(content_width):
                # Get the color value and ensure it's properly formatted
                pixel_value = pixel_grid[min_y + y][min_x + x]
                
                # Ensure pixel_value is a list or tuple of exactly 4 integers
                if isinstance(pixel_value, (list, tuple)) and len(pixel_value) == 4:
                    try:
                        # Convert each value to integer and create a proper RGBA tuple
                        r = int(pixel_value[0])
                        g = int(pixel_value[1])
                        b = int(pixel_value[2])
                        a = int(pixel_value[3])
                        rgba = (r, g, b, a)
                        content_img.putpixel((x, y), rgba)
                    except (ValueError, TypeError):
                        # If conversion fails, use transparent black
                        logger.warning(f"Invalid pixel value at ({x}, {y}): {pixel_value}. Using transparent pixel.")
                        content_img.putpixel((x, y), (0, 0, 0, 0))
                else:
                    # If format is incorrect, use transparent black
                    logger.warning(f"Invalid pixel format at ({x}, {y}): {pixel_value}. Using transparent pixel.")
                    content_img.putpixel((x, y), (0, 0, 0, 0))
        
        # Step 3: Calculate the scaling factor to maintain aspect ratio
        width_ratio = target_width / content_width
        height_ratio = target_height / content_height
        
        # Use the smaller ratio to ensure the image fits within the canvas
        scale_ratio = min(width_ratio, height_ratio)
        
        # Calculate the new dimensions after scaling
        new_width = max(1, int(content_width * scale_ratio))
        new_height = max(1, int(content_height * scale_ratio))
        
        logger.info(f"Scaling content with ratio {scale_ratio:.2f} to {new_width}x{new_height}")
        
        # Step 4: Resize the image while preserving aspect ratio
        resize_modes = {
            "nearest": Image.NEAREST,  # Pixelated look, good for pixel art
            "bilinear": Image.BILINEAR,  # Some smoothing
            "bicubic": Image.BICUBIC,  # More smoothing, can introduce artifacts
            "lanczos": Image.LANCZOS  # High-quality downsampling
        }
        
        if resize_method not in resize_modes:
            logger.warning(f"Unknown resize method: {resize_method}. Using 'nearest' instead.")
            resize_method = "nearest"
        
        resized_img = content_img.resize((new_width, new_height), resize_modes[resize_method])
        
        # Step 5: Create a new blank canvas and position the resized image
        final_img = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
        
        # Determine positioning - always center horizontally and place at bottom of canvas
        x_offset = (target_width - new_width) // 2
        y_offset = target_height - new_height  # Bottom align
        
        # Ensure offsets are valid (non-negative)
        x_offset = max(0, x_offset)
        y_offset = max(0, y_offset)
        
        # Paste the resized image onto the canvas
        final_img.paste(resized_img, (x_offset, y_offset), resized_img)
        
        # Step 6: Convert back to pixel grid format
        processed_grid = []
        for y in range(target_height):
            row = []
            for x in range(target_width):
                rgba = final_img.getpixel((x, y))
                # Convert tuple back to list
                row.append(list(rgba))
            processed_grid.append(row)
    else:
        # Skip intensive processing if content already fills most of the canvas
        logger.info(f"Content already fills {content_ratio:.2%} of canvas, keeping as is")
        processed_grid = pixel_grid
    
    # Step 7: Reduce the color palette if needed
    if max_colors > 0 and SKLEARN_AVAILABLE:
        # Extract all non-transparent pixels
        visible_pixels = []
        pixel_positions = []
        
        for y in range(target_height):
            for x in range(target_width):
                rgba = processed_grid[y][x]
                if rgba[3] > 0:  # Not fully transparent
                    visible_pixels.append(rgba)
                    pixel_positions.append((y, x))
        
        # Count unique colors
        unique_colors = set(tuple(rgba) for rgba in visible_pixels)
        unique_color_count = len(unique_colors)
        
        if unique_color_count > max_colors:
            logger.info(f"Reducing palette from {unique_color_count} to {max_colors} colors")
            
            # Convert pixels to numpy array for K-means
            pixels_array = np.array(visible_pixels)
            
            # Separate RGB and alpha channels
            rgb = pixels_array[:, 0:3]
            alpha = pixels_array[:, 3]
            
            # Apply K-means clustering to find the optimal color palette
            # Use try-except to handle different scikit-learn versions
            try:
                # Try with n_jobs parameter (newer scikit-learn versions)
                kmeans = KMeans(n_clusters=max_colors, random_state=42, n_init=10, n_jobs=1)
                kmeans.fit(rgb)
            except TypeError:
                # Fall back to version without n_jobs for older scikit-learn versions
                try:
                    kmeans = KMeans(n_clusters=max_colors, random_state=42, n_init=10)
                    kmeans.fit(rgb)
                except TypeError:
                    # For very old scikit-learn versions that might use 'init' instead of 'n_init'
                    kmeans = KMeans(n_clusters=max_colors, random_state=42)
                    kmeans.fit(rgb)
            
            # Replace each pixel with its closest centroid
            centroids = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            
            # Reconstruct the palette-reduced image
            for i, (y, x) in enumerate(pixel_positions):
                # Get the cluster center for this pixel
                new_rgb = centroids[labels[i]]
                # Keep the original alpha value
                processed_grid[y][x] = list(new_rgb) + [alpha[i]]
    
    # Step 8: Ensure the final grid has exactly the requested dimensions
    final_height = len(processed_grid)
    final_width = len(processed_grid[0]) if final_height > 0 else 0
    
    if final_height != target_height or final_width != target_width:
        logger.warning(f"Final grid dimensions ({final_width}x{final_height}) don't match target ({target_width}x{target_height})")
        
        # Create a new grid with the exact dimensions
        final_processed_grid = []
        for y in range(target_height):
            row = []
            for x in range(target_width):
                if y < final_height and x < final_width:
                    row.append(processed_grid[y][x])
                else:
                    row.append([0, 0, 0, 0])  # Transparent pixel
            final_processed_grid.append(row)
        
        # Ensure we don't have extra rows
        final_processed_grid = final_processed_grid[:target_height]
        
        processed_grid = final_processed_grid
    
    # Create a new pixel_data dictionary with the processed grid
    processed_data = pixel_data.copy()
    processed_data["pixel_grid"] = processed_grid
    processed_data["original_content_dimensions"] = [content_width, content_height]
    processed_data["resize_method"] = resize_method
    processed_data["final_dimensions"] = [target_width, target_height]
    
    # Log the post-processing details
    logger.info(f"Post-processed image: content size {content_width}x{content_height} → target size {target_width}x{target_height}")
    
    return processed_data