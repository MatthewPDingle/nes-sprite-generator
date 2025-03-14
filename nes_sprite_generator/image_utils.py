"""
Utility functions for image processing and rendering pixel art.
"""
import logging
import random
import colorsys
from typing import Dict, List, Tuple, Any, Optional, Union, Set
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