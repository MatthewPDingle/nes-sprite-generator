#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import json
import time
import concurrent.futures
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from PIL import Image
from openai import OpenAI
import numpy as np

# Filter scikit-learn warnings about CPU cores
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
from sklearn.cluster import KMeans

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class PixelArtGenerator:
    """Generate pixel art using OpenAI models with direct pixel grid generation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """Initialize the pixel art generator with OpenAI API credentials."""
        self.api_key = api_key or self._get_api_key()
        self.client = OpenAI(api_key=self.api_key)
        
        # Handle special o3-mini reasoning effort models
        self.reasoning_effort = None
        if model in ["o3-mini-low", "o3-mini-medium", "o3-mini-high"]:
            parts = model.split("-")
            self.reasoning_effort = parts[-1]  # Get the level (low, medium, high)
            self.model = "o3-mini"  # Actual model name to use
            logger.info(f"Using o3-mini with reasoning_effort: {self.reasoning_effort}")
        else:
            self.model = model
        
    def _get_api_key(self) -> str:
        """Get OpenAI API key from environment or file."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            return api_key
            
        try:
            with open("apikey.txt", "r") as file:
                api_key = file.read().strip()
            if not api_key:
                raise RuntimeError("OpenAI API key not found in apikey.txt.")
            return api_key
        except FileNotFoundError:
            raise RuntimeError("API key not found in environment or apikey.txt file.")
            
    def generate_pixel_grid(self, 
                           prompt: str, 
                           width: int = 16, 
                           height: int = 16, 
                           max_colors: int = 16,
                           style: str = "2D pixel art",
                           feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a pixel grid representation directly using OpenAI's function calling.
        
        Args:
            prompt: Description of the pixel art to create
            width: Width of the pixel canvas
            height: Height of the pixel canvas
            max_colors: Maximum number of unique colors to use
            style: Style guide for the pixel art (e.g., "NES-style", "GameBoy-style")
            feedback: Optional feedback from previous iteration for refinement
            
        Returns:
            Dictionary containing the pixel grid, palette, and metadata
        """
        # Ensure width and height are valid
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}. Both width and height must be positive integers.")
        
        # Define the function to call
        functions = [
            {
                "name": "create_pixel_art",
                "description": "Create a pixel art image based on a description",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pixel_grid": {
                            "type": "array",
                            "description": f"A {height}x{width} grid where each cell is an RGBA color. The outer array represents rows (height), and each inner array represents pixels in that row (width).",
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "description": "RGBA color value where each component is 0-255",
                                    "items": {"type": "integer", "minimum": 0, "maximum": 255},
                                    "minItems": 4,
                                    "maxItems": 4
                                },
                                "minItems": width,
                                "maxItems": width
                            },
                            "minItems": height,
                            "maxItems": height
                        },
                        "palette": {
                            "type": "array",
                            "description": f"The color palette used, with a maximum of {max_colors} unique colors.",
                            "items": {
                                "type": "array",
                                "description": "RGBA color value",
                                "items": {"type": "integer", "minimum": 0, "maximum": 255},
                                "minItems": 4,
                                "maxItems": 4
                            },
                            "maxItems": max_colors
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Brief explanation of design choices and color palette"
                        }
                    },
                    "required": ["pixel_grid", "palette", "explanation"]
                }
            }
        ]
        
        # Prepare system message
        system_content = f"""You are a master pixel art designer specializing in {style}.
        You create beautiful, expressive pixel art within tight constraints, maximizing detail and visual appeal even in small canvases.
        
        CRITICAL GUIDELINES:
        - You MUST create a pixel grid EXACTLY {width} pixels wide by {height} pixels tall
        - You MUST use a fully transparent background (NOT colored backgrounds)
        - Always use RGBA format [0,0,0,0] for transparent pixels
        - The subject should fill the majority of the available canvas, not tiny sprites in a large empty space
        - Use rich color gradients and shading to create depth and dimension
        - Aim to use the full range of your palette for richness and detail
        - Create bold, visually distinct sprites with clear silhouettes and character
        - Use color strategically for emphasis, depth, and texture - including gradients within areas to show form
        - Employ techniques like selective dithering and careful anti-aliasing when appropriate
        - Add small details and texture to break up large areas of solid color
        - Use highlights and shadows to create a 3D effect in the 2D space
        
        Your pixel art should look complete and refined, not basic or minimal. Even with limited resolution, 
        create art that feels rich and detailed through clever use of color and subtle shading.
        THE BACKGROUND MUST BE TRANSPARENT - DO NOT USE SOLID COLOR BACKGROUNDS.
        """
        
        # Prepare user message based on whether feedback is provided
        if feedback:
            user_content = f"""Create a {width}x{height} pixel art of: {prompt}
            
            Based on previous feedback:
            {feedback}
            
            IMPORTANT REQUIREMENTS:
            1. Create an image EXACTLY {width} pixels wide by {height} pixels tall.
            2. Make the subject LARGE - it should fill most of the canvas.
            3. Use rich colors and subtle gradients to create depth and texture.
            4. Add highlights and shadows to give dimension, not flat colors.
            5. Include small details that make the pixel art feel complete and refined.
            6. For transparency, use [0,0,0,0] as the RGBA value.
            
            Your goal is to create rich, detailed pixel art that makes maximum use of the limited canvas.
            Don't create tiny sprites with lots of empty space around them."""
        else:
            user_content = f"""Create a {width}x{height} pixel art of: {prompt}
            
            IMPORTANT REQUIREMENTS:
            1. Create an image EXACTLY {width} pixels wide by {height} pixels tall.
            2. Make the subject LARGE - it should fill most of the canvas.
            3. Use rich colors and subtle gradients to create depth and texture.
            4. Add highlights and shadows to give dimension, not flat colors.
            5. Include small details that make the pixel art feel complete and refined.
            6. For transparency, use [0,0,0,0] as the RGBA value.
            
            Your goal is to create rich, detailed pixel art that makes maximum use of the limited canvas.
            Don't create tiny sprites with lots of empty space around them."""
        
        # Assemble messages
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        # Make the API call
        try:
            # Prepare the request parameters
            request_params = {
                "model": self.model,
                "messages": messages,
                "tools": [{"type": "function", "function": functions[0]}],
                "tool_choice": {"type": "function", "function": {"name": "create_pixel_art"}}
            }
            
            # Add reasoning_effort if using o3-mini with specific level
            if self.model == "o3-mini" and self.reasoning_effort in ["low", "medium", "high"]:
                request_params["reasoning_effort"] = self.reasoning_effort
                logger.info(f"Using reasoning_effort: {self.reasoning_effort}")
            
            response = self.client.chat.completions.create(**request_params)
            
            # Extract the function call arguments
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            
            # Log the generation
            self._log_generation(prompt, function_args, feedback is not None)
            
            return function_args
            
        except Exception as e:
            logger.error(f"Failed to generate pixel art: {e}")
            raise
    
    def render_pixel_grid(self, pixel_data: Dict[str, Any], output_file: str) -> str:
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
    
    def _log_generation(self, prompt: str, result: Dict[str, Any], is_refinement: bool) -> None:
        """Log the pixel art generation details."""
        log_file = "pixel_art_generations.log"
        timestamp = datetime.now().isoformat()
        
        with open(log_file, "a", encoding="utf-8") as f:  # Fix encoding issue
            f.write(f"=== {'Refinement' if is_refinement else 'Generation'} at {timestamp} ===\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Model: {self.model}")
            if self.model == "o3-mini" and self.reasoning_effort:
                f.write(f" with reasoning_effort: {self.reasoning_effort}")
            f.write("\n")
            f.write(f"Explanation: {result.get('explanation', 'No explanation provided')}\n")
            f.write(f"Palette size: {len(result.get('palette', []))}\n")
            f.write(f"Canvas size: {len(result['pixel_grid'])}x{len(result['pixel_grid'][0])}\n")
            f.write("="*40 + "\n\n")
    
    def create_preview(self, pixel_data: Dict[str, Any], output_file: str, scale: int = 8) -> str:
        """
        Create a preview of the pixel art with a grid overlay and color palette.
        
        Args:
            pixel_data: Dictionary with pixel_grid and palette
            output_file: Path to save the preview image
            scale: Scale factor for the preview
            
        Returns:
            Path to the saved preview image
        """
        pixel_grid = pixel_data["pixel_grid"]
        palette = pixel_data["palette"]
        height = len(pixel_grid)
        width = len(pixel_grid[0]) if height > 0 else 0
        
        # Create a larger image for the preview
        preview_width = width * scale + 2  # Add border
        preview_height = height * scale + 2  # Add border
        
        preview = Image.new("RGBA", (preview_width, preview_height), (240, 240, 240, 255))
        
        # Draw the upscaled pixel art
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
                        
                        # Skip fully transparent pixels
                        if a == 0:
                            continue
                            
                        # Draw the upscaled pixel
                        for dy in range(scale):
                            for dx in range(scale):
                                preview.putpixel((x * scale + dx + 1, y * scale + dy + 1), rgba)
                    except (ValueError, TypeError):
                        # If conversion fails, skip this pixel
                        logger.warning(f"Invalid pixel value at ({x}, {y}): {pixel_value}. Skipping.")
                else:
                    # If format is incorrect, skip this pixel
                    logger.warning(f"Invalid pixel format at ({x}, {y}): {pixel_value}. Skipping.")
        
        # Draw grid lines
        grid_color = (180, 180, 180, 255)
        for x in range(width + 1):
            for dy in range(height * scale + 2):
                preview.putpixel((x * scale + 1, dy), grid_color)
        
        for y in range(height + 1):
            for dx in range(width * scale + 2):
                preview.putpixel((dx, y * scale + 1), grid_color)
        
        preview.save(output_file)
        logger.info(f"Preview saved to {output_file}")
        return output_file
    
    def post_process_image(self, 
                      pixel_data: dict[str, any], 
                      target_width: int, 
                      target_height: int,
                      resize_method: str = "nearest",
                      max_colors: int = 16) -> dict[str, any]:
        """
        Post-process the generated pixel art to fit the target dimensions while preserving aspect ratio.
        
        This function:
        1. Analyzes the image to find the actual content area (ignoring transparent borders)
        2. Crops to that content area
        3. Resizes while preserving aspect ratio to fit either width or height
        4. Positions the result (bottom-aligned for tall images, center-aligned for wide images)
        5. Reduces the color palette to match the maximum allowed colors
        
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
        
        # Step 1: Find the content boundaries (ignore transparent pixels)
        min_x, max_x, min_y, max_y = width, 0, height, 0
        
        for y in range(height):
            for x in range(width):
                # If pixel has any opacity (not fully transparent)
                if pixel_grid[y][x][3] > 0:
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
        
        # Check if we found any non-transparent pixels
        if min_x > max_x or min_y > max_y:
            logger.warning("No non-transparent pixels found in the image!")
            return pixel_data  # Return original data if no content found
        
        # Calculate the content dimensions
        content_width = max_x - min_x + 1
        content_height = max_y - min_y + 1
        
        logger.info(f"Detected content dimensions: {content_width}x{content_height}")
        
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
        new_width = int(content_width * scale_ratio)
        new_height = int(content_height * scale_ratio)
        
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
        
        # Determine positioning
        if width_ratio <= height_ratio:
            # Width-limited: center horizontally, bottom-align
            x_offset = (target_width - new_width) // 2
            y_offset = target_height - new_height  # Bottom align
        else:
            # Height-limited: center horizontally, bottom-align
            x_offset = (target_width - new_width) // 2
            y_offset = target_height - new_height  # Bottom align
        
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
        
        # Step 7: Reduce the color palette if needed
        if max_colors > 0:
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
                kmeans = KMeans(n_clusters=max_colors, random_state=42, n_init=10)
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
        
        # Create a new pixel_data dictionary with the processed grid
        processed_data = pixel_data.copy()
        processed_data["pixel_grid"] = processed_grid
        processed_data["original_content_dimensions"] = [content_width, content_height]
        processed_data["resize_method"] = resize_method
        processed_data["scaling_ratio"] = scale_ratio
        processed_data["final_dimensions"] = [target_width, target_height]
        
        # Log the post-processing details
        logger.info(f"Post-processed image: content size {content_width}x{content_height} → target size {target_width}x{target_height}")
        logger.info(f"Resize method: {resize_method}, position: x={x_offset}, y={y_offset}")
        
        return processed_data
    
    def generate_sprite_sheet(self, 
                             prompt: str, 
                             width: int = 16, 
                             height: int = 16, 
                             frames: int = 4,
                             animation_type: str = "walk cycle",
                             max_colors: int = 16,
                             style: str = "2D pixel art") -> Dict[str, Any]:
        """
        Generate a sprite sheet with multiple animation frames.
        
        Args:
            prompt: Description of the sprite to create
            width: Width of each frame
            height: Height of each frame
            frames: Number of animation frames
            animation_type: Type of animation (e.g., "walk cycle", "idle", "attack")
            max_colors: Maximum number of colors
            style: Art style
            
        Returns:
            Dictionary with frames, palette, and metadata
        """
        # Define the function to call
        functions = [
            {
                "name": "create_sprite_sheet",
                "description": f"Create a {frames}-frame sprite sheet for a {animation_type} animation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "frames": {
                            "type": "array",
                            "description": f"Array of {frames} animation frames, each a {height}x{width} pixel grid",
                            "items": {
                                "type": "array",
                                "description": "A single frame's pixel grid",
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "description": "RGBA color value",
                                        "items": {"type": "integer", "minimum": 0, "maximum": 255},
                                        "minItems": 4,
                                        "maxItems": 4
                                    },
                                    "minItems": width,
                                    "maxItems": width
                                },
                                "minItems": height,
                                "maxItems": height
                            },
                            "minItems": frames,
                            "maxItems": frames
                        },
                        "palette": {
                            "type": "array",
                            "description": f"Color palette used across all frames (max {max_colors} colors)",
                            "items": {
                                "type": "array",
                                "description": "RGBA color value",
                                "items": {"type": "integer", "minimum": 0, "maximum": 255},
                                "minItems": 4,
                                "maxItems": 4
                            },
                            "maxItems": max_colors
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Explanation of animation design and how frames progress"
                        }
                    },
                    "required": ["frames", "palette", "explanation"]
                }
            }
        ]
        
        # Prepare the message
        messages = [
                            {
                "role": "system", 
                "content": f"""You are a master pixel art animator specializing in {style} game sprites.
                You create cohesive, expressive animation cycles with smooth transitions between frames.
                
                CRITICAL GUIDELINES:
                - Use a rich and consistent palette of colors across all frames
                - The character MUST be large and fill most of the available canvas
                - Create detailed sprites with shading, highlights, and texture
                - Use color gradients and dithering techniques to add depth and dimension
                - Maintain perfect consistency between frames (same character proportions and details)
                - Create a smooth {animation_type} with natural progression and fluid movement
                - Focus on key poses and anticipation for expressive, dynamic movement
                - Add small details and secondary motion elements for richness
                - Use [0,0,0,0] for transparent pixels (only at the edges where needed)
                
                Your animation should look professional and refined, not basic or minimal.
                """
            },
            {
                "role": "user",
                "content": f"""Create a {frames}-frame {animation_type} animation for: {prompt}
                
                IMPORTANT REQUIREMENTS:
                1. Each frame MUST be EXACTLY {width}x{height} pixels.
                2. Make the character LARGE - it should fill most of the canvas.
                3. Use rich colors and subtle gradients to create depth and texture.
                4. Add highlights and shadows to give dimension - avoid flat coloring.
                5. Create smooth, natural transitions between frames.
                6. Include small details that make the animation feel professional.
                
                Your goal is to create a rich, detailed animation that makes maximum use of the limited canvas.
                """
            }
        ]
        
        # Make the API call
        try:
            # Prepare the request parameters
            request_params = {
                "model": self.model,
                "messages": messages,
                "tools": [{"type": "function", "function": functions[0]}],
                "tool_choice": {"type": "function", "function": {"name": "create_sprite_sheet"}}
            }
            
            # Add reasoning_effort if using o3-mini with specific level
            if self.model == "o3-mini" and self.reasoning_effort in ["low", "medium", "high"]:
                request_params["reasoning_effort"] = self.reasoning_effort
                logger.info(f"Using reasoning_effort: {self.reasoning_effort}")
            
            response = self.client.chat.completions.create(**request_params)
            
            # Extract the function call arguments
            tool_call = response.choices[0].message.tool_calls[0]
            sprite_sheet_data = json.loads(tool_call.function.arguments)
            
            # Log the generation
            logger.info(f"Generated sprite sheet with {frames} frames for '{prompt}'")
            
            return sprite_sheet_data
            
        except Exception as e:
            logger.error(f"Failed to generate sprite sheet: {e}")
            raise
    
    def render_sprite_sheet(self, 
                           sprite_data: Dict[str, Any], 
                           output_file: str,
                           layout: str = "horizontal") -> str:
        """
        Render a sprite sheet to an image file.
        
        Args:
            sprite_data: Dictionary with frames and palette
            output_file: Path to save the output image
            layout: How to arrange frames ("horizontal", "grid", or "vertical")
            
        Returns:
            Path to the saved sprite sheet
        """
        frames = sprite_data["frames"]
        num_frames = len(frames)
        height = len(frames[0])
        width = len(frames[0][0]) if height > 0 else 0
        
        # Determine sprite sheet dimensions based on layout
        if layout == "horizontal":
            sheet_width = width * num_frames
            sheet_height = height
            frame_positions = [(i * width, 0) for i in range(num_frames)]
        elif layout == "vertical":
            sheet_width = width
            sheet_height = height * num_frames
            frame_positions = [(0, i * height) for i in range(num_frames)]
        else:  # grid
            import math
            cols = int(math.ceil(math.sqrt(num_frames)))
            rows = int(math.ceil(num_frames / cols))
            sheet_width = width * cols
            sheet_height = height * rows
            frame_positions = [(x * width, y * height) 
                              for y in range(rows) 
                              for x in range(cols) 
                              if y * cols + x < num_frames]
        
        # Create the sprite sheet image
        sheet = Image.new("RGBA", (sheet_width, sheet_height), (0, 0, 0, 0))
        
        # Place each frame
        for i, frame in enumerate(frames):
            frame_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            
            # Draw the frame
            for y in range(height):
                for x in range(width):
                    # Get the color value and ensure it's properly formatted
                    pixel_value = frame[y][x]
                    
                    # Ensure pixel_value is a list or tuple of exactly 4 integers
                    if isinstance(pixel_value, (list, tuple)) and len(pixel_value) == 4:
                        try:
                            # Convert each value to integer and create a proper RGBA tuple
                            r = int(pixel_value[0])
                            g = int(pixel_value[1])
                            b = int(pixel_value[2])
                            a = int(pixel_value[3])
                            rgba = (r, g, b, a)
                            frame_img.putpixel((x, y), rgba)
                        except (ValueError, TypeError):
                            # If conversion fails, use transparent black
                            logger.warning(f"Invalid pixel value at frame {i}, pos ({x}, {y}): {pixel_value}. Using transparent pixel.")
                            frame_img.putpixel((x, y), (0, 0, 0, 0))
                    else:
                        # If format is incorrect, use transparent black
                        logger.warning(f"Invalid pixel format at frame {i}, pos ({x}, {y}): {pixel_value}. Using transparent pixel.")
                        frame_img.putpixel((x, y), (0, 0, 0, 0))
            
            # Paste the frame onto the sprite sheet
            x, y = frame_positions[i]
            sheet.paste(frame_img, (x, y), frame_img)
        
        # Save the sprite sheet
        sheet.save(output_file)
        logger.info(f"Saved sprite sheet to {output_file}")
        
        return output_file
    
    def create_animation_preview(self, sprite_data: Dict[str, Any], output_file: str, fps: int = 8) -> str:
        """
        Create an animated GIF preview of the sprite sheet.
        
        Args:
            sprite_data: Dictionary with frames and palette
            output_file: Path to save the animated GIF
            fps: Frames per second for the animation
            
        Returns:
            Path to the saved GIF
        """
        frames = sprite_data["frames"]
        num_frames = len(frames)
        height = len(frames[0])
        width = len(frames[0][0]) if height > 0 else 0
        
        # Create PIL images for each frame
        frame_images = []
        for frame in frames:
            img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            
            # Draw the frame
            for y in range(height):
                for x in range(width):
                    # Get the color value and ensure it's properly formatted
                    pixel_value = frame[y][x]
                    
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
                            logger.warning(f"Invalid pixel value in animation frame, pos ({x}, {y}): {pixel_value}. Using transparent pixel.")
                            img.putpixel((x, y), (0, 0, 0, 0))
                    else:
                        # If format is incorrect, use transparent black
                        logger.warning(f"Invalid pixel format in animation frame, pos ({x}, {y}): {pixel_value}. Using transparent pixel.")
                        img.putpixel((x, y), (0, 0, 0, 0))
            
            # Upscale the frame for better visibility
            img = img.resize((width * 8, height * 8), Image.NEAREST)
            
            frame_images.append(img)
        
        # Save as animated GIF
        frame_images[0].save(
            output_file,
            save_all=True,
            append_images=frame_images[1:],
            optimize=False,
            duration=int(1000 / fps),  # ms per frame
            loop=0  # Loop forever
        )
        
        logger.info(f"Saved animation preview to {output_file}")
        return output_file

def fix_dimensions(pixel_data, target_width, target_height):
    """
    Safely fix the dimensions of a pixel grid to match the target dimensions.
    Handles cases where the model generated incorrect dimensions.
    
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
    
    logger.warning(f"Model generated incorrect dimensions: {grid_width}x{grid_height}, requested: {target_width}x{target_height}")
    
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
    logger.info("Fixed dimensions to match requested size")
    
    return fixed_data

# New function to process a single version of pixel art
def process_single_version(version, args, generator, total_versions):
    """Process a single version of pixel art generation"""
    try:
        # Adjust output filename for multiple versions
        if total_versions > 1:
            base, ext = os.path.splitext(args.output)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S") + f"_{version}"
            version_output = f"{base}_{timestamp}{ext}"
            logger.info(f"Starting generation of version {version}/{total_versions}...")
        else:
            version_output = args.output
        
        # Generate a pixel art image
        pixel_data = generator.generate_pixel_grid(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            max_colors=args.colors,
            style=args.style
        )
        
        # Verify the required fields exist in the response
        if not all(key in pixel_data for key in ["pixel_grid", "palette"]):
            logger.error(f"Version {version}: API response missing required fields")
            missing_keys = [key for key in ["pixel_grid", "palette"] if key not in pixel_data]
            error_msg = f"API response missing: {', '.join(missing_keys)}"
            return {
                "version": version,
                "success": False,
                "error": error_msg,
                "output_file": version_output
            }
        
        # Fix dimensions if needed
        pixel_data = fix_dimensions(pixel_data, args.width, args.height)
        
        # Apply post-processing if requested
        if args.post_process:
            logger.info(f"Post-processing image to fit content to {args.width}x{args.height}...")
            pixel_data = generator.post_process_image(
                pixel_data, 
                args.width, 
                args.height,
                args.resize_method,
                args.colors
            )
        
        # Save the raw pixel art
        generator.render_pixel_grid(pixel_data, version_output)
        
        # Save a preview with upscaling (only if requested)
        preview_file = None
        if args.preview:
            preview_file = f"preview_{version_output}"
            generator.create_preview(pixel_data, preview_file, scale=8)
        
        # Return results for this version
        return {
            "version": version,
            "output_file": version_output,
            "preview_file": preview_file,
            "pixel_data": pixel_data,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error processing version {version}: {e}")
        # Try to determine output filename even in case of error
        output_file = None
        if total_versions > 1:
            base, ext = os.path.splitext(args.output)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S") + f"_{version}"
            output_file = f"{base}_{timestamp}{ext}"
        else:
            output_file = args.output
            
        return {
            "version": version,
            "success": False,
            "error": str(e),
            "output_file": output_file
        }

def main():
    """Main entry point for the pixel art generator."""
    parser = argparse.ArgumentParser(description="Generate pixel art using AI")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Parser for single pixel art generation
    single_parser = subparsers.add_parser("single", help="Generate a single pixel art image")
    single_parser.add_argument("prompt", help="Description of the pixel art to generate")
    single_parser.add_argument("--width", type=int, default=16, help="Width of the pixel canvas")
    single_parser.add_argument("--height", type=int, default=16, help="Height of the pixel canvas")
    single_parser.add_argument("--colors", type=int, default=16, help="Maximum number of colors")
    single_parser.add_argument("--style", type=str, default="2D pixel art", help="Style guide")
    single_parser.add_argument("--output", type=str, default="pixel_art.png", help="Output file name")
    single_parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use")
    single_parser.add_argument("--versions", type=int, default=1, help="Number of versions to generate")
    single_parser.add_argument("--preview", action="store_true", help="Generate a preview image")
    single_parser.add_argument("--post-process", action="store_true", help="Post-process the image to fit content to desired dimensions")
    single_parser.add_argument("--resize-method", type=str, default="nearest", 
                              choices=["nearest", "bilinear", "bicubic", "lanczos"], 
                              help="Method to use for resizing during post-processing")
    single_parser.add_argument("--max-workers", type=int, default=None, 
                              help="Maximum number of parallel workers for generation (default: CPU count)")
    
    # Parser for sprite sheet generation
    sprite_parser = subparsers.add_parser("sprite", help="Generate a sprite sheet with animation frames")
    sprite_parser.add_argument("prompt", help="Description of the sprite to generate")
    sprite_parser.add_argument("--width", type=int, default=16, help="Width of each frame")
    sprite_parser.add_argument("--height", type=int, default=16, help="Height of each frame")
    sprite_parser.add_argument("--frames", type=int, default=4, help="Number of animation frames")
    sprite_parser.add_argument("--animation", type=str, default="walk cycle", help="Animation type")
    sprite_parser.add_argument("--colors", type=int, default=16, help="Maximum number of colors")
    sprite_parser.add_argument("--style", type=str, default="2D pixel art", help="Style guide")
    sprite_parser.add_argument("--output", type=str, default="sprite_sheet.png", help="Output file name")
    sprite_parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use")
    sprite_parser.add_argument("--fps", type=int, default=8, help="Animation frames per second for preview")
    sprite_parser.add_argument("--preview", action="store_true", help="Generate a preview GIF")
    sprite_parser.add_argument("--post-process", action="store_true", help="Post-process each frame to fit content to desired dimensions")
    sprite_parser.add_argument("--resize-method", type=str, default="nearest", 
                              choices=["nearest", "bilinear", "bicubic", "lanczos"], 
                              help="Method to use for resizing during post-processing")
    
    # Parser for refinement
    refine_parser = subparsers.add_parser("refine", help="Refine existing pixel art")
    refine_parser.add_argument("input_file", help="JSON file with pixel data to refine")
    refine_parser.add_argument("prompt", help="Original or updated description")
    refine_parser.add_argument("--feedback", type=str, help="Feedback for refinement")
    refine_parser.add_argument("--output", type=str, default=None, help="Output file name")
    refine_parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use")
    refine_parser.add_argument("--preview", action="store_true", help="Generate a preview image")
    refine_parser.add_argument("--post-process", action="store_true", help="Post-process the image to fit content to desired dimensions")
    refine_parser.add_argument("--resize-method", type=str, default="nearest", 
                              choices=["nearest", "bilinear", "bicubic", "lanczos"], 
                              help="Method to use for resizing during post-processing")
    
    args = parser.parse_args()
    
    # Initialize the generator
    model = args.model if hasattr(args, 'model') else "gpt-4o"
    generator = PixelArtGenerator(model=model)
    
    # Handle different commands
    if args.command == "single":
        # Use concurrent.futures for parallel processing if multiple versions
        if args.versions > 1:
            print(f"Generating {args.versions} versions in parallel...")
            
            # Determine max workers - if not specified, use CPU count
            max_workers = args.max_workers if hasattr(args, 'max_workers') and args.max_workers else None
            
            # Create a ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit each version for processing
                futures = [
                    executor.submit(process_single_version, version, args, generator, args.versions) 
                    for version in range(1, args.versions + 1)
                ]
                
                # Process results as they come in
                completed = 0
                success_count = 0
                results = []
                
                # Create a progress display
                print(f"0/{args.versions} versions completed", end='\r')
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        completed += 1
                        
                        # Update progress
                        print(f"{completed}/{args.versions} versions completed ({success_count} successful)", end='\r')
                        
                        # If successful, provide immediate feedback
                        if result["success"]:
                            success_count += 1
                            if "pixel_data" in result:
                                pixel_data = result["pixel_data"]
                                logger.info(f"Completed version {result['version']}/{args.versions}: {result['output_file']}")
                    except Exception as e:
                        logger.error(f"Error in version processing: {e}")
                
                print(f"\nAll {completed} versions completed! ({success_count} successful)")
                
                # Print summary of all successful versions
                successful = [r for r in results if r["success"] and "pixel_data" in r]
                if successful:
                    print("\n=== Generated Versions ===")
                    for result in successful:
                        version = result["version"]
                        output_file = result["output_file"]
                        preview_file = result["preview_file"]
                        pixel_data = result["pixel_data"]
                        
                        print(f"\nVersion {version}:")
                        print(f"Image: {output_file}")
                        if preview_file:
                            print(f"Preview: {preview_file}")
                        
                        # Safely access the palette information
                        if "palette" in pixel_data:
                            print(f"Palette: {len(pixel_data['palette'])} colors used out of {args.colors} maximum")
                        else:
                            print(f"Palette information not available")
                
                # If any failed, print those as well
                failed = [r for r in results if not r["success"]]
                if failed:
                    print("\n=== Failed Versions ===")
                    for result in failed:
                        print(f"Version {result['version']}: {result['error']}")
        else:
            # Process single version (original code path)
            result = process_single_version(1, args, generator, 1)
            
            if result["success"] and "pixel_data" in result:
                pixel_data = result["pixel_data"]
                print(f"\nPixel Art Generation Complete!")
                print(f"Image: {result['output_file']}")
                if result["preview_file"]:
                    print(f"Preview: {result['preview_file']}")
                print("\nExplanation:")
                print(pixel_data.get("explanation", "No explanation provided"))
                
                # Safely access palette information
                if "palette" in pixel_data:
                    print(f"\nPalette: {len(pixel_data['palette'])} colors used out of {args.colors} maximum")
                else:
                    print("\nPalette information not available")
                
                # Provide info about model and reasoning level if using o3-mini
                if args.model.startswith("o3-mini-"):
                    print(f"\nUsing o3-mini with reasoning effort: {args.model.split('-')[-1]}")
                    print("Higher reasoning effort levels may produce more detailed and refined pixel art.")
            else:
                print(f"\nError generating pixel art: {result['error']}")
        
    elif args.command == "sprite":
        # Generate a sprite sheet
        sprite_data = generator.generate_sprite_sheet(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            frames=args.frames,
            animation_type=args.animation,
            max_colors=args.colors,
            style=args.style
        )
        
        # Fix dimensions for each frame if needed
        for i, frame in enumerate(sprite_data["frames"]):
            frame_data = {"pixel_grid": frame, "palette": sprite_data["palette"]}
            fixed_frame_data = fix_dimensions(frame_data, args.width, args.height)
            sprite_data["frames"][i] = fixed_frame_data["pixel_grid"]
        
        # Apply post-processing to each frame if requested
        if args.post_process:
            logger.info(f"Post-processing frames to fit content to {args.width}x{args.height}...")
            
            # Create a single-frame data structure for each frame
            for i, frame in enumerate(sprite_data["frames"]):
                frame_data = {
                    "pixel_grid": frame,
                    "palette": sprite_data["palette"]
                }
                
                # Process this frame
                processed_frame_data = generator.post_process_image(
                    frame_data,
                    args.width,
                    args.height,
                    args.resize_method,
                    args.colors
                )
                
                # Update the frame in the sprite sheet
                sprite_data["frames"][i] = processed_frame_data["pixel_grid"]
        
        # Save the sprite sheet
        generator.render_sprite_sheet(sprite_data, args.output)
        
        # Create an animated preview (only if requested)
        if args.preview:
            preview_file = os.path.splitext(args.output)[0] + "_preview.gif"
            generator.create_animation_preview(sprite_data, preview_file, fps=args.fps)
        
        print(f"\nSprite Sheet Generation Complete!")
        print(f"Sprite Sheet: {args.output}")
        if args.preview:
            print(f"Animation Preview: {preview_file}")
        print("\nExplanation:")
        print(sprite_data.get("explanation", "No explanation provided"))
        print(f"\nFrames: {len(sprite_data['frames'])}")
        print(f"Palette: {len(sprite_data['palette'])} colors used out of {args.colors} maximum")
        
        # Provide info about model and reasoning level if using o3-mini
        if args.model.startswith("o3-mini-"):
            print(f"\nUsing o3-mini with reasoning effort: {args.model.split('-')[-1]}")
            print("Higher reasoning effort levels may produce more detailed and refined pixel art.")
        
    elif args.command == "refine":
        # Load existing pixel data
        with open(args.input_file, "r") as f:
            existing_data = json.load(f)
        
        # Get feedback for refinement
        feedback = args.feedback if args.feedback else input("Enter feedback for refinement: ")
        
        # Determine output file
        if not args.output:
            base, ext = os.path.splitext(args.input_file)
            output_file = f"{base}_refined{ext}"
        else:
            output_file = args.output
        
        # Get dimensions from existing data
        width = len(existing_data["pixel_grid"][0])
        height = len(existing_data["pixel_grid"])
        max_colors = len(existing_data["palette"])
        
        # Generate refined pixel art
        refined_data = generator.generate_pixel_grid(
            prompt=args.prompt,
            width=width,
            height=height,
            max_colors=max_colors,
            feedback=feedback
        )
        
        # Fix dimensions if needed
        refined_data = fix_dimensions(refined_data, width, height)
            
        # Apply post-processing if requested
        if args.post_process:
            logger.info(f"Post-processing image to fit content to {width}x{height}...")
            refined_data = generator.post_process_image(
                refined_data, 
                width, 
                height,
                args.resize_method,
                max_colors
            )
        
        # Save the refined pixel art
        png_file = output_file if output_file.endswith(".png") else output_file + ".png"
        generator.render_pixel_grid(refined_data, png_file)
        
        # Save a preview if requested
        if args.preview:
            preview_file = f"preview_{png_file}"
            generator.create_preview(refined_data, preview_file, scale=8)
        
        print(f"\nRefinement Complete!")
        print(f"Refined Image: {png_file}")
        if args.preview:
            print(f"Preview: {preview_file}")
        print("\nExplanation:")
        print(refined_data.get("explanation", "No explanation provided"))
        
        # Provide info about model and reasoning level if using o3-mini
        if args.model.startswith("o3-mini-"):
            print(f"\nUsing o3-mini with reasoning effort: {args.model.split('-')[-1]}")
            print("Higher reasoning effort levels may produce more detailed and refined pixel art.")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()