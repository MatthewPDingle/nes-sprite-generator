import os
import json
import logging
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import random

from . import create_client
from .image_utils import render_pixel_grid, optimize_colors, ensure_dimensions

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class PixelArtGenerator:
    """Class for generating pixel art using AI models."""
    
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize the pixel art generator.
        
        Args:
            model: The AI model to use
            api_key: Optional API key (if not provided, will be loaded from environment or config)
        """
        self.model = model
        self.client = create_client(model, api_key)
        logger.info(f"PixelArtGenerator initialized with model: {model}")
        
    def generate_pixel_grid(self, prompt: str, width: int = 16, height: int = 16, 
                           max_colors: int = 16, style: str = "2D pixel art") -> Dict[str, Any]:
        """
        Generate a pixel art grid based on the prompt.
        
        Args:
            prompt: Description of the desired pixel art
            width: Width of the pixel grid
            height: Height of the pixel grid
            max_colors: Maximum number of colors to use
            style: Style guide for the pixel art
            
        Returns:
            Dictionary containing the pixel grid and palette
        """
        # Construct the detailed prompt
        system_prompt = self._get_system_prompt(width, height, max_colors, style)
        user_prompt = f"Create pixel art of: {prompt}"
        
        # Call the AI model
        try:
            logger.info(f"Calling {self.model} to generate pixel art of: {prompt}")
            result = self.client.generate_pixel_art(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                width=width,
                height=height,
                max_colors=max_colors
            )
            
            logger.info(f"Successfully generated pixel art response")
            return result
        except Exception as e:
            logger.error(f"Error generating pixel art: {e}")
            raise
    
    def process_image(self, pixel_data: Dict[str, Any], output_file: str, 
                     post_process: bool = True, target_width: Optional[int] = None, 
                     target_height: Optional[int] = None, resize_method: str = "nearest",
                     max_colors: Optional[int] = None) -> Dict[str, Any]:
        """
        Process the pixel data and save it as an image.
        
        Args:
            pixel_data: The pixel data from the generate_pixel_grid method
            output_file: Path to save the output image
            post_process: Whether to post-process the image to correct dimensions
            target_width: Target width for post-processing
            target_height: Target height for post-processing
            resize_method: Method to use for resizing ("nearest", "bilinear", etc.)
            max_colors: Maximum number of colors (for palette optimization)
            
        Returns:
            Dictionary with information about the saved image
        """
        # Verify we have the required data
        if not all(key in pixel_data for key in ["pixel_grid", "palette"]):
            missing_keys = [key for key in ["pixel_grid", "palette"] if key not in pixel_data]
            raise ValueError(f"Pixel data missing required fields: {', '.join(missing_keys)}")
        
        # Get the pixel grid and palette
        pixel_grid = pixel_data["pixel_grid"]
        palette = pixel_data["palette"]
        
        # Log the original dimensions
        orig_height = len(pixel_grid)
        orig_width = len(pixel_grid[0]) if orig_height > 0 else 0
        
        # Post-process if needed
        if post_process:
            if target_width is not None and target_height is not None:
                grid_width = orig_width
                grid_height = orig_height
                
                # Check if dimensions match
                if grid_width != target_width or grid_height != target_height:
                    logger.warning(f"Image dimensions mismatch. Got {grid_width}x{grid_height}, expected {target_width}x{target_height}")
                    logger.info(f"Post-processing to correct dimensions using {resize_method} method")
                    
                    # Ensure correct dimensions
                    pixel_grid = ensure_dimensions(
                        pixel_grid=pixel_grid,
                        target_width=target_width,
                        target_height=target_height,
                        method=resize_method
                    )
            
            # Optimize colors if needed
            if max_colors is not None and len(palette) > max_colors:
                logger.warning(f"Too many colors in palette: {len(palette)}. Reducing to {max_colors}")
                pixel_grid, palette = optimize_colors(pixel_grid, palette, max_colors)
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Render and save the image
        img = render_pixel_grid(pixel_grid, palette, scale=8)  # 8x scale for better visibility
        img.save(output_file)
        
        return {
            "output": output_file,
            "dimensions": (len(pixel_grid[0]), len(pixel_grid)),
            "original_dimensions": (orig_width, orig_height),
            "colors": len(palette)
        }
    
    def _get_system_prompt(self, width: int, height: int, max_colors: int, style: str) -> str:
        """
        Create the system prompt for the AI model.
        
        Args:
            width: Width of the pixel grid
            height: Height of the pixel grid
            max_colors: Maximum number of colors to use
            style: Style guide for the pixel art
            
        Returns:
            System prompt string
        """
        return f"""You are a master pixel artist specializing in {style} for the NES system.

Your task is to create a detailed pixel art sprite with the following specifications:
- Dimensions: {width}x{height} pixels (width x height)
- Color palette: Maximum of {max_colors} colors (including transparency)
- Style: {style}

You must provide your response in the following JSON format:
{{
  "pixel_grid": [
    ["#RRGGBB", "#RRGGBB", ...],  // First row (width elements)
    ["#RRGGBB", "#RRGGBB", ...],  // Second row
    ...  // height rows total
  ],
  "palette": ["#RRGGBB", "#RRGGBB", ...],  // Unique colors used (maximum {max_colors})
  "explanation": "Brief explanation of your design choices and any NES-specific limitations you considered"
}}

Important guidelines:
1. Create a visually clear and recognizable sprite that would work well in an NES game
2. Use transparency (represented as null) for pixels that should be transparent
3. Ensure your palette has no more than {max_colors} unique colors
4. Make every pixel count - NES sprites need to be recognizable despite limited resolution
5. Follow NES technical limitations when possible (limited sprites per scanline, etc.)
6. Your pixel_grid must be EXACTLY {height} rows with EXACTLY {width} elements per row
7. JSON format must be valid and match the structure above"""