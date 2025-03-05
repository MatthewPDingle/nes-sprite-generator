#!/usr/bin/env python3
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from . import create_client, AVAILABLE_MODELS
from .image_utils import (
    render_pixel_grid, post_process_image,
    fix_dimensions, analyze_content_dimensions
)

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class PixelArtGenerator:
    """Generate pixel art using AI models with direct pixel grid generation."""
    
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize the pixel art generator.
        
        Args:
            model: Model identifier to use for generation
            api_key: Optional API key (if not provided, it will be loaded from environment or file)
        """
        # Create the appropriate client based on the model name
        self.client = create_client(model, api_key)
        self.model = model
        
        logger.info(f"Initialized PixelArtGenerator with model: {model}")
    
    def generate_pixel_grid(self, 
                           prompt: str, 
                           width: int = 16, 
                           height: int = 16, 
                           max_colors: int = 16,
                           style: str = "2D pixel art") -> Dict[str, Any]:
        """
        Generate a pixel grid representation using the AI service.
        
        Args:
            prompt: Description of the pixel art to create
            width: Width of the pixel canvas
            height: Height of the pixel canvas
            max_colors: Maximum number of unique colors to use
            style: Style guide for the pixel art
            
        Returns:
            Dictionary containing the pixel grid, palette, and metadata
        """
        # Ensure width and height are valid
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}. Both width and height must be positive integers.")
        
        # Delegate to the client
        pixel_data = self.client.generate_pixel_grid(
            prompt=prompt,
            width=width,
            height=height,
            max_colors=max_colors,
            style=style
        )
        
        # Log the generation
        self._log_generation(prompt, pixel_data)
        
        return pixel_data
    
    def _log_generation(self, prompt: str, result: Dict[str, Any]) -> None:
        """Log the pixel art generation details."""
        log_file = "pixel_art_generations.log"
        timestamp = datetime.now().isoformat()
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"=== Generation at {timestamp} ===\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Model: {self.model}\n")
            
            # Add more detailed model info depending on the client type
            if hasattr(self.client, 'reasoning_effort') and self.client.reasoning_effort:
                f.write(f"Reasoning effort: {self.client.reasoning_effort}\n")
            elif hasattr(self.client, 'thinking_level') and self.client.thinking_level:
                f.write(f"Extended thinking level: {self.client.thinking_level}\n")
                
            f.write(f"Explanation: {result.get('explanation', 'No explanation provided')}\n")
            f.write(f"Palette size: {len(result.get('palette', []))}\n")
            
            pixel_grid = result.get('pixel_grid', [])
            if pixel_grid:
                grid_height = len(pixel_grid)
                grid_width = len(pixel_grid[0]) if grid_height > 0 else 0
                f.write(f"Canvas size: {grid_width}x{grid_height}\n")
                
                # Also analyze the actual content dimensions (non-transparent pixels)
                content_width, content_height = analyze_content_dimensions(pixel_grid)
                f.write(f"Content size: {content_width}x{content_height}\n")
            else:
                f.write("Canvas size: 0x0 (empty grid)\n")
                
            f.write("="*40 + "\n\n")
    
    def process_image(self, 
                    pixel_data: Dict[str, Any],
                    output_file: str,
                    post_process: bool = True,  # Changed default to True for better handling
                    target_width: Optional[int] = None,
                    target_height: Optional[int] = None,
                    resize_method: str = "nearest",
                    max_colors: int = 16) -> Dict[str, str]:
        """
        Process and save pixel art data to files.
        Always ensures the output has exactly the requested dimensions.
        
        Args:
            pixel_data: Dictionary with pixel_grid and palette
            output_file: Path to save the output image
            post_process: Whether to post-process the image
            target_width: Target width for post-processing (defaults to original width)
            target_height: Target height for post-processing (defaults to original height)
            resize_method: Method to use for resizing during post-processing
            max_colors: Maximum number of colors allowed in the final image
            
        Returns:
            Dictionary with paths to saved files
        """
        result = {"output": None}
        
        # Verify and get current dimensions
        original_grid = pixel_data["pixel_grid"]
        original_height = len(original_grid)
        original_width = len(original_grid[0]) if original_height > 0 else 0
        
        # Set default target dimensions if not provided
        if target_width is None:
            target_width = original_width
        if target_height is None:
            target_height = original_height
        
        # Analyze content dimensions before any processing
        content_width, content_height = analyze_content_dimensions(original_grid)
        logger.info(f"Original dimensions: {original_width}x{original_height}, Content dimensions: {content_width}x{content_height}")
        
        # ALWAYS strictly fix dimensions first to ensure canvas size matches requested size
        pixel_data = fix_dimensions(pixel_data, target_width, target_height)
        
        # Verify dimensions after fixing
        fixed_grid = pixel_data["pixel_grid"]
        fixed_height = len(fixed_grid)
        fixed_width = len(fixed_grid[0]) if fixed_height > 0 else 0
        
        if fixed_width != target_width or fixed_height != target_height:
            logger.error(f"Dimension fixing failed! Got {fixed_width}x{fixed_height}, expected {target_width}x{target_height}")
            # Try one more time with stricter enforcement
            temp_grid = []
            for y in range(target_height):
                row = []
                for x in range(target_width):
                    if y < fixed_height and x < fixed_width:
                        row.append(fixed_grid[y][x])
                    else:
                        row.append([0, 0, 0, 0])  # Add transparent pixel
                temp_grid.append(row)
            pixel_data["pixel_grid"] = temp_grid[:target_height]  # Cut off any extra rows
        
        # Apply post-processing if requested
        if post_process:
            logger.info(f"Post-processing image to fit content to {target_width}x{target_height}...")
            pixel_data = post_process_image(
                pixel_data, 
                target_width, 
                target_height,
                resize_method,
                max_colors
            )
        
        # One final verification of dimensions
        final_grid = pixel_data["pixel_grid"]
        final_height = len(final_grid)
        final_width = len(final_grid[0]) if final_height > 0 else 0
        
        if final_width != target_width or final_height != target_height:
            logger.error(f"Final dimensions incorrect: {final_width}x{final_height}, forcing exact dimensions")
            # Force the dimensions one last time
            final_fixed_grid = []
            for y in range(target_height):
                row = []
                for x in range(target_width):
                    if y < final_height and x < final_width:
                        row.append(final_grid[y][x])
                    else:
                        row.append([0, 0, 0, 0])  # Add transparent pixel
                final_fixed_grid.append(row)
            pixel_data["pixel_grid"] = final_fixed_grid[:target_height]  # Cut off any extra rows
        
        # Save the pixel art image
        output_path = render_pixel_grid(pixel_data, output_file)
        result["output"] = output_path
        
        return result
    
    @staticmethod
    def list_models():
        """List all available models."""
        return AVAILABLE_MODELS