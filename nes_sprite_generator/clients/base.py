#!/usr/bin/env python3
from abc import ABC, abstractmethod

class BaseClient(ABC):
    """Base class for AI service clients."""
    
    @abstractmethod
    def generate_pixel_grid(self, prompt, width, height, max_colors, style, reference_image=None):
        """
        Generate a pixel grid representation using the AI service.
        
        Args:
            prompt: Description of the pixel art to create
            width: Width of the pixel canvas
            height: Height of the pixel canvas
            max_colors: Maximum number of unique colors to use
            style: Style guide for the pixel art
            reference_image: Optional PIL Image object to use as reference
            
        Returns:
            Dictionary containing the pixel grid, palette, and metadata
        """
        pass