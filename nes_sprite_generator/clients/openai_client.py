#!/usr/bin/env python3
import json
import logging
from typing import Dict, Any, Optional

from openai import OpenAI

from .base import BaseClient
from ..config import get_api_key

logger = logging.getLogger(__name__)

class OpenAIClient(BaseClient):
    """Client for generating pixel art using OpenAI models."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """Initialize the OpenAI client with credentials."""
        self.api_key = api_key or get_api_key("openai")
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
    
    def generate_pixel_grid(self, 
                           prompt: str, 
                           width: int = 16, 
                           height: int = 16, 
                           max_colors: int = 16,
                           style: str = "2D pixel art") -> Dict[str, Any]:
        """
        Generate a pixel grid representation directly using OpenAI's function calling.
        
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
                            "description": "A very brief explanation (max 150 characters) of key design choices"
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
        - Keep your explanation EXTREMELY brief - under 150 characters maximum
        
        Your pixel art should look complete and refined, not basic or minimal. Even with limited resolution, 
        create art that feels rich and detailed through clever use of color and subtle shading.
        THE BACKGROUND MUST BE TRANSPARENT - DO NOT USE SOLID COLOR BACKGROUNDS.
        """
        
        # Prepare user message
        user_content = f"""Create a {width}x{height} pixel art of: {prompt}
        
        IMPORTANT REQUIREMENTS:
        1. Create an image EXACTLY {width} pixels wide by {height} pixels tall.
        2. Make the subject LARGE - it should fill most of the canvas.
        3. Use rich colors and subtle gradients to create depth and texture.
        4. Add highlights and shadows to give dimension, not flat colors.
        5. Include small details that make the pixel art feel complete and refined.
        6. For transparency, use [0,0,0,0] as the RGBA value.
        7. Your explanation must be extremely short (under 150 characters).
        
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
            
            return function_args
            
        except Exception as e:
            logger.error(f"Failed to generate pixel art: {e}")
            raise