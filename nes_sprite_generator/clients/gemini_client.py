#!/usr/bin/env python3
import json
import logging
from typing import Dict, Any, Optional
import re

import openai

from .base import BaseClient
from ..config import get_api_key

logger = logging.getLogger(__name__)

class GeminiClient(BaseClient):
    """Client for generating pixel art using Google's Gemini models."""
    
    # Define Gemini model constants
    GEMINI_FLASH = "gemini-2.0-flash"
    GEMINI_PRO = "gemini-2.0-pro-exp-02-05"
    GEMINI_FLASH_THINKING = "gemini-2.0-flash-thinking-exp-01-21"
    
    def __init__(self, api_key: Optional[str] = None, model: str = GEMINI_FLASH):
        """
        Initialize the Gemini client with credentials.
        
        Args:
            api_key: Google API key.
            model: Model name or identifier.
        """
        self.api_key = api_key or get_api_key("google")
        self.model = model
        
        # Use OpenAI client configured to use Google's Gemini endpoints
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
        # Handle thinking mode for flash-thinking model
        self.thinking_level = None
        if "thinking" in model.lower():
            # Extract thinking level if present in model name - for future use
            if "-low" in model or "-medium" in model or "-high" in model:
                self.thinking_level = model.split("-")[-1]
            else:
                # Default thinking level
                self.thinking_level = "medium"
            
            logger.info(f"Using Gemini with thinking level: {self.thinking_level}")
        
        logger.info(f"Initialized GeminiClient with model: {model}")
    
    def generate_pixel_art(self, 
                         system_prompt: str,
                         user_prompt: str,
                         width: int = 16, 
                         height: int = 16, 
                         max_colors: int = 16) -> Dict[str, Any]:
        """
        Generate pixel art using Gemini.
        
        Args:
            system_prompt: System prompt with detailed instructions
            user_prompt: User prompt describing what to generate
            width: Width of the pixel canvas
            height: Height of the pixel canvas
            max_colors: Maximum number of colors to use
            
        Returns:
            Dictionary containing the pixel grid, palette, and explanation
        """
        # Check if we're using a thinking model
        is_thinking_model = "thinking" in self.model.lower()
        
        if is_thinking_model:
            # Text-based approach for thinking models (no function calling)
            return self._generate_with_text_based_approach(system_prompt, user_prompt, width, height, max_colors)
        else:
            # Function calling approach for standard models
            return self._generate_with_function_calling(system_prompt, user_prompt, width, height, max_colors)
    
    def _generate_with_function_calling(self, system_prompt, user_prompt, width, height, max_colors):
        """Generate pixel art using function calling (for standard models)."""
        # Define the function to call
        functions = [
            {
                "type": "function",
                "function": {
                    "name": "create_pixel_art",
                    "description": "Create a pixel art image based on a description",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pixel_grid": {
                                "type": "array",
                                "description": f"A {height}x{width} grid where each cell is a hex color string or null for transparency.",
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": ["string", "null"],
                                        "description": "Hex color string (e.g., '#FF0000') or null for transparency"
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
                                    "type": "string",
                                    "description": "Hex color string (e.g., '#FF0000')"
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
            }
        ]
        
        # Assemble messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Make the API call
        try:
            # Prepare the request parameters
            request_params = {
                "model": self.model,
                "messages": messages,
                "tools": functions,
                "tool_choice": {"type": "function", "function": {"name": "create_pixel_art"}}
            }
                
            logger.info(f"Sending function-calling request to Gemini API with model: {self.model}")
            response = self.client.chat.completions.create(**request_params)
            
            # Extract the function call arguments
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            
            # Validate response format
            if not all(key in function_args for key in ["pixel_grid", "palette", "explanation"]):
                missing_keys = [key for key in ["pixel_grid", "palette", "explanation"] if key not in function_args]
                logger.error(f"Gemini response missing required fields: {missing_keys}")
                
                # If palette is missing but we have a pixel grid, we can extract it
                if "pixel_grid" in function_args and "palette" not in function_args:
                    logger.warning("Missing 'palette' in response. Extracting from pixel_grid.")
                    unique_colors = set()
                    for row in function_args["pixel_grid"]:
                        for pixel in row:
                            if pixel is not None:
                                unique_colors.add(pixel)
                    function_args["palette"] = list(unique_colors)
                
                # If explanation is missing, add a default one
                if "explanation" not in function_args:
                    function_args["explanation"] = "Pixel art created with Gemini model."
            
            return function_args
            
        except Exception as e:
            logger.error(f"Failed to generate pixel art with Gemini function calling: {e}")
            
            # Provide additional error context if possible
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                logger.error(f"Response content: {e.response.text}")
                
            raise
            
    def _generate_with_text_based_approach(self, system_prompt, user_prompt, width, height, max_colors):
        """Generate pixel art using a text-based approach (for thinking models)."""
        # Modify the prompts to request hex color format for thinking models
        system_prompt = system_prompt + "\n\nIMPORTANT: For thinking models, use hex color strings (e.g., '#FF0000') instead of RGBA arrays."
        user_prompt = user_prompt + "\n\nUse hex color strings (e.g., '#FF0000') and null for transparency."
        
        # Assemble messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Make the API call
        try:
            # No tools or function calling for thinking models
            request_params = {
                "model": self.model,
                "messages": messages,
                # Thinking models need specific parameters to ensure full output
                "max_tokens": 8192,  # Use a large token limit for response
                "temperature": 0.5   # Lower temperature for more deterministic output
            }
                
            logger.info(f"Sending text-based request to Gemini API with thinking model: {self.model}")
            response = self.client.chat.completions.create(**request_params)
            
            # Extract the text response
            response_text = response.choices[0].message.content
            logger.info(f"Received text response from Gemini (length: {len(response_text)})")
            
            # Extract JSON from the response - handle potential markdown code blocks
            json_content = None
            
            # Try to extract from markdown code blocks with json tag
            if "```json" in response_text:
                pattern = r'```json\s*([\s\S]+?)\s*```'
                matches = re.findall(pattern, response_text)
                if matches and len(matches) > 0:
                    json_content = matches[0].strip()
                    logger.info("Successfully extracted JSON from code block with json tag")
                else:
                    logger.error("Failed to find JSON content between ```json markers")
            
            # If that didn't work, try generic code blocks
            if json_content is None and "```" in response_text:
                pattern = r'```\s*([\s\S]+?)\s*```'
                matches = re.findall(pattern, response_text)
                if matches and len(matches) > 0:
                    # Get the first code block
                    json_content = matches[0].strip()
                    logger.info("Successfully extracted JSON from generic code block")
                else:
                    logger.error("Failed to find JSON content between ``` markers")
            
            # If still no content, look for JSON object patterns
            if json_content is None:
                pattern = r'(\{\s*"pixel_grid"\s*:[\s\S]+\})'
                matches = re.findall(pattern, response_text)
                if matches and len(matches) > 0:
                    json_content = matches[0].strip()
                    logger.info("Successfully extracted JSON using object pattern search")
                else:
                    # Last resort: just use the whole response
                    logger.warning("Could not extract JSON using any pattern, using entire response")
                    json_content = response_text.strip()
            
            # Clean up the extracted JSON
            # Remove any extra backticks that might have been included
            json_content = json_content.replace("```json", "").replace("```", "").strip()
            
            # Replace any trailing commas before closing brackets (invalid JSON)
            json_content = re.sub(r',\s*([}\]])', r'\1', json_content)
            
            # Try parsing the JSON
            try:
                pixel_data = json.loads(json_content)
                
                # Validate the response
                if "pixel_grid" not in pixel_data:
                    raise ValueError("Missing pixel_grid in response")
                if "palette" not in pixel_data:
                    # Extract palette from pixel grid
                    unique_colors = set()
                    for row in pixel_data["pixel_grid"]:
                        for pixel in row:
                            if pixel is not None:
                                unique_colors.add(pixel)
                    pixel_data["palette"] = list(unique_colors)
                if "explanation" not in pixel_data:
                    pixel_data["explanation"] = "Pixel art created using Gemini thinking model"
                
                return pixel_data
                
            except Exception as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"JSON content: {json_content[:500]}...")
                
                # Fallback: create a minimal valid response
                fallback_data = {
                    "pixel_grid": [[None for _ in range(width)] for _ in range(height)],
                    "palette": ["#FF0000", "#00FF00", "#0000FF"],
                    "explanation": "Fallback pixel art (JSON parsing failed)"
                }
                return fallback_data
                
        except Exception as e:
            logger.error(f"Failed to generate pixel art with text-based approach: {e}")
            
            # Provide additional error context if possible
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                logger.error(f"Response content: {e.response.text}")
                
            raise
            
    def generate_pixel_grid(self, 
                           prompt: str, 
                           width: int = 16, 
                           height: int = 16, 
                           max_colors: int = 16,
                           style: str = "2D pixel art") -> Dict[str, Any]:
        """
        Generate a pixel grid representation using Gemini.
        Uses function calling for standard models, and text-based approach for thinking models.
        
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
        
        # Prepare system message
        system_content = f"""You are a master pixel art designer specializing in {style}.
        You create beautiful, expressive pixel art within tight constraints, maximizing detail and visual appeal even in small canvases.
        
        CRITICAL GUIDELINES:
        - You MUST create a pixel grid EXACTLY {width} pixels wide by {height} pixels tall
        - You MUST use a fully transparent background (NOT colored backgrounds)
        - Always use null for transparent pixels
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
        6. For transparency, use null.
        7. Your explanation must be extremely short (under 150 characters).
        
        Your goal is to create rich, detailed pixel art that makes maximum use of the limited canvas.
        Don't create tiny sprites with lots of empty space around them."""
        
        return self.generate_pixel_art(
            system_prompt=system_content,
            user_prompt=user_content,
            width=width,
            height=height,
            max_colors=max_colors
        )