#!/usr/bin/env python3
import json
import logging
from typing import Dict, Any, Optional

import openai

from .base import BaseClient
from ..config import get_api_key

logger = logging.getLogger(__name__)

class GeminiClient(BaseClient):
    """Client for generating pixel art using Google's Gemini models."""
    
    # Define Gemini model constants
    GEMINI_FLASH = "gemini-2.0-flash"
    GEMINI_FLASH_EXP = "gemini-2.0-flash-exp"
    GEMINI_PRO = "gemini-2.0-pro-exp-02-05"
    GEMINI_FLASH_THINKING = "gemini-2.0-flash-thinking-exp-01-21"
    
    # Models with image generation capabilities
    IMAGE_GENERATION_MODELS = [GEMINI_FLASH_EXP]
    
    def __init__(self, api_key: Optional[str] = None, model: str = GEMINI_FLASH):
        """
        Initialize the Gemini client with credentials.
        
        Args:
            api_key: Google API key.
            model: Model name or identifier.
        """
        self.api_key = api_key or get_api_key("google")
        self.model = model
        
        # Log model details for debugging
        logger.info(f"GeminiClient initialized with model string: '{model}'")
        logger.info(f"Image generation models: {self.IMAGE_GENERATION_MODELS}")
        
        # Check for image generation capability
        supports_image_gen = any(gen_model in model for gen_model in self.IMAGE_GENERATION_MODELS)
        logger.info(f"Model supports image generation: {supports_image_gen}")
        
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
    
    def generate_pixel_grid(self, 
                           prompt: str, 
                           width: int = 16, 
                           height: int = 16, 
                           max_colors: int = 16,
                           style: str = "2D pixel art") -> Dict[str, Any]:
        """
        Generate a pixel grid representation using Gemini.
        Uses function calling for standard models, and text-based approach for thinking models.
        For supported models, can directly generate images.
        
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
        
        # Check if this is a model with direct image generation capability
        # Use a more flexible comparison to handle any variations in the model string
        if any(model_name in self.model for model_name in self.IMAGE_GENERATION_MODELS):
            logger.info(f"Using direct image generation with model: {self.model}")
            return self._generate_with_image_generation(prompt, width, height, max_colors, style)
        
        # Check if we're using a thinking model
        is_thinking_model = "thinking" in self.model.lower()
        
        if is_thinking_model:
            # Text-based approach for thinking models (no function calling)
            return self._generate_with_text_based_approach(prompt, width, height, max_colors, style)
        else:
            # Function calling approach for standard models
            return self._generate_with_function_calling(prompt, width, height, max_colors, style)
            
    def _generate_with_image_generation(self, prompt: str, width: int, height: int, 
                                       max_colors: int, style: str) -> Dict[str, Any]:
        """
        Generate pixel art by directly requesting an image from Gemini.
        Only available with supported models.
        
        Args:
            prompt: Description of the pixel art to create
            width: Width of the pixel canvas
            height: Height of the pixel canvas
            max_colors: Maximum number of colors to use
            style: Style guide for the pixel art
            
        Returns:
            Dictionary containing the pixel grid, palette, and explanation
        """
        try:
            from ..image_utils import process_raw_image, image_to_pixel_grid
            import base64
        except ImportError as e:
            logger.error(f"Failed to import required modules for image generation: {e}")
            logger.warning("Falling back to function calling approach")
            return self._generate_with_function_calling(prompt, width, height, max_colors, style)
        
        # Prepare the prompt for image generation
        image_prompt = f"""Generate an NES-style pixel art of: {prompt}
        
        IMPORTANT REQUIREMENTS:
        1. The image MUST be pixel art in NES style.
        2. Center the subject on a pure white background.
        3. Make the subject LARGE and clear.
        4. Use rich colors and subtle gradients to create depth and texture.
        5. Add highlights and shadows to give dimension.
        6. Include small details that make the pixel art feel complete and refined.
        
        The subject should be the clear focus, centered on a pure white background.
        """
        
        logger.info(f"Generating image with {self.model} for prompt: {prompt}")
        
        try:
            # Make the API call with direct parameters for image generation
            logger.info("Making API call to Gemini for direct image generation")
            
            try:
                # Based on the error message, the config parameter is not supported in this client
                # Try using parameters directly at the top level
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": image_prompt}],
                    response_modalities=["IMAGE"],
                    temperature=0.7,  # Adjust as needed
                    max_tokens=1024
                )
                logger.info(f"Made API call to {self.model} with response_modalities parameter")
            except Exception as e:
                logger.warning(f"Error using response_modalities: {e}")
                
                # Try alternative approach with a special system message
                try:
                    logger.info("Trying alternative approach with system message")
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You can create images. Always respond with an image for visual prompts."},
                            {"role": "user", "content": f"Create an image: {image_prompt}"}
                        ],
                        temperature=0.7,
                        max_tokens=1024
                    )
                except Exception as e2:
                    logger.error(f"Alternative approach also failed: {e2}")
                    logger.warning("Falling back to function calling approach")
                    return self._generate_with_function_calling(prompt, width, height, max_colors, style)
                
            logger.info("Successfully received response from Gemini image generation API")
            
            # Check if we have an image in the response - examine all possible attributes
            logger.info(f"Response structure: {dir(response)}")
            
            # Get image data - try different attributes that might hold the image
            image_data = None
            
            # Try different possible attributes
            if hasattr(response, 'images') and response.images:
                logger.info("Found 'images' attribute in response")
                image_data = response.images[0].image_data if hasattr(response.images[0], 'image_data') else None
            
            # Try content field for multimodal response
            if not image_data and hasattr(response, 'choices') and response.choices:
                logger.info("Looking in 'choices' attribute for image content")
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = choice.message.content
                    logger.info(f"Content type: {type(content)}")
                    
                    # If content is a list, look for image parts
                    if isinstance(content, list):
                        for part in content:
                            if hasattr(part, 'type') and part.type == 'image':
                                if hasattr(part, 'image_data') or hasattr(part, 'data'):
                                    image_data = getattr(part, 'image_data', None) or getattr(part, 'data', None)
                                    logger.info("Found image in multimodal content")
                                    break
            
            # If we still don't have an image, fall back
            if not image_data:
                logger.warning("No image was returned by the Gemini model - falling back to function calling")
                return self._generate_with_function_calling(prompt, width, height, max_colors, style)
            
            logger.info("Successfully extracted image data from response")
            
            logger.info(f"Received image from {self.model}, processing...")
            
            try:
                # Process the image according to our NES sprite workflow
                processed_image = process_raw_image(
                    image_data=image_data,
                    target_width=width,
                    target_height=height,
                    max_colors=max_colors,
                    white_tolerance=10,  # Adjust tolerance for background removal if needed
                    save_steps=True,
                    output_prefix=f"gemini_{self.model.replace('-', '_')}_prompt_{prompt[:20].replace(' ', '_').replace('/', '_')}"
                )
                logger.info("Successfully processed the image through the NES workflow")
            except Exception as e:
                logger.error(f"Error processing the image: {e}")
                logger.warning("Falling back to function calling approach")
                return self._generate_with_function_calling(prompt, width, height, max_colors, style)
            
            # Convert the processed image to a pixel grid and palette
            pixel_grid, palette = image_to_pixel_grid(processed_image)
            
            # Return in the same format as the other generation methods
            return {
                "pixel_grid": pixel_grid,
                "palette": palette,
                "explanation": f"NES-style pixel art of {prompt} generated with {self.model}"
            }
            
        except Exception as e:
            logger.error(f"Error generating image with {self.model}: {e}")
            # If image generation fails, fall back to the standard approach
            logger.warning(f"Falling back to function calling approach")
            return self._generate_with_function_calling(prompt, width, height, max_colors, style)
            
    def _generate_with_function_calling(self, prompt, width, height, max_colors, style):
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
                            if isinstance(pixel, list) and len(pixel) == 4:
                                unique_colors.add(tuple(pixel))
                    function_args["palette"] = [list(color) for color in unique_colors]
                
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
            
    def _generate_with_text_based_approach(self, prompt, width, height, max_colors, style):
        """Generate pixel art using a text-based approach (for thinking models)."""
        import re
        
        # Prepare system message for text-based approach
        system_content = f"""You are a master pixel art designer specializing in {style}.
        Your task is to create a detailed {width}x{height} pixel art based on a user description.
        
        Return your response in EXACTLY this JSON format, without any markdown markers:
        
        {{
            "pixel_grid": [
                // {height} rows of pixels with {width} RGBA values per row
                [[r,g,b,a], [r,g,b,a], ...], // Row 1
                [[r,g,b,a], [r,g,b,a], ...], // Row 2
                // ... more rows ...
            ],
            "palette": [
                // List of unique RGBA colors
                [r,g,b,a], [r,g,b,a], ...
            ],
            "explanation": "Brief design explanation"
        }}
        
        CRITICAL REQUIREMENTS:
        - Return ONLY the valid JSON with no code block markers or other text
        - The grid MUST be EXACTLY {height} rows with EXACTLY {width} pixels per row
        - Use RGBA format with values 0-255 (e.g. [255,0,0,255] for red)
        - Use [0,0,0,0] for transparent pixels
        - Make the subject fill most of the canvas
        - The explanation must be brief (under 150 characters)
        """
        
        # Prepare user message
        user_content = f"""Create a {width}x{height} pixel art of: {prompt}
        
        Design requirements:
        1. Subject should be large and fill most of the canvas
        2. Use rich colors with highlights and shadows
        3. Include small details for a refined look
        4. Keep the background transparent ([0,0,0,0])
        
        IMPORTANT: Return ONLY pure JSON without any markdown or code block markers.
        Just return the JSON object directly, starting with {{ and ending with }}.
        """
        
        # Assemble messages
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
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
                    
            # Check for empty content
            if not json_content or json_content.isspace():
                logger.error("Extracted JSON content is empty!")
                json_content = "{}"  # Empty JSON object as fallback
            
            # Log the extracted JSON for debugging (truncated for readability)
            json_preview = json_content[:100] + "..." if len(json_content) > 100 else json_content
            logger.info(f"Extracted JSON content (preview): {json_preview}")
            
            # Clean up the extracted JSON
            # Remove any extra backticks that might have been included
            json_content = json_content.replace("```json", "").replace("```", "").strip()
            
            # Replace any trailing commas before closing brackets (invalid JSON)
            json_content = re.sub(r',\s*([}\]])', r'\1', json_content)
            
            # Parse the JSON content
            try:
                # Check if the content is empty
                if not json_content or json_content.isspace():
                    raise ValueError("Empty JSON content")
                
                # Log exact content being parsed (first 500 chars)
                log_content = json_content[:500] + "..." if len(json_content) > 500 else json_content
                logger.info(f"Attempting to parse JSON: {log_content}")
                
                pixel_data = json.loads(json_content)
                logger.info("Successfully parsed JSON response")
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                error_pos = e.pos
                error_line, error_col = 1, 1
                
                # Calculate line and column for better error reporting
                lines = json_content[:error_pos].split('\n')
                if lines:
                    error_line = len(lines)
                    error_col = len(lines[-1]) + 1
                
                # Extract context around the error
                start = max(0, error_pos - 40)
                end = min(len(json_content), error_pos + 40)
                context = json_content[start:end]
                if start > 0:
                    context = "..." + context
                if end < len(json_content):
                    context = context + "..."
                
                logger.error(f"Error at line {error_line}, column {error_col} (char {error_pos})")
                logger.error(f"Context: {context}")
                
                # Try some basic repairs
                logger.info("Attempting JSON repairs...")
                fixed_content = json_content
                
                # Check if this is a truncation issue (common with large responses)
                if error_pos > 0 and error_pos == len(json_content) - 1:
                    logger.warning("JSON appears to be truncated - attempting to complete it")
                    
                    # 1. First try to find the last complete object (array or dict)
                    last_array_close = json_content.rfind(']')
                    last_obj_close = json_content.rfind('}')
                    last_complete = max(last_array_close, last_obj_close)
                    
                    if last_complete > 0 and last_complete < len(json_content) - 10:
                        # If we have a substantial portion of the JSON and found a closing marker,
                        # truncate to that point and then complete the structure
                        logger.info(f"Truncating to last complete object at position {last_complete + 1}")
                        fixed_content = json_content[:last_complete + 1]
                        
                        # Now complete the structure by counting opening and closing braces/brackets
                        # Count from the beginning to our truncation point
                        open_braces = fixed_content.count('{')
                        close_braces = fixed_content.count('}')
                        open_brackets = fixed_content.count('[')
                        close_brackets = fixed_content.count(']')
                        
                        # Add enough closing markers to properly terminate the structure
                        if open_brackets > close_brackets:
                            logger.info(f"Adding {open_brackets - close_brackets} missing closing brackets")
                            fixed_content += ']' * (open_brackets - close_brackets)
                        
                        if open_braces > close_braces:
                            logger.info(f"Adding {open_braces - close_braces} missing closing braces")
                            fixed_content += '}' * (open_braces - close_braces)
                
                # 2. Try removing text before first JSON object
                first_brace = fixed_content.find('{')
                if first_brace > 0:
                    old_content = fixed_content
                    fixed_content = fixed_content[first_brace:]
                    logger.info(f"Removed {first_brace} characters before first opening brace")
                    
                    # Recount braces after this operation
                    open_braces = fixed_content.count('{')
                    close_braces = fixed_content.count('}')
                    open_brackets = fixed_content.count('[')
                    close_brackets = fixed_content.count(']')
                
                # 3. Fix mismatched braces and brackets
                if open_braces > close_braces:
                    logger.info(f"Adding {open_braces - close_braces} missing closing braces")
                    fixed_content += '}' * (open_braces - close_braces)
                    
                if open_brackets > close_brackets:
                    logger.info(f"Adding {open_brackets - close_brackets} missing closing brackets")
                    fixed_content += ']' * (open_brackets - close_brackets)
                
                # 4. Additional fixes for common issues
                # Replace any trailing commas in arrays [1,2,3,] -> [1,2,3]
                fixed_content = re.sub(r',(\s*[\]}])', r'\1', fixed_content)
                
                # Try parsing again
                try:
                    logger.info(f"Attempting to parse repaired JSON: {fixed_content[:100]}...")
                    pixel_data = json.loads(fixed_content)
                    logger.info("Successfully parsed repaired JSON")
                except json.JSONDecodeError as e2:
                    # If all else fails, try to build a minimal valid response
                    logger.error(f"JSON repair failed: {e2}")
                    logger.warning("Building fallback response")
                    
                    # Create a minimal valid structure
                    pixel_data = {
                        "pixel_grid": [[[0,0,0,0] for _ in range(width)] for _ in range(height)],
                        "palette": [[0,0,0,0], [100,100,100,255], [200,200,200,255]],
                        "explanation": "Generated with fallback due to JSON parsing error"
                    }
            
            # Validate and repair the response
            if not isinstance(pixel_data, dict):
                raise ValueError(f"Gemini response is not a JSON object: {type(pixel_data)}")
                
            # Check required fields
            if "pixel_grid" not in pixel_data:
                raise ValueError("Missing 'pixel_grid' in Gemini response")
                
            # Extract palette if missing
            if "palette" not in pixel_data:
                logger.warning("Missing 'palette' in response. Extracting from pixel_grid.")
                unique_colors = set()
                for row in pixel_data["pixel_grid"]:
                    for pixel in row:
                        if isinstance(pixel, list) and len(pixel) == 4:
                            unique_colors.add(tuple(pixel))
                pixel_data["palette"] = [list(color) for color in unique_colors]
                
            # Add explanation if missing
            if "explanation" not in pixel_data:
                pixel_data["explanation"] = "Pixel art created with Gemini thinking model."
            
            # Verify and repair grid dimensions
            grid = pixel_data["pixel_grid"]
            if len(grid) != height:
                logger.warning(f"Grid height mismatch: expected {height}, got {len(grid)}")
                
                # Pad or trim to match expected height
                if len(grid) < height:
                    # Pad with empty rows
                    empty_row = [[0,0,0,0] for _ in range(width)]
                    grid.extend([empty_row.copy() for _ in range(height - len(grid))])
                else:
                    # Trim rows
                    pixel_data["pixel_grid"] = grid[:height]
            
            # Check and fix width for each row
            for i, row in enumerate(grid):
                if len(row) != width:
                    logger.warning(f"Grid width mismatch at row {i}: expected {width}, got {len(row)}")
                    
                    # Pad or trim the row
                    if len(row) < width:
                        # Pad with transparent pixels
                        row.extend([[0,0,0,0] for _ in range(width - len(row))])
                    else:
                        # Trim the row
                        grid[i] = row[:width]
            
            return pixel_data
            
        except Exception as e:
            logger.error(f"Failed to generate pixel art with Gemini text-based approach: {e}")
            
            # Provide additional error context if possible
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                logger.error(f"Response content: {e.response.text}")
                
            raise