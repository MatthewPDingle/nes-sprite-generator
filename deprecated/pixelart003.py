#!/usr/bin/env python3
import os
import argparse
import logging
import base64
import json
import sys
import time
import random
import hashlib
from io import BytesIO
from datetime import datetime
from PIL import Image
import numpy as np
import requests
from openai import OpenAI

# Handle different versions of the OpenAI Python library
try:
    # For OpenAI Python library v1.0.0+
    from openai.types.error import APIError, RateLimitError
except ImportError:
    # For older versions of the OpenAI Python library
    try:
        from openai.error import APIError, RateLimitError
    except ImportError:
        # Define fallback error classes if neither import works
        class APIError(Exception):
            status_code = None
            pass
        
        class RateLimitError(Exception):
            pass

# Import gradio only when needed to avoid errors if not installed
try:
    import gradio as gr
except ImportError:
    gr = None

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class APICostTracker:
    """Track API usage costs"""
    
    def __init__(self):
        self.total_tokens = 0
        self.total_images = 0
        self.estimated_cost = 0
        self.requests = 0
        
    def track_completion(self, response):
        """Track completion API usage"""
        self.requests += 1
        if hasattr(response, 'usage') and response.usage:
            self.total_tokens += response.usage.total_tokens
            # Approximate cost calculation (update rates as needed)
            model = response.model
            
            # GPT-3.5 Turbo rates (approximate)
            if "gpt-3.5" in model:
                input_cost = 0.0000015 * response.usage.prompt_tokens
                output_cost = 0.000002 * response.usage.completion_tokens
                
            # GPT-4 rates (approximate)
            elif "gpt-4" in model:
                input_cost = 0.00003 * response.usage.prompt_tokens
                output_cost = 0.00006 * response.usage.completion_tokens
            else:
                input_cost = 0
                output_cost = 0
                
            self.estimated_cost += input_cost + output_cost
    
    def track_image(self, size, quality):
        """Track image generation cost"""
        self.requests += 1
        self.total_images += 1
        
        # DALL-E 3 approximate costs
        if size == "1024x1024":
            if quality == "hd":
                self.estimated_cost += 0.080  # HD cost
            else:
                self.estimated_cost += 0.040  # Standard cost
        elif size == "1792x1024" or size == "1024x1792":
            if quality == "hd":
                self.estimated_cost += 0.120  # HD cost for rectangular
            else:
                self.estimated_cost += 0.080  # Standard cost for rectangular
        elif size == "512x512":
            self.estimated_cost += 0.018  # DALL-E 2 rate
            
    def get_usage_report(self):
        """Get a usage report"""
        return {
            "requests": self.requests,
            "total_tokens": self.total_tokens,
            "total_images": self.total_images,
            "estimated_cost_usd": round(self.estimated_cost, 3)
        }


def api_call_with_backoff(api_function, max_retries=5, initial_delay=1):
    """
    Make an OpenAI API call with exponential backoff for rate limits and errors.
    
    Args:
        api_function: A function that makes the actual API call
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        
    Returns:
        The API response if successful
        
    Raises:
        Exception: If all retry attempts fail
    """
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return api_function()
        
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            
            # Add jitter to avoid synchronized retries
            jitter = random.uniform(0, 0.1) * delay
            sleep_time = delay + jitter
            
            logger.warning(f"Rate limit exceeded. Retrying in {sleep_time:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(sleep_time)
            
            # Exponential backoff with full jitter
            delay *= 2
        
        except APIError as e:
            # Only retry on 429 (rate limit) or 5xx (server errors)
            if hasattr(e, 'status_code') and (e.status_code == 429 or (e.status_code >= 500 and e.status_code < 600)):
                if attempt == max_retries - 1:
                    raise
                
                jitter = random.uniform(0, 0.1) * delay
                sleep_time = delay + jitter
                
                logger.warning(f"API error {e.status_code}. Retrying in {sleep_time:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(sleep_time)
                
                # Exponential backoff with full jitter
                delay *= 2
            else:
                # Don't retry on other API errors
                raise
        
        except Exception as e:
            # Don't retry on other exceptions
            raise


class PixelForgeClient:
    """Client for generating and refining pixel art using AI models."""
    
    def __init__(self, api_key=None, models_config=None, cache_dir=None):
        """Initialize the client with API key and model configuration."""
        self.api_key = api_key or self._get_api_key()
        
        # Handle different OpenAI client initialization methods
        try:
            # New version (v1.0.0+)
            self.client = OpenAI(api_key=self.api_key)
        except TypeError:
            # Older versions
            import openai
            openai.api_key = self.api_key
            self.client = openai
            
        self.cost_tracker = APICostTracker()
        
        # Default models configuration
        self.models_config = models_config or {
            "image_generation": "dall-e-3",
            "image_refinement": "dall-e-3",
            "evaluation": "gpt-4o",
            "editing": "dall-e-2"  # For image editing capabilities
        }
        
        # Setup cache directory
        self.cache_enabled = True
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "pixelforge_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Ensure output directory exists
        self.output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up logging for API calls
        self.api_log_path = os.path.join(self.output_dir, "api_calls.log")
    
    def _get_api_key(self):
        """Get API key from environment variable or file."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            try:
                with open("apikey.txt", "r") as file:
                    api_key = file.read().strip()
            except FileNotFoundError:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or create apikey.txt file.")
        return api_key
    
    def _log_api_call(self, endpoint, request_params, response, label):
        """Log API calls to file for debugging and tracking."""
        timestamp = datetime.now().isoformat()
        with open(self.api_log_path, "a", encoding="utf-8") as f:
            f.write(f"==== {timestamp} - {label} ====\n")
            f.write(f"Endpoint: {endpoint}\n")
            f.write(f"Request params: {json.dumps(request_params, default=str)}\n")
            f.write(f"Response: {json.dumps(response, default=str)}\n")
            f.write("====================================\n\n")
    
    def _get_cache_key(self, prompt, size, quality, model, style):
        """Generate a unique cache key based on request parameters."""
        data = f"{prompt}|{size}|{quality}|{model}|{style}"
        return hashlib.md5(data.encode()).hexdigest()

    def _check_cache(self, cache_key):
        """Check if result exists in cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                logger.info(f"Cache hit for key: {cache_key}")
                return json.load(f)
        return None

    def _save_to_cache(self, cache_key, result):
        """Save result to cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        with open(cache_file, "w") as f:
            json.dump(result, f)
        logger.info(f"Saved to cache: {cache_key}")
    
    def _create_pixel_art_prompt(self, base_prompt, style_strength=0.8):
        """Create an enhanced prompt specifically for pixel art generation."""
        # Define the core prompt enhancement for pixel art
        pixel_resolution = 16 + int(style_strength * 16)
        
        enhanced_prompt = f"""Create a pixel art sprite of {base_prompt}. 
        
The image must be in authentic {pixel_resolution}x{pixel_resolution} pixel art style with the following characteristics:
- Each pixel must be clearly visible and distinct
- Use a limited color palette of no more than 16 colors
- No anti-aliasing or blending between pixels
- Sharp, crisp pixel edges
- Flat color areas with no gradients
- Bold black outlines around the sprite
- The subject must fill the entire canvas and be perfectly centered
- The background should be transparent (represented as a checkerboard pattern)
- Classic retro game aesthetic similar to NES, SNES, or Game Boy Advance era

DO NOT generate a realistic or smooth image. This MUST look like authentic low-resolution pixel art with visible individual pixels, not a modern digital painting."""
        
        return enhanced_prompt
    
    def generate_pixel_art(self, prompt, size="1024x1024", quality="standard", style="vivid", use_cache=True, response_format="url"):
        """Generate pixel art image directly using DALL-E."""
        # Enhance the prompt with pixel art specific instructions
        enhanced_prompt = self._create_pixel_art_prompt(prompt)
        
        # Check cache first if enabled
        if use_cache and self.cache_enabled:
            cache_key = self._get_cache_key(enhanced_prompt, size, quality, self.models_config["image_generation"], style)
            cached_result = self._check_cache(cache_key)
            if cached_result:
                return cached_result
        
        logger.info(f"Generating pixel art with prompt: {enhanced_prompt[:100]}...")
        
        def generate_call():
            return self.client.images.generate(
                model=self.models_config["image_generation"],
                prompt=enhanced_prompt,
                size=size,
                quality=quality,
                style=style,
                response_format=response_format,
                n=1
            )
        
        try:
            # Generate the image using DALL-E with retry logic
            response = api_call_with_backoff(generate_call)
            
            # Track cost
            self.cost_tracker.track_image(size, quality)
            
            # Log the API call
            self._log_api_call(
                "images.generate", 
                {"model": self.models_config["image_generation"], "prompt": enhanced_prompt, "size": size}, 
                str(response), 
                f"generate_pixel_art"
            )
            
            # Process response based on format
            result = {}
            if response_format == "b64_json":
                result = {
                    "success": True,
                    "b64_json": response.data[0].b64_json,
                }
                
                # Save image from base64
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.output_dir, f"pixel_art_{timestamp}.png")
                
                img_data = base64.b64decode(response.data[0].b64_json)
                with open(filename, "wb") as f:
                    f.write(img_data)
                
                result["filename"] = filename
            else:
                image_url = response.data[0].url
                result = {
                    "success": True,
                    "url": image_url,
                }
                
                # Download and save the image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.output_dir, f"pixel_art_{timestamp}.png")
                
                # Download the image
                img_data = requests.get(image_url).content
                with open(filename, "wb") as f:
                    f.write(img_data)
                
                result["filename"] = filename
            
            logger.info(f"Saved generated image to {filename}")
            
            # Cache the result if enabled
            if self.cache_enabled and use_cache:
                self._save_to_cache(cache_key, result)
            
            return result
                
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_variations(self, image_path, n=3, size="512x512"):
        """Create variations of an existing pixel art image."""
        try:
            # Read the image file
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()
            
            def variation_call():
                return self.client.images.create_variation(
                    image=image_bytes,
                    n=n,
                    size=size
                )
            
            # Generate variations using DALL-E
            response = api_call_with_backoff(variation_call)
            
            # Track costs (assuming DALL-E 2 pricing for variations)
            for _ in range(n):
                self.cost_tracker.track_image(size, "standard")
            
            # Log the API call
            self._log_api_call(
                "images.create_variation", 
                {"n": n, "size": size}, 
                str(response), 
                f"create_variations"
            )
            
            # Process and save variations
            variation_files = []
            for i, img_data in enumerate(response.data):
                image_url = img_data.url
                
                # Download and save the variation
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.output_dir, f"variation_{timestamp}_{i+1}.png")
                
                # Download the image
                img_data = requests.get(image_url).content
                with open(filename, "wb") as f:
                    f.write(img_data)
                
                variation_files.append(filename)
            
            logger.info(f"Generated {n} variations and saved to {self.output_dir}")
            
            return {
                "success": True,
                "filenames": variation_files,
                "urls": [img_data.url for img_data in response.data]
            }
        
        except Exception as e:
            logger.error(f"Error creating variations: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def edit_image(self, image_path, mask_path, prompt, size="512x512"):
        """Edit parts of an image based on a mask using DALL-E 2."""
        try:
            # Read image and mask files
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()
            
            with open(mask_path, "rb") as mask_file:
                mask_bytes = mask_file.read()
            
            # Create enhanced prompt for the edit
            edit_prompt = f"Create pixel art style {prompt} with clear distinct pixels, limited color palette, and sharp edges"
            
            def edit_call():
                return self.client.images.edit(
                    model=self.models_config["editing"],
                    image=image_bytes,
                    mask=mask_bytes,
                    prompt=edit_prompt,
                    size=size,
                    n=1
                )
            
            # Make API call with retry logic
            response = api_call_with_backoff(edit_call)
            
            # Track cost (using DALL-E 2 edit pricing)
            self.cost_tracker.track_image(size, "standard")
            
            # Log the API call
            self._log_api_call(
                "images.edit", 
                {"model": self.models_config["editing"], "prompt": edit_prompt, "size": size}, 
                str(response), 
                "edit_image"
            )
            
            # Process and save the edited image
            image_url = response.data[0].url
            
            # Download and save the edited image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"edited_{timestamp}.png")
            
            # Download the image
            img_data = requests.get(image_url).content
            with open(filename, "wb") as f:
                f.write(img_data)
            
            logger.info(f"Saved edited image to {filename}")
            
            return {
                "success": True,
                "filename": filename,
                "url": image_url
            }
        
        except Exception as e:
            logger.error(f"Error editing image: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_improvement_points(self, feedback):
        """Extract key improvement points from feedback text."""
        # Split feedback into lines and look for suggestions
        lines = feedback.split('\n')
        improvement_points = []
        
        for i, line in enumerate(lines):
            # Look for lines that contain improvement suggestions
            if any(keyword in line.lower() for keyword in ['improve', 'suggestion', 'should', 'could', 'better', 'enhance']):
                improvement_points.append(line.strip())
            # Also look for low scores
            elif any(f"{i}: " in line for i in range(1, 6)):  # Scores 1-5 out of 10
                if i < len(lines) - 1:  # Get the next line which might explain the low score
                    improvement_points.append(lines[i+1].strip())
        
        # If no specific points found, use the whole feedback
        if not improvement_points:
            improvement_points = [feedback]
            
        return improvement_points
        
    def _create_refinement_prompt(self, original_prompt, improvement_points):
        """Create an enhanced prompt for refinement based on feedback."""
        improvements = "\n".join([f"- {point}" for point in improvement_points])
        
        refinement_prompt = f"""Create a pixel art sprite of {original_prompt}.

IMPORTANT: This is a refinement of a previous attempt that needed these improvements:
{improvements}

The image must be in authentic pixel art style with the following characteristics:
- Each pixel must be clearly visible and distinct
- Limited color palette with no more than 16 colors 
- No anti-aliasing or blending between pixels
- Sharp, crisp pixel edges 
- Bold black outlines around the sprite
- The subject must fill the entire canvas and be perfectly centered
- Classic retro game aesthetic similar to NES or SNES era

DO NOT generate a realistic or smooth image. This MUST look like authentic low-resolution pixel art with visible individual pixels, not a modern digital painting."""
        
        return refinement_prompt
    
    def evaluate_image(self, image_path, original_prompt):
        """Evaluate the generated pixel art using the vision model."""
        try:
            # Read the image file
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()
            
            # Convert to base64
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            data_uri = f"data:image/png;base64,{base64_image}"
            
            # Create the evaluation prompt
            evaluation_prompt = (
                "As a pixel art expert, evaluate this image based on the following criteria:\n\n"
                "1. Pixel Art Authenticity: Does it have clear pixel boundaries and limited palette?\n"
                "2. Composition: Is the subject centered and using the full canvas?\n"
                "3. Prompt Adherence: Does it match this description?\n"
                f"\"{original_prompt}\"\n"
                "4. Detail Quality: Is there sufficient pixel-level detail?\n"
                "5. Visual Appeal: Is it aesthetically pleasing as pixel art?\n\n"
                "Rate each criterion from 1-10. Then provide specific improvement suggestions."
            )
            
            # Prepare the function for API call with retry
            def evaluation_call():
                return self.client.chat.completions.create(
                    model=self.models_config["evaluation"],
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": evaluation_prompt},
                                {"type": "image_url", "image_url": {"url": data_uri}}
                            ]
                        }
                    ]
                )
            
            # Make the API call with retry
            response = api_call_with_backoff(evaluation_call)
            
            # Track token usage
            self.cost_tracker.track_completion(response)
            
            # Log the API call
            self._log_api_call(
                "chat.completions", 
                {"model": self.models_config["evaluation"]}, 
                str(response), 
                "evaluate_image"
            )
            
            # Extract the evaluation feedback
            feedback = response.choices[0].message.content
            
            # Save feedback to a file
            feedback_path = f"{os.path.splitext(image_path)[0]}_feedback.txt"
            with open(feedback_path, "w", encoding="utf-8") as f:
                f.write(feedback)
            
            logger.info(f"Saved evaluation feedback to {feedback_path}")
            
            return {
                "success": True,
                "feedback": feedback,
                "feedback_path": feedback_path
            }
            
        except Exception as e:
            logger.error(f"Error evaluating image: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def analyze_pixel_art_structured(self, image_path, original_prompt=None):
        """Analyze pixel art with structured output using function calling."""
        try:
            # Read the image file
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()
            
            # Convert to base64
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            data_uri = f"data:image/png;base64,{base64_image}"
            
            # Define the evaluation function for structured output
            functions = [
                {
                    "name": "evaluate_pixel_art",
                    "description": "Evaluate the quality of pixel art",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pixel_clarity": {
                                "type": "integer",
                                "description": "Rating from 1-10 of how clear and distinct the pixels are"
                            },
                            "color_palette": {
                                "type": "integer",
                                "description": "Rating from 1-10 of the color palette choice and limitation"
                            },
                            "composition": {
                                "type": "integer",
                                "description": "Rating from 1-10 of the composition and use of space"
                            },
                            "prompt_adherence": {
                                "type": "integer",
                                "description": "Rating from 1-10 of how well the image matches the prompt"
                            },
                            "overall_score": {
                                "type": "integer",
                                "description": "Overall rating from 1-10 of the pixel art quality"
                            },
                            "pixel_count_estimate": {
                                "type": "object",
                                "properties": {
                                    "width": {"type": "integer"},
                                    "height": {"type": "integer"}
                                },
                                "description": "Estimated pixel resolution of the sprite (width x height)"
                            },
                            "color_count_estimate": {
                                "type": "integer",
                                "description": "Estimated number of unique colors used in the palette"
                            },
                            "improvements": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "List of suggested improvements"
                            }
                        },
                        "required": ["pixel_clarity", "color_palette", "composition", "overall_score", "improvements"]
                    }
                }
            ]
            
            # Create the user message based on whether we have a prompt or not
            user_content = (
                "Analyze this pixel art image and provide a structured evaluation of its quality, "
                "pixel clarity, color palette, composition, and other technical aspects."
            )
            
            if original_prompt:
                user_content += f" The image was created based on this prompt: \"{original_prompt}\""
            
            # Prepare the function for API call with retry
            def analyze_call():
                return self.client.chat.completions.create(
                    model=self.models_config["evaluation"],
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_content},
                                {"type": "image_url", "image_url": {"url": data_uri}}
                            ]
                        }
                    ],
                    functions=functions,
                    function_call={"name": "evaluate_pixel_art"}
                )
            
            # Make the API call with retry
            response = api_call_with_backoff(analyze_call)
            
            # Track token usage
            self.cost_tracker.track_completion(response)
            
            # Log the API call
            self._log_api_call(
                "chat.completions", 
                {"model": self.models_config["evaluation"], "functions": functions}, 
                str(response), 
                "analyze_pixel_art_structured"
            )
            
            # Extract the structured evaluation
            function_call = response.choices[0].message.function_call
            if function_call and function_call.name == "evaluate_pixel_art":
                try:
                    evaluation = json.loads(function_call.arguments)
                    
                    # Save evaluation to a file
                    eval_path = f"{os.path.splitext(image_path)[0]}_analysis.json"
                    with open(eval_path, "w", encoding="utf-8") as f:
                        json.dump(evaluation, f, indent=2)
                    
                    logger.info(f"Saved structured analysis to {eval_path}")
                    
                    return {
                        "success": True,
                        "evaluation": evaluation,
                        "evaluation_path": eval_path
                    }
                except json.JSONDecodeError:
                    logger.error("Failed to parse function call arguments as JSON")
                    return {
                        "success": False,
                        "error": "Failed to parse structured analysis"
                    }
            else:
                logger.error("Model did not return a function call")
                return {
                    "success": False,
                    "error": "Model did not return structured analysis"
                }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def refine_image(self, image_path, original_prompt, feedback, iteration):
        """Refine the pixel art based on evaluation feedback."""
        try:
            # Extract key improvement points from feedback
            improvement_points = self._extract_improvement_points(feedback)
            
            # Create refinement prompt based on feedback
            refinement_prompt = self._create_refinement_prompt(original_prompt, improvement_points)
            
            # Log the refinement prompt
            logger.info(f"Refining image with prompt: {refinement_prompt[:100]}...")
            
            def refinement_call():
                return self.client.images.generate(
                    model=self.models_config["image_generation"],
                    prompt=refinement_prompt,
                    size="1024x1024",
                    quality="standard",
                    style="vivid",  # Using vivid style for more distinct colors
                    n=1
                )
            
            # Generate the refined image with retry logic
            response = api_call_with_backoff(refinement_call)
            
            # Track cost
            self.cost_tracker.track_image("1024x1024", "standard")
            
            # Log the API call
            self._log_api_call(
                "images.generate", 
                {"model": self.models_config["image_generation"], "prompt": refinement_prompt}, 
                str(response), 
                f"refine_image_iteration_{iteration}"
            )
            
            # Get the image data
            image_url = response.data[0].url
            
            # Save the image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"refined_{timestamp}_iter_{iteration}.png")
            
            # Download the image
            img_data = requests.get(image_url).content
            with open(filename, "wb") as f:
                f.write(img_data)
            
            logger.info(f"Saved refined image to {filename}")
            return {
                "success": True,
                "filename": filename,
                "url": image_url
            }
                
        except Exception as e:
            logger.error(f"Error refining image: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def post_process_image(self, image_path, pixelate=True, palette_size=16):
        """Apply post-processing to ensure pixel art aesthetic if needed."""
        try:
            # Open the image
            img = Image.open(image_path)
            
            # Create output path
            processed_path = f"{os.path.splitext(image_path)[0]}_processed.png"
            
            # Always process the image to ensure pixel art aesthetic
            # Get original dimensions
            width, height = img.size
            
            # Step 1: Pixelate the image
            # Calculate target dimensions for pixelation (much smaller)
            target_pixels = 32  # Target number of pixels for the largest dimension
            ratio = min(target_pixels / width, target_pixels / height)
            small_width = max(16, int(width * ratio))
            small_height = max(16, int(height * ratio))
            
            # Resize down to create pixelation effect
            small_img = img.resize((small_width, small_height), Image.NEAREST)
            
            # Step 2: Quantize colors to reduce the palette
            # Convert to RGB mode if it's not already
            if small_img.mode != "RGB":
                small_img = small_img.convert("RGB")
            
            # Use numpy for color quantization
            img_array = np.array(small_img)
            h, w, c = img_array.shape
            
            # Reshape the array for color quantization
            pixels = img_array.reshape(-1, c)
            
            # Simple color quantization: round each color channel to fewer levels
            levels = max(2, int(256 / palette_size))
            pixels = ((pixels // levels) * levels).astype(np.uint8)
            
            # Reshape back to image dimensions
            img_array = pixels.reshape(h, w, c)
            
            # Convert back to PIL Image
            quantized_img = Image.fromarray(img_array)
            
            # Step 3: Resize back up with nearest neighbor to maintain pixel edges
            pixelated = quantized_img.resize((width, height), Image.NEAREST)
            
            # Step 4: Optionally add pixel grid for more retro feel
            if small_width <= 64 and small_height <= 64:  # Only add grid for smaller sprites
                pixelated_array = np.array(pixelated)
                
                # Get the pixel size in the upscaled image
                pixel_width = width // small_width
                pixel_height = height // small_height
                
                # Add grid lines
                if pixel_width > 4 and pixel_height > 4:  # Only add grid if pixels are large enough
                    for i in range(small_width-1):
                        x = (i+1) * pixel_width - 1
                        pixelated_array[:, x, :] = (0, 0, 0, 255)  # Black grid lines
                    
                    for j in range(small_height-1):
                        y = (j+1) * pixel_height - 1
                        pixelated_array[y, :, :] = (0, 0, 0, 255)  # Black grid lines
                    
                    pixelated = Image.fromarray(pixelated_array)
            
            # Save the processed image
            pixelated.save(processed_path, "PNG")
            
            logger.info(f"Saved post-processed image to {processed_path}")
            return {
                "success": True,
                "filename": processed_path
            }
            
        except Exception as e:
            logger.error(f"Error post-processing image: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "filename": image_path  # Return original if processing fails
            }
    
    def batch_generate(self, prompts, size="256x256", quality="standard"):
        """Generate multiple pixel art images from a list of prompts."""
        results = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating image {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            # Generate the image
            result = self.generate_pixel_art(prompt, size=size, quality=quality)
            results.append(result)
            
            # Add a small delay to avoid rate limits
            if i < len(prompts) - 1:
                time.sleep(0.5)
        
        return results
    
    def generate_sprite_sheet(self, base_prompt, variations, rows=None, cols=None, sprite_size="128x128"):
        """Generate a sprite sheet with variations of a character or object."""
        # Generate all the individual sprites
        logger.info(f"Generating {len(variations)} sprites for sprite sheet...")
        
        sprites = []
        for i, variation in enumerate(variations):
            # Create a prompt for this specific variation
            variation_prompt = f"{base_prompt}, {variation}"
            
            # Generate the sprite
            result = self.generate_pixel_art(variation_prompt, size=sprite_size, quality="standard")
            
            if result["success"]:
                sprites.append(result["filename"])
            else:
                logger.error(f"Failed to generate sprite {i+1}: {result.get('error', 'Unknown error')}")
            
            # Add a delay to avoid rate limits
            if i < len(variations) - 1:
                time.sleep(0.5)
        
        # If we didn't get any sprites, return an error
        if not sprites:
            return {
                "success": False,
                "error": "Failed to generate any sprites"
            }
        
        # Determine sprite sheet layout
        if rows is None and cols is None:
            # Auto-determine layout based on the number of sprites
            count = len(sprites)
            cols = int(np.ceil(np.sqrt(count)))
            rows = int(np.ceil(count / cols))
        elif rows is None:
            # Auto-determine rows based on columns
            rows = int(np.ceil(len(sprites) / cols))
        elif cols is None:
            # Auto-determine columns based on rows
            cols = int(np.ceil(len(sprites) / rows))
        
        # Create the sprite sheet
        logger.info(f"Creating sprite sheet with {rows} rows and {cols} columns...")
        
        try:
            # Get the size of a single sprite
            first_sprite = Image.open(sprites[0])
            sprite_width, sprite_height = first_sprite.size
            
            # Create a new image for the sprite sheet
            sheet_width = cols * sprite_width
            sheet_height = rows * sprite_height
            
            # Create the sprite sheet with transparent background
            sprite_sheet = Image.new("RGBA", (sheet_width, sheet_height), (0, 0, 0, 0))
            
            # Place each sprite in the sheet
            for i, sprite_path in enumerate(sprites):
                if i >= rows * cols:
                    break  # Skip if we have more sprites than cells
                
                sprite = Image.open(sprite_path)
                
                # Calculate position
                row = i // cols
                col = i % cols
                
                x = col * sprite_width
                y = row * sprite_height
                
                # Paste the sprite onto the sheet
                sprite_sheet.paste(sprite, (x, y), sprite)
            
            # Save the sprite sheet
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sheet_path = os.path.join(self.output_dir, f"sprite_sheet_{timestamp}.png")
            sprite_sheet.save(sheet_path, "PNG")
            
            logger.info(f"Saved sprite sheet to {sheet_path}")
            
            return {
                "success": True,
                "filename": sheet_path,
                "sprites": sprites,
                "layout": {
                    "rows": rows,
                    "columns": cols,
                    "sprite_width": sprite_width,
                    "sprite_height": sprite_height
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating sprite sheet: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "sprites": sprites  # Return the individual sprites even if sheet creation failed
            }


# Command-line interface
def cli():
    """Command-line interface for the pixel art generator."""
    parser = argparse.ArgumentParser(description="PixelForge: AI-Powered Pixel Art Generator")
    
    # Main subparser for different operations
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate a pixel art image")
    generate_parser.add_argument("prompt", help="Description of the pixel art to generate")
    generate_parser.add_argument("-s", "--size", type=str, default="1024x1024", 
                                 help="Size of the output image (e.g., 256x256, 512x512, 1024x1024)")
    generate_parser.add_argument("-q", "--quality", type=str, default="standard", choices=["standard", "hd"],
                                 help="Quality of the image (standard or hd)")
    generate_parser.add_argument("--style", type=str, default="vivid", choices=["vivid", "natural"],
                                 help="Style of the image (vivid or natural)")
    generate_parser.add_argument("--no-cache", action="store_true", help="Disable cache for this request")
    generate_parser.add_argument("-p", "--post-process", action="store_true", 
                                 help="Apply post-processing to ensure pixel art aesthetic")
    
    # Refine command
    refine_parser = subparsers.add_parser("refine", help="Refine an existing pixel art image")
    refine_parser.add_argument("image", help="Path to the image to refine")
    refine_parser.add_argument("prompt", help="Original prompt used to generate the image")
    refine_parser.add_argument("-f", "--feedback", type=str, 
                               help="Manual feedback for improvement (if not provided, will generate automatically)")
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a pixel art image")
    evaluate_parser.add_argument("image", help="Path to the image to evaluate")
    evaluate_parser.add_argument("-p", "--prompt", type=str, help="Original prompt used to generate the image")
    evaluate_parser.add_argument("--structured", action="store_true", 
                                 help="Use structured evaluation with function calling")
    
    # Variations command
    variations_parser = subparsers.add_parser("variations", help="Create variations of an existing image")
    variations_parser.add_argument("image", help="Path to the source image")
    variations_parser.add_argument("-n", "--count", type=int, default=3, help="Number of variations to generate")
    variations_parser.add_argument("-s", "--size", type=str, default="512x512", 
                                   help="Size of the variations (e.g., 256x256, 512x512)")
    
    # Edit command
    edit_parser = subparsers.add_parser("edit", help="Edit an image with a mask")
    edit_parser.add_argument("image", help="Path to the source image")
    edit_parser.add_argument("mask", help="Path to the mask image (white areas will be edited)")
    edit_parser.add_argument("prompt", help="Description of what to generate in the masked area")
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Generate multiple images from a list of prompts")
    batch_parser.add_argument("file", help="Path to a text file with one prompt per line")
    batch_parser.add_argument("-s", "--size", type=str, default="256x256", 
                              help="Size of the output images (e.g., 256x256, 512x512)")
    batch_parser.add_argument("-q", "--quality", type=str, default="standard", choices=["standard", "hd"],
                              help="Quality of the images (standard or hd)")
    
    # Sprite sheet command
    spritesheet_parser = subparsers.add_parser("spritesheet", help="Generate a sprite sheet")
    spritesheet_parser.add_argument("prompt", help="Base prompt for the sprite sheet")
    spritesheet_parser.add_argument("variations", help="Path to a text file with one variation per line")
    spritesheet_parser.add_argument("-r", "--rows", type=int, help="Number of rows in the sprite sheet")
    spritesheet_parser.add_argument("-c", "--cols", type=int, help="Number of columns in the sprite sheet")
    spritesheet_parser.add_argument("-s", "--size", type=str, default="128x128", 
                                   help="Size of individual sprites (e.g., 128x128)")
    
    # Post-process command
    postprocess_parser = subparsers.add_parser("postprocess", help="Apply pixel art post-processing to an image")
    postprocess_parser.add_argument("image", help="Path to the image to post-process")
    postprocess_parser.add_argument("-p", "--palette", type=int, default=16, 
                                    help="Number of colors in the target palette")
    
    # Interactive web UI command
    web_parser = subparsers.add_parser("web", help="Launch interactive web interface")
    
    # Common options
    parser.add_argument("-o", "--output", type=str, help="Output directory for generated images")
    parser.add_argument("--disable-cache", action="store_true", help="Disable caching globally")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (overrides apikey.txt and environment variable)")
    
    args = parser.parse_args()
    
    # If no command is specified, print help and exit
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Create PixelForge client
    pixel_forge = PixelForgeClient(api_key=args.api_key)
    
    # Set output directory if specified
    if args.output:
        pixel_forge.output_dir = args.output
        os.makedirs(pixel_forge.output_dir, exist_ok=True)
    
    # Disable cache if requested
    if args.disable_cache:
        pixel_forge.cache_enabled = False
    
    # Process the command
    if args.command == "generate":
        # Generate a single image
        result = pixel_forge.generate_pixel_art(
            args.prompt, 
            size=args.size, 
            quality=args.quality,
            style=args.style,
            use_cache=not args.no_cache
        )
        
        if result["success"]:
            logger.info(f"Generated image: {result['filename']}")
            
            # Post-process if requested
            if args.post_process:
                post_result = pixel_forge.post_process_image(result["filename"])
                
                if post_result["success"]:
                    logger.info(f"Post-processed image: {post_result['filename']}")
                else:
                    logger.error(f"Failed to post-process image: {post_result.get('error', 'Unknown error')}")
        else:
            logger.error(f"Failed to generate image: {result.get('error', 'Unknown error')}")
    
    elif args.command == "refine":
        # Refine an existing image
        if args.feedback:
            feedback = args.feedback
        else:
            # Generate feedback automatically
            logger.info("Evaluating image for refinement...")
            eval_result = pixel_forge.evaluate_image(args.image, args.prompt)
            
            if not eval_result["success"]:
                logger.error(f"Failed to evaluate image: {eval_result.get('error', 'Unknown error')}")
                sys.exit(1)
            
            feedback = eval_result["feedback"]
            logger.info(f"Generated feedback: {feedback[:100]}...")
        
        # Refine the image
        result = pixel_forge.refine_image(args.image, args.prompt, feedback, 1)
        
        if result["success"]:
            logger.info(f"Refined image: {result['filename']}")
        else:
            logger.error(f"Failed to refine image: {result.get('error', 'Unknown error')}")
    
    elif args.command == "evaluate":
        # Evaluate an image
        if args.structured:
            result = pixel_forge.analyze_pixel_art_structured(args.image, args.prompt)
            
            if result["success"]:
                logger.info(f"Structured analysis saved to: {result['evaluation_path']}")
                
                # Print a summary of the evaluation
                eval_data = result["evaluation"]
                print("\nPixel Art Analysis Summary:")
                print(f"Overall Score: {eval_data.get('overall_score', 'N/A')}/10")
                print(f"Pixel Clarity: {eval_data.get('pixel_clarity', 'N/A')}/10")
                print(f"Color Palette: {eval_data.get('color_palette', 'N/A')}/10")
                print(f"Composition: {eval_data.get('composition', 'N/A')}/10")
                
                if "pixel_count_estimate" in eval_data:
                    pixel_est = eval_data["pixel_count_estimate"]
                    print(f"Estimated Resolution: {pixel_est.get('width', 'N/A')}x{pixel_est.get('height', 'N/A')} pixels")
                
                if "color_count_estimate" in eval_data:
                    print(f"Estimated Colors: {eval_data.get('color_count_estimate', 'N/A')}")
                
                print("\nSuggested Improvements:")
                for i, imp in enumerate(eval_data.get("improvements", []), 1):
                    print(f"{i}. {imp}")
            else:
                logger.error(f"Failed to analyze image: {result.get('error', 'Unknown error')}")
        else:
            result = pixel_forge.evaluate_image(args.image, args.prompt)
            
            if result["success"]:
                logger.info(f"Evaluation saved to: {result['feedback_path']}")
                
                # Print the feedback
                print("\nEvaluation Feedback:")
                print(result["feedback"])
            else:
                logger.error(f"Failed to evaluate image: {result.get('error', 'Unknown error')}")
    
    elif args.command == "variations":
        # Create variations of an image
        result = pixel_forge.create_variations(args.image, n=args.count, size=args.size)
        
        if result["success"]:
            logger.info(f"Generated {len(result['filenames'])} variations:")
            for i, filename in enumerate(result["filenames"], 1):
                logger.info(f"  {i}. {filename}")
        else:
            logger.error(f"Failed to create variations: {result.get('error', 'Unknown error')}")
    
    elif args.command == "edit":
        # Edit an image with a mask
        result = pixel_forge.edit_image(args.image, args.mask, args.prompt)
        
        if result["success"]:
            logger.info(f"Edited image: {result['filename']}")
        else:
            logger.error(f"Failed to edit image: {result.get('error', 'Unknown error')}")
    
    elif args.command == "batch":
        # Generate multiple images from a list of prompts
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip()]
            
            if not prompts:
                logger.error(f"No prompts found in file: {args.file}")
                sys.exit(1)
            
            logger.info(f"Found {len(prompts)} prompts in file")
            
            # Generate the images
            results = pixel_forge.batch_generate(prompts, size=args.size, quality=args.quality)
            
            # Summarize the results
            success_count = sum(1 for r in results if r.get("success", False))
            logger.info(f"Generated {success_count}/{len(prompts)} images successfully")
            
            # List the generated files
            for i, result in enumerate(results, 1):
                if result.get("success", False):
                    logger.info(f"  {i}. {result['filename']}")
                else:
                    logger.error(f"  {i}. Failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
    
    elif args.command == "spritesheet":
        # Generate a sprite sheet
        try:
            with open(args.variations, "r", encoding="utf-8") as f:
                variations = [line.strip() for line in f if line.strip()]
            
            if not variations:
                logger.error(f"No variations found in file: {args.variations}")
                sys.exit(1)
            
            logger.info(f"Found {len(variations)} variations in file")
            
            # Generate the sprite sheet
            result = pixel_forge.generate_sprite_sheet(
                args.prompt, 
                variations, 
                rows=args.rows, 
                cols=args.cols, 
                sprite_size=args.size
            )
            
            if result["success"]:
                logger.info(f"Generated sprite sheet: {result['filename']}")
                logger.info(f"Layout: {result['layout']['rows']} rows × {result['layout']['columns']} columns")
                logger.info(f"Sprite dimensions: {result['layout']['sprite_width']}×{result['layout']['sprite_height']} pixels")
            else:
                logger.error(f"Failed to generate sprite sheet: {result.get('error', 'Unknown error')}")
                
                # If we have individual sprites, list them
                if "sprites" in result and result["sprites"]:
                    logger.info(f"Generated {len(result['sprites'])} individual sprites:")
                    for i, sprite in enumerate(result["sprites"], 1):
                        logger.info(f"  {i}. {sprite}")
        except Exception as e:
            logger.error(f"Error generating sprite sheet: {str(e)}")
    
    elif args.command == "postprocess":
        # Post-process an image
        result = pixel_forge.post_process_image(args.image, palette_size=args.palette)
        
        if result["success"]:
            logger.info(f"Post-processed image: {result['filename']}")
        else:
            logger.error(f"Failed to post-process image: {result.get('error', 'Unknown error')}")
    
    elif args.command == "web":
        # Launch web interface
        launch_web_interface(pixel_forge)
    
    # Print usage stats
    usage = pixel_forge.cost_tracker.get_usage_report()
    logger.info(f"API Usage: {usage['requests']} requests, {usage['total_tokens']} tokens, {usage['total_images']} images")
    logger.info(f"Estimated cost: ${usage['estimated_cost_usd']}")


# Web interface using Gradio
def launch_web_interface(pixel_forge):
    """Launch web interface for the pixel art generator using Gradio."""
    if gr is None:
        print("Gradio is not installed. Please install it with 'pip install gradio'")
        return
    
    # Define interface functions
    def generate(prompt, size, quality, style_strength, iterations, post_process, use_cache):
        """Generate pixel art from a prompt."""
        images = []
        feedbacks = []
        urls = []
        
        # Map style strength to style (0.0-0.5 -> natural, 0.5-1.0 -> vivid)
        style = "natural" if style_strength < 0.5 else "vivid"
        
        # Generate initial image
        result = pixel_forge.generate_pixel_art(
            prompt, 
            size=size, 
            quality=quality, 
            style=style,
            use_cache=use_cache
        )
        
        if not result["success"]:
            return None, f"Error: {result.get('error', 'Unknown error')}", None, None
        
        current_image_path = result["filename"]
        current_url = result.get("url", "")
        
        images.append(Image.open(current_image_path))
        urls.append(current_url)
        
        # Iterative refinement
        for i in range(int(iterations) - 1):
            # Evaluate current image
            eval_result = pixel_forge.evaluate_image(current_image_path, prompt)
            
            if not eval_result["success"]:
                feedbacks.append(f"Evaluation error: {eval_result.get('error', 'Unknown error')}")
                break
            
            feedback = eval_result["feedback"]
            feedbacks.append(feedback)
            
            # Refine the image
            refine_result = pixel_forge.refine_image(current_image_path, prompt, feedback, i + 1)
            
            if not refine_result["success"]:
                break
            
            current_image_path = refine_result["filename"]
            current_url = refine_result.get("url", "")
            
            images.append(Image.open(current_image_path))
            urls.append(current_url)
        
        # Post-process final image if requested
        if post_process and images:
            post_result = pixel_forge.post_process_image(current_image_path)
            
            if post_result["success"]:
                post_processed = Image.open(post_result["filename"])
                images.append(post_processed)
                urls.append("")  # No URL for post-processed image
        
        # Return the final image, feedback, and all iterations
        final_image = images[-1] if images else None
        all_iterations = images if images else None
        feedback_text = "\n\n".join(feedbacks) if feedbacks else "No feedback available."
        
        # Get usage stats
        usage = pixel_forge.cost_tracker.get_usage_report()
        usage_text = f"API Usage: {usage['requests']} requests, {usage['total_tokens']} tokens, {usage['total_images']} images\n"
        usage_text += f"Estimated cost: ${usage['estimated_cost_usd']}"
        
        return final_image, feedback_text, all_iterations, usage_text
    
    def create_variations(image, num_variations, variation_size):
        """Create variations of an uploaded image."""
        if image is None:
            return None, "Please upload an image first.", None
        
        # Save the uploaded image to a temporary file
        temp_path = os.path.join(pixel_forge.output_dir, "temp_upload.png")
        image.save(temp_path)
        
        # Generate variations
        result = pixel_forge.create_variations(temp_path, n=num_variations, size=variation_size)
        
        if not result["success"]:
            return None, f"Error: {result.get('error', 'Unknown error')}", None
        
        # Load the variations as PIL images
        variations = [Image.open(path) for path in result["filenames"]]
        
        # Get usage stats
        usage = pixel_forge.cost_tracker.get_usage_report()
        usage_text = f"API Usage: {usage['requests']} requests, {usage['total_tokens']} tokens, {usage['total_images']} images\n"
        usage_text += f"Estimated cost: ${usage['estimated_cost_usd']}"
        
        return variations, f"Generated {len(variations)} variations successfully.", usage_text
    
    def analyze_image(image, original_prompt):
        """Analyze an uploaded image."""
        if image is None:
            return "Please upload an image first.", None
        
        # Save the uploaded image to a temporary file
        temp_path = os.path.join(pixel_forge.output_dir, "temp_analysis.png")
        image.save(temp_path)
        
        # Analyze the image
        result = pixel_forge.analyze_pixel_art_structured(temp_path, original_prompt)
        
        if not result["success"]:
            return f"Error: {result.get('error', 'Unknown error')}", None
        
        # Format the analysis as text
        analysis = result["evaluation"]
        
        text = "# Pixel Art Analysis\n\n"
        text += f"## Overall Score: {analysis.get('overall_score', 'N/A')}/10\n\n"
        text += "## Detailed Ratings:\n"
        text += f"- Pixel Clarity: {analysis.get('pixel_clarity', 'N/A')}/10\n"
        text += f"- Color Palette: {analysis.get('color_palette', 'N/A')}/10\n"
        text += f"- Composition: {analysis.get('composition', 'N/A')}/10\n"
        
        if "prompt_adherence" in analysis:
            text += f"- Prompt Adherence: {analysis.get('prompt_adherence', 'N/A')}/10\n"
        
        if "pixel_count_estimate" in analysis:
            pixel_est = analysis["pixel_count_estimate"]
            text += f"\n## Technical Details:\n"
            text += f"- Estimated Resolution: {pixel_est.get('width', 'N/A')}×{pixel_est.get('height', 'N/A')} pixels\n"
        
        if "color_count_estimate" in analysis:
            text += f"- Estimated Colors: {analysis.get('color_count_estimate', 'N/A')}\n"
        
        text += "\n## Suggested Improvements:\n"
        for i, imp in enumerate(analysis.get("improvements", []), 1):
            text += f"{i}. {imp}\n"
        
        # Get usage stats
        usage = pixel_forge.cost_tracker.get_usage_report()
        usage_text = f"API Usage: {usage['requests']} requests, {usage['total_tokens']} tokens, {usage['total_images']} images\n"
        usage_text += f"Estimated cost: ${usage['estimated_cost_usd']}"
        
        return text, usage_text
    
    def post_process_uploaded(image, palette_size):
        """Post-process an uploaded image."""
        if image is None:
            return None, "Please upload an image first.", None
        
        # Save the uploaded image to a temporary file
        temp_path = os.path.join(pixel_forge.output_dir, "temp_postprocess.png")
        image.save(temp_path)
        
        # Post-process the image
        result = pixel_forge.post_process_image(temp_path, palette_size=palette_size)
        
        if not result["success"]:
            return None, f"Error: {result.get('error', 'Unknown error')}", None
        
        # Load the post-processed image
        processed = Image.open(result["filename"])
        
        # Get usage stats (no API usage for post-processing)
        usage = pixel_forge.cost_tracker.get_usage_report()
        usage_text = f"API Usage: {usage['requests']} requests, {usage['total_tokens']} tokens, {usage['total_images']} images\n"
        usage_text += f"Estimated cost: ${usage['estimated_cost_usd']}"
        
        return processed, "Image successfully post-processed.", usage_text
    
    # Create the Gradio interface
    with gr.Blocks(title="PixelForge: AI Pixel Art Generator") as interface:
        gr.Markdown("# PixelForge: AI-Powered Pixel Art Generator")
        gr.Markdown("Create beautiful pixel art for your games with AI assistance")
        
        with gr.Tabs():
            # Generation Tab
            with gr.TabItem("Generate Pixel Art"):
                with gr.Row():
                    with gr.Column(scale=2):
                        prompt_input = gr.Textbox(
                            label="Describe your pixel art", 
                            lines=3, 
                            placeholder="A heroic knight with a sword and shield in a fantasy pixel art style"
                        )
                        
                        with gr.Row():
                            size_dropdown = gr.Dropdown(
                                label="Output Size", 
                                choices=["256x256", "512x512", "1024x1024", "1024x1792", "1792x1024"], 
                                value="1024x1024"
                            )
                            quality_dropdown = gr.Dropdown(
                                label="Quality", 
                                choices=["standard", "hd"], 
                                value="standard"
                            )
                        
                        with gr.Row():
                            style_slider = gr.Slider(
                                label="Style (Natural to Vivid)", 
                                minimum=0.0, 
                                maximum=1.0, 
                                value=0.8, 
                                step=0.1
                            )
                            iterations_slider = gr.Slider(
                                label="Refinement Iterations", 
                                minimum=1, 
                                maximum=5, 
                                value=2, 
                                step=1
                            )
                        
                        with gr.Row():
                            post_process_checkbox = gr.Checkbox(
                                label="Apply Post-Processing", 
                                value=True
                            )
                            use_cache_checkbox = gr.Checkbox(
                                label="Use Cache", 
                                value=True
                            )
                        
                        generate_button = gr.Button("Generate Pixel Art", variant="primary")
                    
                    with gr.Column(scale=3):
                        output_image = gr.Image(label="Generated Pixel Art", type="pil")
                        feedback_output = gr.Textbox(label="Evaluation Feedback", lines=8)
                
                with gr.Row():
                    gr.Markdown("## Iteration History")
                    iteration_gallery = gr.Gallery(label="All Iterations", show_label=False, object_fit="contain", columns=5)
                
                usage_output = gr.Textbox(label="API Usage", lines=2)
                
                generate_button.click(
                    generate,
                    inputs=[
                        prompt_input, size_dropdown, quality_dropdown, style_slider, 
                        iterations_slider, post_process_checkbox, use_cache_checkbox
                    ],
                    outputs=[output_image, feedback_output, iteration_gallery, usage_output]
                )
            
            # Variations Tab
            with gr.TabItem("Create Variations"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(label="Upload an image", type="pil")
                        
                        with gr.Row():
                            variation_count = gr.Slider(
                                label="Number of Variations", 
                                minimum=1, 
                                maximum=10, 
                                value=4, 
                                step=1
                            )
                            variation_size = gr.Dropdown(
                                label="Variation Size", 
                                choices=["256x256", "512x512", "1024x1024"], 
                                value="512x512"
                            )
                        
                        variations_button = gr.Button("Create Variations", variant="primary")
                        variations_status = gr.Textbox(label="Status")
                    
                    with gr.Column(scale=2):
                        variations_gallery = gr.Gallery(label="Generated Variations", show_label=True, object_fit="contain", columns=2)
                
                variations_usage = gr.Textbox(label="API Usage", lines=2)
                
                variations_button.click(
                    create_variations,
                    inputs=[input_image, variation_count, variation_size],
                    outputs=[variations_gallery, variations_status, variations_usage]
                )
            
            # Analysis Tab
            with gr.TabItem("Analyze Pixel Art"):
                with gr.Row():
                    with gr.Column(scale=1):
                        analysis_image = gr.Image(label="Upload pixel art to analyze", type="pil")
                        analysis_prompt = gr.Textbox(
                            label="Original prompt (optional)", 
                            lines=2, 
                            placeholder="If you know the original prompt, enter it here for better analysis"
                        )
                        analyze_button = gr.Button("Analyze Pixel Art", variant="primary")
                    
                    with gr.Column(scale=2):
                        analysis_output = gr.Markdown(label="Analysis")
                
                analysis_usage = gr.Textbox(label="API Usage", lines=2)
                
                analyze_button.click(
                    analyze_image,
                    inputs=[analysis_image, analysis_prompt],
                    outputs=[analysis_output, analysis_usage]
                )
            
            # Post-Processing Tab
            with gr.TabItem("Post-Processing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        postprocess_image = gr.Image(label="Upload image to post-process", type="pil")
                        
                        palette_size = gr.Slider(
                            label="Target Color Palette Size", 
                            minimum=4, 
                            maximum=32, 
                            value=16, 
                            step=4
                        )
                        
                        postprocess_button = gr.Button("Post-Process Image", variant="primary")
                        postprocess_status = gr.Textbox(label="Status")
                    
                    with gr.Column(scale=1):
                        postprocessed_image = gr.Image(label="Post-Processed Image", type="pil")
                
                postprocess_usage = gr.Textbox(label="API Usage", lines=2)
                
                postprocess_button.click(
                    post_process_uploaded,
                    inputs=[postprocess_image, palette_size],
                    outputs=[postprocessed_image, postprocess_status, postprocess_usage]
                )
        
        gr.Markdown("""
        ## Tips for Great Pixel Art
        - Be specific about characters, objects, and environments
        - Mention color palette preferences if you have any
        - Specify the style (e.g., 8-bit, 16-bit, fantasy RPG, sci-fi)
        - For game sprites, mention the perspective (top-down, side view, isometric)
        
        ## Commands
        You can also use PixelForge from the command line:
        ```
        python pixelforge.py generate "a cyberpunk robot character sprite" --size 256x256 --quality standard
        ```
        """)
    
    # Launch the interface
    interface.launch(share=True)


if __name__ == "__main__":
    cli()