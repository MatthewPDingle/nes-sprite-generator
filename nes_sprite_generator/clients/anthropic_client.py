import json
import logging
import requests
import re
import time
import threading
from typing import Dict, Any, Optional, List

from .base import BaseClient
from ..config import get_api_key

logger = logging.getLogger(__name__)

class AnthropicRateLimiter:
    """Manages rate limiting for Anthropic API requests."""
    
    def __init__(self, requests_per_minute=45, input_tokens_per_minute=18000):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute (default: 45 to stay under 50 limit)
            input_tokens_per_minute: Maximum input tokens per minute
        """
        self.requests_per_minute = requests_per_minute
        self.seconds_per_request = 60.0 / requests_per_minute
        self.input_tokens_per_minute = input_tokens_per_minute
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Request tracking
        self.request_timestamps = []
        self.token_usage = []
        
        # Initialize with current time
        self.last_request_time = time.time()
    
    def wait_for_capacity(self, estimated_tokens=0):
        """
        Wait until we have capacity to make a new request.
        
        Args:
            estimated_tokens: Estimated number of input tokens for this request
            
        Returns:
            Float: The wait time in seconds
        """
        with self.lock:
            current_time = time.time()
            
            # Clean up old timestamps (older than 60 seconds)
            cutoff_time = current_time - 60.0
            self.request_timestamps = [ts for ts in self.request_timestamps if ts > cutoff_time]
            self.token_usage = [(ts, tok) for ts, tok in self.token_usage if ts > cutoff_time]
            
            # Check if we're exceeding request rate
            if len(self.request_timestamps) >= self.requests_per_minute:
                # Calculate time needed for oldest request to expire
                oldest_timestamp = self.request_timestamps[0]
                wait_time_requests = (oldest_timestamp + 60.0) - current_time
            else:
                # Ensure minimum time between requests
                time_since_last = current_time - self.last_request_time
                wait_time_requests = max(0, self.seconds_per_request - time_since_last)
            
            # Check if we're exceeding token rate
            total_tokens = sum(tokens for _, tokens in self.token_usage)
            if total_tokens + estimated_tokens > self.input_tokens_per_minute:
                # Calculate time needed for tokens to expire
                oldest_token_timestamp = self.token_usage[0][0] if self.token_usage else current_time
                wait_time_tokens = (oldest_token_timestamp + 60.0) - current_time
            else:
                wait_time_tokens = 0
            
            # Use the larger wait time
            wait_time = max(wait_time_requests, wait_time_tokens)
            
            if wait_time > 0:
                logger.info(f"Rate limiting: waiting {wait_time:.1f}s (requests: {len(self.request_timestamps)}/{self.requests_per_minute}, tokens: {total_tokens}/{self.input_tokens_per_minute})")
            
            return wait_time
    
    def record_request(self, estimated_tokens=0):
        """Record that a request was made."""
        with self.lock:
            current_time = time.time()
            self.request_timestamps.append(current_time)
            self.token_usage.append((current_time, estimated_tokens))
            self.last_request_time = current_time

class AnthropicClient(BaseClient):
    """Client for generating pixel art using Anthropic's Claude models."""
    
    # Model constants
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-20250219"
    
    # Updated thinking token budgets for extended thinking mode
    THINKING_BUDGETS = {
        "low": 1024,     # Lower thinking effort as per demo.py
        "medium": 2048,  
        "high": 4096    
    }
    
    # Create a shared rate limiter for all instances
    _rate_limiter = AnthropicRateLimiter()
    
    def __init__(self, api_key: Optional[str] = None, model: str = CLAUDE_3_7_SONNET):
        """
        Initialize the Anthropic client with credentials.
        
        Args:
            api_key: Anthropic API key.
            model: Model name or identifier. Can include reasoning level suffix for 3.7
                   (e.g., 'claude-3-7-sonnet-low', 'claude-3-7-sonnet-medium', 'claude-3-7-sonnet-high').
        """
        self.api_key = api_key or get_api_key("anthropic")
        # Use the messages endpoint as in the demo.
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.api_version = "2023-06-01"
        
        # For non-thinking mode, default max_tokens.
        self.max_tokens = 4096  
        self.temperature = 0.5
        
        # Parse model name for extended thinking versions.
        if model.endswith(('-low', '-medium', '-high')):
            parts = model.rsplit('-', 1)
            self.thinking_level = parts[-1]  # 'low', 'medium', or 'high'
            self.thinking_budget = self.THINKING_BUDGETS[self.thinking_level]
            base_model = parts[0]
            # Append the version date if missing.
            if not base_model.endswith("20250219"):
                self.model = f"{base_model}-20250219"
            else:
                self.model = base_model
            self.use_extended_thinking = True

            # Extended thinking mode does not accept a temperature parameter.
            if self.temperature != 0.0:
                logger.warning("Extended thinking mode is enabled; omitting the temperature parameter.")
            # Adjust max_tokens; using a margin as in demo.py.
            if self.max_tokens < self.thinking_budget + self.max_tokens:
                self.max_tokens = self.thinking_budget + self.max_tokens
                logger.info(f"Max tokens increased to {self.max_tokens} to accommodate extended thinking budget.")
        else:
            self.model = model
            self.thinking_level = None
            self.thinking_budget = None
            self.use_extended_thinking = False
            
        logger.info(f"Initialized Anthropic client with model: {self.model}")

    def _make_api_request(self, messages: List[Dict[str, str]], tools=None, retries=3) -> Dict[str, Any]:
        """
        Make a request to the Anthropic API with retry logic and rate limiting.
        
        Args:
            messages: List of message objects with role and content.
            tools: Optional tools configuration.
            retries: Number of retry attempts.
            
        Returns:
            API response as a dictionary.
        """
        # Build headers. Only include the beta header if NOT using extended thinking.
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "Content-Type": "application/json",
        }
        if not self.use_extended_thinking:
            headers["anthropic-beta"] = "output-128k-2025-02-19"
        
        # Convert roles: change "human" to "user".
        converted_messages = []
        for msg in messages:
            role = msg.get("role", "").strip().lower()
            if role == "human":
                role = "user"
            converted_messages.append({
                "role": role,
                "content": msg.get("content", "").strip()
            })

        request_data = {
            "model": self.model,
            "messages": converted_messages,
            "max_tokens": self.max_tokens,
        }
        # Only add the temperature parameter if not in extended thinking mode.
        if not self.use_extended_thinking:
            request_data["temperature"] = self.temperature
        
        # Add extended thinking if enabled.
        if self.use_extended_thinking and self.thinking_budget:
            request_data["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget
            }
        if tools:
            request_data["tools"] = tools

        # Estimate token count (very rough estimation)
        estimated_tokens = int(sum(len(msg.get("content", "").split()) * 1.5 for msg in messages))
        
        attempt = 0
        backoff_factor = 1.0
        
        while attempt < retries:
            try:
                # Wait for rate limiting
                wait_time = self._rate_limiter.wait_for_capacity(estimated_tokens)
                if wait_time > 0:
                    time.sleep(wait_time)
                
                # Record this request
                self._rate_limiter.record_request(estimated_tokens)
                
                # Make the API call
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=request_data,
                    timeout=60  # Add timeout to prevent hanging requests
                )
                
                # Handle specific error codes
                if response.status_code == 429:  # Rate limit exceeded
                    attempt += 1
                    wait_time = 5 * backoff_factor
                    logger.warning(f"Rate limit exceeded (429). Backing off for {wait_time:.1f}s before retry {attempt}/{retries}")
                    time.sleep(wait_time)
                    backoff_factor *= 2  # Exponential backoff
                    continue
                    
                elif response.status_code == 529:  # Service overloaded
                    attempt += 1
                    wait_time = 10 * backoff_factor  # Longer wait for server overload
                    logger.warning(f"Service overloaded (529). Backing off for {wait_time:.1f}s before retry {attempt}/{retries}")
                    time.sleep(wait_time)
                    backoff_factor *= 2  # Exponential backoff
                    continue
                
                # For all other error codes, raise the exception
                response.raise_for_status()
                
                # If we got here, the request was successful
                return response.json()
                
            except requests.exceptions.RequestException as e:
                status = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
                attempt += 1
                
                # If we got a 429 or 529 error code but didn't catch it above
                if status in (429, 529):
                    wait_time = 5 * backoff_factor if status == 429 else 10 * backoff_factor
                    logger.warning(f"Rate limiting error ({status}). Backing off for {wait_time:.1f}s before retry {attempt}/{retries}")
                    time.sleep(wait_time)
                    backoff_factor *= 2  # Exponential backoff
                else:
                    # For other errors, use standard backoff
                    wait_time = backoff_factor * 2 ** attempt
                    logger.error(f"API request failed (attempt {attempt}/{retries}): {e}")
                    if hasattr(e, 'response') and e.response:
                        logger.error(f"Response status: {status}")
                        logger.error(f"Response content: {e.response.text}")
                    logger.info(f"Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
            
        raise Exception("Exceeded maximum retry attempts")

    def _extract_text_from_response(self, response: Dict[str, Any]) -> str:
        """
        Extract raw text from Anthropic API response.
        
        Args:
            response: Anthropic API response.
            
        Returns:
            Raw text content from the response.
        """
        text_content = ""
        for block in response.get("content", []):
            if block.get("type") == "text":
                text_content += block.get("text", "")
        return text_content
    
    def generate_pixel_art(self, 
                         system_prompt: str,
                         user_prompt: str,
                         width: int = 16, 
                         height: int = 16, 
                         max_colors: int = 16) -> Dict[str, Any]:
        """
        Generate pixel art using Anthropic Claude.
        
        Args:
            system_prompt: System prompt with detailed instructions
            user_prompt: User prompt describing what to generate
            width: Width of the pixel canvas
            height: Height of the pixel canvas
            max_colors: Maximum number of colors to use
            
        Returns:
            Dictionary containing the pixel grid, palette, and explanation
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Make the API request
        response = self._make_api_request(messages)
        response_text = self._extract_text_from_response(response)
        
        # Extract JSON from the response
        try:
            # First, try to find JSON block in markdown
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].split("```")[0].strip()
            else:
                # Try to find the JSON object
                match = re.search(r'(\{.*\})', response_text, re.DOTALL)
                if match:
                    json_text = match.group(1)
                else:
                    json_text = response_text.strip()
            
            # Clean up the JSON text
            json_text = re.sub(r',\s*([}\]])', r'\1', json_text)  # Remove trailing commas
            
            # Replace any truncation markers
            json_text = re.sub(r'\[\s*\.\.\.\s*\]', '[]', json_text)
            json_text = re.sub(r'\[\s*etc\.\s*\]', '[]', json_text)
            
            # Parse the JSON
            try:
                pixel_data = json.loads(json_text)
            except json.JSONDecodeError as e:
                # Try one more repair: Find and complete incomplete objects
                logger.warning(f"Initial JSON parsing failed: {e}. Attempting deeper repair.")
                
                # Complete missing closing brackets/braces
                open_braces = json_text.count('{')
                close_braces = json_text.count('}')
                if open_braces > close_braces:
                    json_text += '}' * (open_braces - close_braces)
                    
                open_brackets = json_text.count('[')
                close_brackets = json_text.count(']')
                if open_brackets > close_brackets:
                    json_text += ']' * (open_brackets - close_brackets)
                
                # Try parsing again
                pixel_data = json.loads(json_text)
            
            # Validate the response
            if "pixel_grid" not in pixel_data:
                raise ValueError("Missing pixel_grid in response")
            if "palette" not in pixel_data:
                raise ValueError("Missing palette in response")
            
            # Return the pixel data
            return pixel_data
            
        except Exception as e:
            logger.error(f"Failed to parse response from Claude: {e}")
            logger.error(f"Response text: {response_text[:500]}...")
            raise RuntimeError(f"Failed to parse pixel art response: {e}")
    
    def generate_pixel_grid(self, 
                            prompt: str, 
                            width: int = 16, 
                            height: int = 16, 
                            max_colors: int = 16,
                            style: str = "2D pixel art") -> Dict[str, Any]:
        """
        Generate a pixel grid representation using Anthropic.
        
        Args:
            prompt: Description of the pixel art to create.
            width: Width of the pixel canvas.
            height: Height of the pixel canvas.
            max_colors: Maximum number of unique colors to use.
            style: Style guide for the pixel art.
            
        Returns:
            Dictionary containing the pixel grid, palette, and metadata.
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}. Both width and height must be positive integers.")
        
        # Step 1: Generate the design concept.
        system_prompt_design = (
            f"You are a master pixel art designer specializing in {style}. \n"
            f"Your task is to design a {width}x{height} pixel art as described by the user.\n\n"
            "Follow these guidelines:\n"
            f"- Create detailed, expressive pixel art with rich colors and shading\n"
            f"- Use a palette of up to {max_colors} colors\n"
            "- Use a fully transparent background ([0,0,0,0] in RGBA)\n"
            "- Make the subject fill most of the canvas\n"
            "- Include shadows, highlights, and small details\n"
            "- Keep your design concept brief and focused - under 150 words\n\n"
            "DO NOT return JSON or code in this response. Instead, describe your design concept, \n"
            "color choices, and how you will approach creating this pixel art in a short paragraph."
        )
        
        user_prompt_design = (
            f"Design a {width}x{height} pixel art of: {prompt}\n\n"
            "Please describe your design approach, color palette, and key details you'll include in a SHORT paragraph. \n"
            "Don't create the actual pixel grid yet - just explain your creative vision briefly."
        )
        
        design_messages = [
            {"role": "human", "content": system_prompt_design},
            {"role": "human", "content": user_prompt_design}
        ]
        
        logger.info("Step 1: Generating pixel art design concept...")
        design_response = self._make_api_request(design_messages)
        design_text = self._extract_text_from_response(design_response)
        logger.info("Design concept generated successfully")
        
        # Step 2: Generate the actual pixel grid as JSON.
        system_prompt_json = (
            f"You are a pixel art generator that converts design concepts into precise JSON representations.\n\n"
            "Given a design concept and description, your task is to return ONLY a JSON object representing \n"
            "the pixel art with the following structure:\n\n"
            "{\n"
            f'    "pixel_grid": [\n'
            f'        // {height} rows, each with {width} pixels\n'
            "        // Each pixel is an RGBA array [r,g,b,a] with values 0-255\n"
            "        // Use [0,0,0,0] for fully transparent pixels\n"
            "    ],\n"
            '    "palette": [\n'
            f'        // List of unique RGBA colors used in the grid\n'
            f'        // Maximum {max_colors} colors\n'
            "    ],\n"
            '    "explanation": "Brief explanation (max 150 characters) of the design"\n'
            "}\n\n"
            "CRITICAL REQUIREMENTS:\n"
            f"1. The pixel_grid MUST be exactly {height} rows with {width} pixels per row\n"
            "2. Each pixel MUST be a 4-element array representing RGBA values (0-255)\n"
            "3. Use transparency ([0,0,0,0]) for background pixels\n"
            "4. Return ONLY valid JSON data that can be parsed by json.loads()\n"
            "5. Keep the explanation extremely brief - under 150 characters\n"
            '6. DO NOT include placeholders like "..." or template text'
        )
        
        user_prompt_json = (
            f"Based on this design concept:\n\n---\n{design_text}\n---\n\n"
            f"Create a complete JSON representation of a {width}x{height} pixel art for: {prompt}\n\n"
            "Return ONLY a valid, parseable JSON object with no placeholders, comments, or template text.\n"
            "Every single pixel in the grid must be specified as a complete [r,g,b,a] array.\n"
            "IMPORTANT: Keep the explanation extremely short - under 150 characters maximum."
        )
        
        json_messages = [
            {"role": "human", "content": system_prompt_json},
            {"role": "human", "content": user_prompt_json}
        ]
        
        logger.info("Step 2: Generating pixel grid JSON representation...")
        json_response = self._make_api_request(json_messages)
        json_text = self._extract_text_from_response(json_response)
        
        try:
            # First, clean up potential JSON issues
            if "```json" in json_text:
                json_content = json_text.split("```json")[1].split("```")[0].strip()
            elif "```" in json_text:
                json_content = json_text.split("```")[1].split("```")[0].strip()
            else:
                # Find the outermost JSON object with regex
                match = re.search(r'(\{.*\})', json_text, re.DOTALL)
                if match:
                    json_content = match.group(1)
                else:
                    json_content = json_text.strip()
            
            # Check for and remove ellipses which might indicate truncated content
            if "[...]" in json_content:
                logger.warning("Detected truncated JSON with [...]. Attempting to repair.")
                # Replace [...] with an empty array []
                json_content = json_content.replace("[...]", "[]")
            
            # Replace other potential truncation markers
            json_content = re.sub(r'\[\s*\.\.\.\s*\]', '[]', json_content)
            json_content = re.sub(r'\[\s*etc\.\s*\]', '[]', json_content)
            
            # Remove trailing commas which are invalid in JSON
            json_content = re.sub(r',\s*([}\]])', r'\1', json_content)
            
            logger.info("Attempting to parse JSON response...")
            try:
                pixel_data = json.loads(json_content)
            except json.JSONDecodeError as e:
                # Try one more repair: Find and complete incomplete objects
                logger.warning(f"Initial JSON parsing failed: {e}. Attempting deeper repair.")
                
                # Complete missing closing brackets/braces
                open_braces = json_content.count('{')
                close_braces = json_content.count('}')
                if open_braces > close_braces:
                    json_content += '}' * (open_braces - close_braces)
                    
                open_brackets = json_content.count('[')
                close_brackets = json_content.count(']')
                if open_brackets > close_brackets:
                    json_content += ']' * (open_brackets - close_brackets)
                
                # Try parsing again
                pixel_data = json.loads(json_content)
            
            if not isinstance(pixel_data, dict):
                raise ValueError("Response is not a JSON object")
            
            # Validate and repair the response if needed
            if "pixel_grid" not in pixel_data:
                raise ValueError("Missing 'pixel_grid' in response")
            if "palette" not in pixel_data:
                # If palette is missing but we have a pixel grid, we can extract it
                logger.warning("Missing 'palette' in response. Extracting from pixel_grid.")
                unique_colors = set()
                for row in pixel_data["pixel_grid"]:
                    for pixel in row:
                        if isinstance(pixel, list) and len(pixel) == 4:
                            unique_colors.add(tuple(pixel))
                pixel_data["palette"] = [list(color) for color in unique_colors]
            
            grid = pixel_data["pixel_grid"]
            if len(grid) != height:
                logger.warning(f"Grid height mismatch: expected {height}, got {len(grid)}")
            
            # Verify and repair grid dimensions if needed
            repaired = False
            for i, row in enumerate(grid):
                if len(row) != width:
                    logger.warning(f"Grid width mismatch at row {i}: expected {width}, got {len(row)}")
                    # Pad or trim the row to match expected width
                    if len(row) < width:
                        # Pad with transparent pixels
                        row.extend([[0,0,0,0]] * (width - len(row)))
                        repaired = True
                    elif len(row) > width:
                        # Trim the row
                        grid[i] = row[:width]
                        repaired = True
            
            if repaired:
                logger.info("Repaired pixel grid dimensions to match expected size")
            
            logger.info("Successfully parsed pixel grid JSON")
            return pixel_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"JSON text: {json_text[:500]}...")
            
            # For diagnostics, log more details about the problem
            error_position = e.pos
            context_start = max(0, error_position - 50)
            context_end = min(len(json_text), error_position + 50)
            error_context = json_text[context_start:context_end]
            logger.error(f"Error context around position {error_position}: {error_context}")
            
            raise RuntimeError(f"Failed to parse JSON response: {e}")
        except Exception as e:
            logger.error(f"Error processing Anthropic response: {e}")
            raise