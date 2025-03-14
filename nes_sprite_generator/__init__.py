#!/usr/bin/env python3
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Importing here to maintain compatibility with existing code
try:
    from .clients.base import BaseClient
    from .clients.openai_client import OpenAIClient
    from .clients.anthropic_client import AnthropicClient
    from .clients.gemini_client import GeminiClient

    # Map model prefixes to client classes
    CLIENT_MAP = {
        "gpt": OpenAIClient,
        "o3": OpenAIClient,
        "claude": AnthropicClient,
        "gemini": GeminiClient
    }
except ImportError as e:
    logger.warning(f"Error importing client modules: {e}")
    # Set up fallback empty maps if imports fail
    CLIENT_MAP = {}

def create_client(model: str, api_key: Optional[str] = None):
    """
    Factory function to create the appropriate client based on the model name.
    
    Args:
        model: Model name or identifier
        api_key: Optional API key (if not provided, it will be loaded from environment or file)
        
    Returns:
        BaseClient: An instance of the appropriate client
    """
    # Default to OpenAI if no prefix match
    model_lower = model.lower()
    
    # Determine the client class based on model prefix
    for prefix, client_class in CLIENT_MAP.items():
        if model_lower.startswith(prefix):
            logger.info(f"Creating {client_class.__name__} for model: {model}")
            return client_class(api_key=api_key, model=model)
    
    # If no match, default to OpenAI
    if "OpenAIClient" in globals():
        logger.warning(f"Unknown model prefix in '{model}', defaulting to OpenAIClient")
        return OpenAIClient(api_key=api_key, model=model)
    else:
        raise ImportError("Could not load any client classes.")

# List of available models with descriptions
AVAILABLE_MODELS = {
    # OpenAI models
    "gpt-4o": "OpenAI GPT-4o model",
    "gpt-4o-mini": "OpenAI GPT-4o-mini model",
    "gpt-5-preview": "OpenAI GPT-5 Preview model",
    "o3-mini-low": "OpenAI o3-mini with low reasoning effort",
    "o3-mini-medium": "OpenAI o3-mini with medium reasoning effort",
    "o3-mini-high": "OpenAI o3-mini with high reasoning effort",
    
    # Anthropic models
    "claude-3-opus-20240229": "Anthropic Claude 3 Opus",
    "claude-3-7-sonnet-20250219": "Anthropic Claude 3.7 Sonnet",
    "claude-3-7-sonnet-low": "Claude 3.7 Sonnet with low extended thinking",
    "claude-3-7-sonnet-medium": "Claude 3.7 Sonnet with medium extended thinking",
    "claude-3-7-sonnet-high": "Claude 3.7 Sonnet with high extended thinking",
    
    # Google Gemini models
    "gemini-2.0-flash": "Google Gemini 2.0 Flash - fast responses (supports function calling)",
    "gemini-2.0-pro-exp-02-05": "Google Gemini 2.0 Pro - more advanced reasoning (supports function calling)",
    "gemini-2.0-flash-thinking-exp-01-21": "Google Gemini 2.0 Flash with enhanced thinking capabilities (text-based approach)"
}

def list_available_models():
    """Return a formatted string listing all available models with descriptions."""
    result = "Available Models:\n"
    
    # Group by provider
    openai_models = [m for m in AVAILABLE_MODELS.keys() if m.startswith(("gpt", "o3"))]
    anthropic_models = [m for m in AVAILABLE_MODELS.keys() if m.startswith("claude")]
    gemini_models = [m for m in AVAILABLE_MODELS.keys() if m.startswith("gemini")]
    
    result += "\nOpenAI Models:\n"
    for model in sorted(openai_models):
        result += f"  - {model}: {AVAILABLE_MODELS[model]}\n"
    
    result += "\nAnthropic Models:\n"
    for model in sorted(anthropic_models):
        result += f"  - {model}: {AVAILABLE_MODELS[model]}\n"
    
    result += "\nGoogle Gemini Models:\n"
    for model in sorted(gemini_models):
        result += f"  - {model}: {AVAILABLE_MODELS[model]}\n"
    
    return result