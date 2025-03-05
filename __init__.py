#!/usr/bin/env python3
import logging
from typing import Optional

from .clients.base import BaseClient
from .clients.openai_client import OpenAIClient
from .clients.anthropic_client import AnthropicClient

logger = logging.getLogger(__name__)

# Map model prefixes to client classes
CLIENT_MAP = {
    "gpt": OpenAIClient,
    "o3": OpenAIClient,
    "claude": AnthropicClient
}

def create_client(model: str, api_key: Optional[str] = None) -> BaseClient:
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
    logger.warning(f"Unknown model prefix in '{model}', defaulting to OpenAIClient")
    return OpenAIClient(api_key=api_key, model=model)

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
    "claude-3-7-sonnet-high": "Claude 3.7 Sonnet with high extended thinking"
}

def list_available_models():
    """Return a formatted string listing all available models with descriptions."""
    result = "Available Models:\n"
    
    # Group by provider
    openai_models = [m for m in AVAILABLE_MODELS.keys() if m.startswith(("gpt", "o3"))]
    anthropic_models = [m for m in AVAILABLE_MODELS.keys() if m.startswith("claude")]
    
    result += "\nOpenAI Models:\n"
    for model in sorted(openai_models):
        result += f"  - {model}: {AVAILABLE_MODELS[model]}\n"
    
    result += "\nAnthropic Models:\n"
    for model in sorted(anthropic_models):
        result += f"  - {model}: {AVAILABLE_MODELS[model]}\n"
    
    return result