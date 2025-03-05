#!/usr/bin/env python3
import os
import logging

logger = logging.getLogger(__name__)

def load_api_keys(file_path="apikeys.txt"):
    """
    Load API keys from a file with KEY=VALUE format.
    
    Args:
        file_path: Path to the API keys file
        
    Returns:
        dict: Dictionary with API keys
    """
    api_keys = {}
    
    # First try environment variables
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if openai_key:
        api_keys["OPENAI_API_KEY"] = openai_key
    if anthropic_key:
        api_keys["ANTHROPIC_API_KEY"] = anthropic_key
    
    # Then try the file
    try:
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        key, value = line.split("=", 1)
                        api_keys[key.strip()] = value.strip()
                    except ValueError:
                        logger.warning(f"Invalid line in {file_path}: {line}")
    except FileNotFoundError:
        if not (openai_key or anthropic_key):
            logger.warning(f"API keys file not found: {file_path}")
    
    return api_keys

def get_api_key(service, api_keys=None):
    """
    Get API key for a specific service.
    
    Args:
        service: Service name (e.g., "OPENAI" or "ANTHROPIC")
        api_keys: Optional dictionary with API keys
        
    Returns:
        str: API key for the specified service
    """
    if api_keys is None:
        api_keys = load_api_keys()
    
    key_name = f"{service.upper()}_API_KEY"
    api_key = api_keys.get(key_name)
    
    if not api_key:
        raise RuntimeError(f"{service} API key not found in environment or apikeys.txt file.")
    
    return api_key