# NES Sprite Generator

A Python tool for generating authentic 8-bit NES-style pixel art sprites using AI.

## Overview

NES Sprite Generator leverages modern AI to create pixel art that follows the technical constraints and aesthetic of the Nintendo Entertainment System (8-bit) era. Generate characters, items, enemies, and other game assets that look like they came straight from the 1980s.

## Features

- Generate NES-compatible pixel art sprites using AI
- Support for multiple AI providers (OpenAI, Anthropic)
- Color palette enforcement
- Command-line interface for easy use
- Configurable output formats and sizes

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nes-sprite-generator.git
cd nes-sprite-generator

# Install the package
pip install -e .
```

## Requirements

- Python 3.8+
- API keys for at least one of the supported AI services (OpenAI, Anthropic)

## Usage

### Command Line

```bash
# Basic usage
python -m pixelart single "town blacksmith, frontal view" --width 16 --height 24 --colors 32 --model "claude-3-7-sonnet-low"

# Create 3 versions and rescale to 16x24 if the model doesn't create that by default
python -m pixelart single "girl villager, frontal view" --width 16 --height 24 --colors 32 --model "claude-3-7-sonnet-20250219" --versions 3 --post-process --resize-method bilinear
```

## Configuration

Create a `apikeys.txt` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

Alternatively, set these as environment variables or pass them directly when creating clients.

## Project Structure

```
pixelart/
  __init__.py                 # Package initialization
  __main__.py                 # Entry point for running as module
  config.py                   # Configuration and API key handling
  generator.py                # Main pixel art generation logic
  image_utils.py              # Image processing utilities
  cli.py                      # Command-line interface
  clients/
    __init__.py               # Client factory
    base.py                   # Base class for AI clients
    openai_client.py          # OpenAI-specific implementation
    anthropic_client.py       # Anthropic-specific implementation
```

## License

MIT