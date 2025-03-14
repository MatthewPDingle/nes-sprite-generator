# NES Sprite Generator

A Python tool for generating 8-bit NES-style pixel art sprites using LLMs.

## Overview

NES Sprite Generator leverages LLMs to create pixel art that's reminiscent of the Nintendo Entertainment System (8-bit) era. Generate characters, items, enemies, and other game assets that look like they came straight from the 1980s.

## Why Use This?

Why not just use a text-to-image model? While text-to-image models can create amazing art, specialized LoRAs often struggle with strict pixel art adherence, and the outputs can be difficult to fix. ComfyUI workflows, while powerful, can be complex to set up. Modern LLMs excel at structured outputs like vector graphics and ASCII art, making them surprisingly effective for pixel art generation.

## Features

- Generate NES-inspired pixel art sprites using AI
- Support for multiple AI providers (OpenAI, Anthropic, Google)
- Intelligent sprite processing:
  - Automatic cropping of transparent borders
  - Smart aspect ratio preservation
  - Proper positioning (bottom alignment for tall sprites, centered for wide sprites)
- Color palette optimization and enforcement
- Command-line interface with support for generating multiple versions
- Configurable output scales (1:1 for exact pixel art, scaled for better visibility)

## Quality Assessments
- gpt-4o - OK quality, but abstract and often not good with prompt adherence
- gpt-4o-mini - Generally trash
- gpt-4.5-preview - Surprisingly poor quality, often producing very short characters
- o3-mini-low - Decent quality, often wide characters
- o3-mini-medium - Decent quality, often wide characters
- o3-mini-high - Decent quality, often wide characters
- claude-3-opus-20240229 - Abstract, poor prompt adherence
- claude-3-7-sonnet-20250219 - High quality
- claude-3-7-sonnet-low - Sweet spot. High quality with limited thinking budget
- claude-3-7-sonnet-medium - High quality but not really any better than low thinking budget
- claude-3-7-sonnet-high - High quality but not really any better than low thinking budget
- gemini-2.0-flash - Generally trash, often produces all black characters
- gemini-2.0-pro-exp-02-05 - Sometimes decent, but often produces all black characters
- gemini-2.0-flash-thinking-exp-01-21 - Poor quality, undetailed, and often poor prompt adherence

Check out the examples in the sample outputs directory!

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
- API keys for at least one of the supported AI services (OpenAI, Anthropic, Google)
- Required dependencies: Pillow, scikit-learn (for color optimization)

## Usage

### Command Line

```bash
# Basic usage
python run.py single "town blacksmith, frontal view" --width 16 --height 24 --colors 32 --model "claude-3-7-sonnet-low"

# List available models
python run.py models

# Create multiple versions
python run.py single "warrior with sword and shield" --width 16 --height 24 --colors 32 --model "claude-3-7-sonnet-low" --versions 3

# Enable scaling for better visibility (1=exact pixel size, 8=8x larger)
python run.py single "cute slime monster" --width 16 --height 16 --colors 16 --model "gpt-4o" --scale 8

# Control post-processing behavior
python run.py single "girl villager, frontal view" --width 16 --height 24 --colors 32 --model "claude-3-7-sonnet-20250219" --no-post-process
```

## Configuration

Create an `apikeys.txt` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
```

## Debugging Issues

If you're seeing random-colored pixels instead of coherent pixel art:
1. This could be because the format expected by the renderer (hex color strings) doesn't match the format returned by the AI (RGBA arrays).
2. The generator should automatically detect and convert between these formats.
3. If problems persist, try a different model - some models like Claude 3.7 Sonnet tend to produce more consistent results.

## License

MIT