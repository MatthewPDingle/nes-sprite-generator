# NES Sprite Generator

A Python tool for generating 8-bit NES-style pixel art sprites using LLMs.

## Overview

NES Sprite Generator leverages LLMs to create pixel art that hopefully is reminiscent of the Nintendo Entertainment System (8-bit) era. Generate characters, items, enemies, and other game assets that look like they came straight from the 1980s.

## Features

- Generate NES-inspired pixel art sprites using AI
- Support for multiple AI providers (OpenAI, Anthropic, Google)
- Color palette enforcement
- Command-line interface for easy use
- Configurable output formats and sizes

## Quality Assessments
- gpt-4o - OK quality, but abstract and often not good with prompt adherence
- gpt-4o-mini - Generally trash
- gpt-4.5-preview - Surprisingly poor quality, often producing very short characters
- o3-mini-low - Decent quality, often wide characters
- o3-mini-medium - Decent quality, often wide characters
- o3-mini-high - Decent quality, often wide characters
- claude-3-opus-20240229 - Abstract, poor prompt adherence
- claude-3-7-sonnet-20250219 - High quality
- claude-3-7-sonnet-low - Sweet spot.  High quality with limited thinking budget
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
GOOGLE_API_KEY=your_google_api_key
```

## License

MIT