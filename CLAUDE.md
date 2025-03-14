# NES Sprite Generator - Command Reference

## Command Examples
- Generate single sprite: `python -m nes_sprite_generator single "character description" --width 16 --height 24 --colors 32 --model "claude-3-7-sonnet-low"`
- Create multiple versions: `python -m nes_sprite_generator single "your prompt" --versions 3 --post-process --resize-method bilinear`
- List models: `python -m nes_sprite_generator models`
- Alternative CLI: `python run.py single "prompt" --width 16 --height 24 --colors 32 --model "model-name"`

## Code Style Guidelines
- **Imports**: Standard lib → Third-party → Local imports, grouped and sorted
- **Typing**: Full type annotations using `typing` module (Dict, List, Optional, etc.)
- **Naming**: Classes=PascalCase, functions/variables=snake_case, constants=UPPER_CASE
- **Documentation**: Google-style docstrings with Args, Returns sections
- **Error handling**: Specific exceptions with descriptive messages, try/except with logging
- **Formatting**: 4-space indentation, ~100 char line limit, consistent spacing
- **Logging**: Use built-in logging module with appropriate log levels