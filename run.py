#!/usr/bin/env python3
"""
Simple CLI launcher for the sprite generator.
Use this if the 'nes-sprite-generator' command isn't available.

Examples:
    python run.py single "A warrior with sword and shield" --width 16 --height 24 --colors 32 --model "claude-3-7-sonnet-low"
    python run.py models
"""
import sys
from nes_sprite_generator.cli import main

if __name__ == "__main__":
    sys.exit(main())