#!/usr/bin/env python3
"""
Simple launcher for the web interface.
"""
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the webapp
from nes_sprite_generator.webapp.app import run_webapp

if __name__ == "__main__":
    run_webapp(debug=True)