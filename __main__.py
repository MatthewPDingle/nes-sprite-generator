#!/usr/bin/env python3
"""
Entry point for the pixelart package when run as a module.

This allows users to run the package directly with:
    python -m pixelart
"""
import sys
from .cli import main

if __name__ == "__main__":
    # Call the main function from the CLI module
    sys.exit(main())