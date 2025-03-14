#!/usr/bin/env python3
import os
import sys
import json
import time
import logging
import argparse
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from .api import generate_sprite, list_available_models
from . import AVAILABLE_MODELS

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def process_single_version(version, args, total_versions):
    """
    Process a single version of pixel art generation.
    
    Args:
        version: Version number
        args: Parsed command-line arguments
        total_versions: Total number of versions to generate
        
    Returns:
        Dictionary with results and metadata
    """
    try:
        # Adjust output filename for multiple versions
        if total_versions > 1:
            base, ext = os.path.splitext(args.output)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S") + f"_{version}"
            version_output = f"{base}_{timestamp}{ext}"
            logger.info(f"Starting generation of version {version}/{total_versions}...")
        else:
            version_output = args.output
        
        # Generate the sprite
        result = generate_sprite(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            colors=args.colors,
            model=args.model,
            output=version_output,
            style=args.style
        )
        
        # Verify the success of the generation
        if not result["success"]:
            logger.error(f"Version {version}: Sprite generation failed")
            return {
                "version": version,
                "success": False,
                "error": "Sprite generation failed",
                "output_file": version_output
            }
        
        pixel_data = result["pixel_data"]
        
        # Return results for this version
        return {
            "version": version,
            "output_file": result["output_file"],
            "pixel_data": pixel_data,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error processing version {version}: {e}")
        # Try to determine output filename even in case of error
        output_file = None
        if total_versions > 1:
            base, ext = os.path.splitext(args.output)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S") + f"_{version}"
            output_file = f"{base}_{timestamp}{ext}"
        else:
            output_file = args.output
            
        return {
            "version": version,
            "success": False,
            "error": str(e),
            "output_file": output_file
        }

def handle_single_command(args):
    """
    Handle the 'single' command for generating individual pixel art images.
    
    Args:
        args: Parsed command-line arguments
    """
    # Use concurrent.futures for parallel processing if multiple versions
    if args.versions > 1:
        print(f"Generating {args.versions} versions...")
        
        # Determine max workers - adjust based on the model to avoid rate limits
        if args.max_workers is not None:
            max_workers = args.max_workers
        elif args.model.startswith("claude"):
            # For Anthropic models, limit concurrency more strictly to avoid rate limits
            # Typically, lower is better to prevent 429/529 errors
            max_workers = min(3, args.versions)
            print(f"Using {max_workers} workers for Anthropic API to avoid rate limits")
        else:
            # For other models, use a reasonable default based on CPU count but not too many
            max_workers = min(os.cpu_count() or 4, 5, args.versions)
        
        # Create results storage
        results = []
        completed = 0
        success_count = 0
        
        # Create a progress display
        print(f"0/{args.versions} versions completed", end='\r')
        
        # Process in batches to better manage rate limits
        for batch_start in range(1, args.versions + 1, max_workers):
            batch_end = min(batch_start + max_workers - 1, args.versions)
            batch_size = batch_end - batch_start + 1
            
            logger.info(f"Processing batch: versions {batch_start}-{batch_end}")
            
            # Process this batch with concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit each version in this batch for processing
                futures = {
                    executor.submit(process_single_version, version, args, args.versions): version
                    for version in range(batch_start, batch_end + 1)
                }
                
                # Process results as they come in
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        completed += 1
                        
                        # Update progress
                        print(f"{completed}/{args.versions} versions completed ({success_count} successful)", end='\r')
                        
                        # If successful, provide immediate feedback
                        if result["success"]:
                            success_count += 1
                            if "pixel_data" in result:
                                logger.info(f"Completed version {result['version']}/{args.versions}: {result['output_file']}")
                    except Exception as e:
                        logger.error(f"Error in version processing: {e}")
            
            # If not the last batch, add a brief pause between batches to help with rate limits
            if batch_end < args.versions:
                pause_time = 2.0  # seconds
                logger.info(f"Pausing for {pause_time}s between batches...")
                time.sleep(pause_time)
        
        print(f"\nAll {completed} versions completed! ({success_count} successful)")
        
        # Print summary of all successful versions
        successful = [r for r in results if r["success"] and "pixel_data" in r]
        if successful:
            print("\n=== Generated Versions ===")
            for result in successful:
                version = result["version"]
                output_file = result["output_file"]
                pixel_data = result["pixel_data"]
                
                print(f"\nVersion {version}:")
                print(f"Image: {output_file}")
                
                # Safely access the palette information
                if "palette" in pixel_data:
                    print(f"Palette: {len(pixel_data['palette'])} colors used out of {args.colors} maximum")
                else:
                    print(f"Palette information not available")
        
        # If any failed, print those as well
        failed = [r for r in results if not r["success"]]
        if failed:
            print("\n=== Failed Versions ===")
            for result in failed:
                print(f"Version {result['version']}: {result['error']}")
    else:
        # Process single version
        result = process_single_version(1, args, 1)
        
        if result["success"] and "pixel_data" in result:
            pixel_data = result["pixel_data"]
            print(f"\nPixel Art Generation Complete!")
            print(f"Image: {result['output_file']}")
            print("\nExplanation:")
            print(pixel_data.get("explanation", "No explanation provided"))
            
            # Safely access palette information
            if "palette" in pixel_data:
                print(f"\nPalette: {len(pixel_data['palette'])} colors used out of {args.colors} maximum")
            else:
                print("\nPalette information not available")
            
            # Provide info about model
            model_description = AVAILABLE_MODELS.get(args.model, f"Unknown model: {args.model}")
            print(f"\nUsing model: {args.model} - {model_description}")
        else:
            print(f"\nError generating pixel art: {result['error']}")

def handle_list_models_command(_):
    """
    Handle the 'models' command for listing available models.
    
    Args:
        _: Parsed command-line arguments (unused)
    """
    print(list_available_models())

def main():
    """Main entry point for the pixel art generator CLI."""
    parser = argparse.ArgumentParser(description="Generate pixel art using AI")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Parser for single pixel art generation
    single_parser = subparsers.add_parser("single", help="Generate a single pixel art image")
    single_parser.add_argument("prompt", help="Description of the pixel art to generate")
    single_parser.add_argument("--width", type=int, default=16, help="Width of the pixel canvas")
    single_parser.add_argument("--height", type=int, default=16, help="Height of the pixel canvas")
    single_parser.add_argument("--colors", type=int, default=16, help="Maximum number of colors")
    single_parser.add_argument("--style", type=str, default="2D pixel art", help="Style guide")
    single_parser.add_argument("--output", type=str, default="pixel_art.png", help="Output file name")
    single_parser.add_argument("--model", type=str, default="gpt-4o", help="AI model to use")
    single_parser.add_argument("--versions", type=int, default=1, help="Number of versions to generate")
    single_parser.add_argument("--post-process", action="store_true", help="Post-process the image to fit content to desired dimensions")
    single_parser.add_argument("--resize-method", type=str, default="nearest", 
                              choices=["nearest", "bilinear", "bicubic", "lanczos"], 
                              help="Method to use for resizing during post-processing")
    single_parser.add_argument("--max-workers", type=int, default=None, 
                              help="Maximum number of parallel workers for generation (default: auto-selected based on model)")
    single_parser.set_defaults(func=handle_single_command)
    
    # Parser for listing available models
    models_parser = subparsers.add_parser("models", help="List available AI models")
    models_parser.set_defaults(func=handle_list_models_command)
    
    args = parser.parse_args()
    
    # Execute the appropriate command function
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()