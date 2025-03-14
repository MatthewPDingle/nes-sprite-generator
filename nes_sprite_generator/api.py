from .generator import PixelArtGenerator

def generate_sprite(prompt, width=16, height=24, colors=32, model="gpt-4o", output=None, style="2D pixel art"):
    """
    API function to generate a sprite, usable from both CLI and web.
    
    Args:
        prompt: Description of what to generate
        width: Width in pixels
        height: Height in pixels
        colors: Maximum number of colors
        model: AI model to use
        output: Optional output filename (if None, will only return the data)
        style: Style guide for the generation
        
    Returns:
        dict: Result containing pixel data, image path (if saved), etc.
    """
    generator = PixelArtGenerator(model=model)
    
    # Generate the pixel grid
    pixel_data = generator.generate_pixel_grid(
        prompt=prompt,
        width=width,
        height=height,
        max_colors=colors,
        style=style
    )
    
    # Process the result
    if output:
        result = generator.process_image(
            pixel_data=pixel_data,
            output_file=output,
            post_process=True,
            target_width=width,
            target_height=height,
            max_colors=colors
        )
        return {
            "pixel_data": pixel_data,
            "output_file": result["output"],
            "success": True
        }
    else:
        # Just return the pixel data for web display
        return {
            "pixel_data": pixel_data,
            "success": True
        }

def list_available_models():
    """Return available models as a dictionary."""
    from . import AVAILABLE_MODELS
    return AVAILABLE_MODELS