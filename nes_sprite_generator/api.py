from .generator import PixelArtGenerator

def generate_sprite(prompt, width=16, height=24, colors=32, model="gpt-4o", output=None, style="2D pixel art", scale=1, post_process=True, resize_method="nearest"):
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
        scale: Scale factor for output image (1=actual size, 8=8x larger)
        post_process: Whether to post-process the image
        resize_method: Method to use for resizing if post-processing
        
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
            post_process=post_process,
            target_width=width,
            target_height=height,
            resize_method=resize_method,
            max_colors=colors,
            scale=scale
        )
        return {
            "pixel_data": pixel_data,
            "output_file": result["output"],
            "success": True,
            "scale": scale,
            "dimensions": result["dimensions"],
            "image_dimensions": result["image_dimensions"]
        }
    else:
        # Just return the pixel data for web display
        return {
            "pixel_data": pixel_data,
            "success": True,
            "scale": scale
        }

def list_available_models():
    """Return available models as a dictionary."""
    from . import AVAILABLE_MODELS
    return AVAILABLE_MODELS