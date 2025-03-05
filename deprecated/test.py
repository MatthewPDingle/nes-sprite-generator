#!/usr/bin/env python3
"""
Generates a pixel-art NPC sprite (16x24) of a generic townsperson,
upscales it by a given factor (default 4×), and saves it as a PNG with
a transparent background. The sprite is generated using a 32-color palette.
"""

from PIL import Image
import argparse

def generate_default_palette(num_colors):
    """
    Generates a default palette of `num_colors` RGBA tuples.
    The palette is designed for a townsperson sprite and includes
    colors for transparency, skin, hair, clothing, outlines, etc.
    """
    if num_colors < 9:
        raise ValueError("Palette must have at least 9 colors for proper rendering.")

    palette = [None] * num_colors
    # Define key colors:
    palette[0] = (0, 0, 0, 0)           # Transparent
    palette[1] = (255, 220, 177, 255)     # Skin
    palette[2] = (240, 200, 160, 255)     # Alternate skin (unused)
    palette[3] = (60, 40, 20, 255)        # Hair
    palette[4] = (80, 120, 200, 255)      # Shirt
    palette[5] = (50, 50, 100, 255)       # Pants
    palette[6] = (30, 20, 10, 255)        # Shoes
    palette[7] = (0, 0, 0, 255)           # Outline (black)
    palette[8] = (255, 255, 255, 255)     # White (for eyes/highlights)
    palette[9]  = (200, 170, 130, 255)     # Shadow skin
    palette[10] = (100, 80, 60, 255)       # Hair shadow
    palette[11] = (100, 100, 150, 255)     # Dark shirt
    palette[12] = (70, 90, 150, 255)       # Shirt shadow
    palette[13] = (40, 40, 80, 255)        # Dark pants
    palette[14] = (20, 20, 60, 255)        # Pants shadow
    palette[15] = (200, 200, 200, 255)     # Highlight

    # Fill remaining palette slots with arbitrary distinct colors.
    for i in range(16, num_colors):
        r = (i * 8) % 256
        g = (i * 16) % 256
        b = (i * 24) % 256
        palette[i] = (r, g, b, 255)

    return palette

def generate_npc_sprite(width=16, height=24, scale_factor=4, palette_param=32):
    """
    Generates an NPC sprite as a PIL Image.
    
    Parameters:
      - width (int): Width of the sprite in pixels.
      - height (int): Height of the sprite in pixels.
      - scale_factor (int): Factor by which to upscale the sprite.
      - palette_param: Either an integer (number of colors to generate in the palette)
                       or a list of RGB(A) tuples representing a palette.
    
    Returns:
      - A PIL Image object of the upscaled sprite.
    """
    # Basic parameter validation.
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive integers.")
    if scale_factor <= 0:
        raise ValueError("Scale factor must be a positive integer.")

    # Process the palette parameter.
    if isinstance(palette_param, int):
        palette = generate_default_palette(palette_param)
    elif isinstance(palette_param, list):
        if not palette_param:
            raise ValueError("Palette list cannot be empty.")
        palette = palette_param
    else:
        raise ValueError("Palette parameter must be an integer or a list.")

    # Create a new image with a transparent background.
    sprite = Image.new("RGBA", (width, height))
    pixels = sprite.load()  # Pixel access object

    # Mapping of design characters to palette indices.
    char_to_index = {
        " ": 0,   # Transparent
        "O": 7,   # Outline
        "H": 3,   # Hair
        "S": 1,   # Skin
        "W": 8,   # White (eyes/highlights)
        "X": 4,   # Shirt
        "P": 5,   # Pants
        "p": 6,   # Shoes
    }

    # Sprite design (each string represents one row of pixels).
    # The design is based on a 16x24 grid.
    sprite_design = [
        "                ",  # Row  0
        "                ",  # Row  1
        "                ",  # Row  2
        "    OOOOOOOO    ",  # Row  3: Top of head (outline)
        "    OHHHHHHO    ",  # Row  4: Hair on head
        "    OHSSSSHO    ",  # Row  5: Face (skin)
        "    OHSSSSHO    ",  # Row  6: Face (skin)
        "    OHSWWSHO    ",  # Row  7: Eyes row (W = eyes/highlights)
        "    OHSSSSHO    ",  # Row  8: Face (skin)
        "    OOOOOOOO    ",  # Row  9: Chin (outline)
        "   OXXXXXXXXO   ",  # Row 10: Torso (shirt)
        "   OXXXXXXXXO   ",  # Row 11: Torso (shirt)
        "   OXXXXXXXXO   ",  # Row 12: Torso (shirt)
        "   OXXXXXXXXO   ",  # Row 13: Torso (shirt)
        "   OXXXXXXXXO   ",  # Row 14: Torso (shirt)
        "   OXXXXXXXXO   ",  # Row 15: Torso (shirt)
        "  OPPP    PPPO  ",  # Row 16: Upper legs (pants)
        "  OPPP    PPPO  ",  # Row 17: Upper legs (pants)
        "  OPPP    PPPO  ",  # Row 18: Upper legs (pants)
        "  OPPP    PPPO  ",  # Row 19: Upper legs (pants)
        "  oppp    pppo  ",  # Row 20: Shoes
        "  oppp    pppo  ",  # Row 21: Shoes
        "  oppp    pppo  ",  # Row 22: Shoes
        "  oppp    pppo  ",  # Row 23: Shoes
    ]

    # Verify that the design matches the specified dimensions.
    if len(sprite_design) != height:
        raise ValueError("Sprite design height does not match the specified height.")
    for row in sprite_design:
        if len(row) != width:
            raise ValueError("One or more rows in the sprite design do not match the specified width.")

    # Render the sprite by mapping each character to its corresponding palette color.
    for y, row in enumerate(sprite_design):
        for x, char in enumerate(row):
            palette_index = char_to_index.get(char, 0)
            # If the palette index is out of range, default to transparent.
            color = palette[palette_index] if palette_index < len(palette) else (0, 0, 0, 0)
            pixels[x, y] = color

    # Upscale the sprite using nearest-neighbor scaling to preserve the pixelated look.
    upscaled_size = (width * scale_factor, height * scale_factor)
    sprite_upscaled = sprite.resize(upscaled_size, resample=Image.NEAREST)
    return sprite_upscaled

def main():
    """
    Parses command-line arguments and generates the NPC sprite.
    Default parameters: 16x24 resolution, 4x upscale, 32-color palette,
    and output file "npc_sprite.png".
    """
    parser = argparse.ArgumentParser(
        description="Generate a 16x24 pixel NPC sprite and upscale it using nearest-neighbor scaling."
    )
    parser.add_argument("--width", type=int, default=16,
                        help="Width of the sprite (default: 16)")
    parser.add_argument("--height", type=int, default=24,
                        help="Height of the sprite (default: 24)")
    parser.add_argument("--scale", type=int, default=4,
                        help="Upscale factor (default: 4)")
    parser.add_argument("--palette", type=int, default=32,
                        help="Number of colors in the palette (default: 32). "
                             "Alternatively, adjust the code to pass a palette list.")
    parser.add_argument("--output", type=str, default="npc_sprite.png",
                        help="Output filename (default: npc_sprite.png)")

    args = parser.parse_args()

    try:
        sprite_image = generate_npc_sprite(width=args.width,
                                           height=args.height,
                                           scale_factor=args.scale,
                                           palette_param=args.palette)
        sprite_image.save(args.output, "PNG")
        print(f"Sprite saved as {args.output}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
