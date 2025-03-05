#!/usr/bin/env python3
import os
import sys
import logging
import re
from openai import OpenAI  # OpenAI client from the latest openai API

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def generate_code(prompt: str, file_name: str, model: str = "o3-mini") -> str:
    """
    Generate Python code for pixel art based on the given prompt using OpenAI's API.
    Returns the code as a string. Raises an exception if the API call fails.
    
    Reads the API key from apikey.txt.
    Dynamically constructs the messages:
      - For non-o-series models, include a system message.
      - For o-series models (e.g., o1-mini, o1-preview, o1, o3-mini, o3), omit the system message.
      
    The user prompt instructs the model to generate a complete, self-contained Python script that:
      - Uses the Pillow library to create an image.
      - Creates an image with the size specified in the description and with a transparent background.
      - Utilizes a rich and varied color palette of the size found in the description
        and produces a detailed, high-quality sprite similar to classic NES/SNES-era RPG characters.
      - The code saves the generated image to a specified file.
      - Outputs only the code with no extra commentary.
    """
    # Read API key from apikey.txt
    with open("apikey.txt", "r") as file:
        api_key = file.read().strip()

    if not api_key:
        raise RuntimeError("OpenAI API key not found in apikey.txt.")

    client = OpenAI(api_key=api_key)
    
    # Dynamically set messages based on the model
    messages = []
    if not (model.lower().startswith("o1") or model.lower().startswith("o3")):
        messages.append({"role": "system", "content": "You are a Python coding assistant who outputs only code."})
    
    user_message = (
        f"Write a complete, self-contained Python script using the Pillow library to generate pixel art based on the following description: "
         f"<description>{prompt}</description>. "
        "The script must create an image with the size specified in the description and with a transparent background. "
        "Ensure that the pixel art fills the entire image, uses a rich and varied color palette of the size found in the description, "
        "and produces a detailed, high-quality sprite similar to those seen in classic NES/SNES RPGs. "
        f"Save the generated image to a file named '{file_name}'. "
        "Output only the code, with no additional commentary."
    )

    messages.append({"role": "user", "content": user_message})
    
    try:
        response = client.chat.completions.create(model=model, messages=messages)
    except Exception as e:
        logger.error(f"OpenAI API request failed: {e}")
        raise

    try:
        code_content = response.choices[0].message.content
    except Exception as e:
        logger.error(f"Failed to parse code from response: {e}")
        raise

    if code_content.strip().startswith("```"):
        code_content = code_content.strip().strip("```")
        code_content = code_content.replace("python\n", "", 1).replace("python\r\n", "", 1)
    return code_content.strip()

def execute_code(code: str):
    """
    Execute the given Python code string safely. Any output produced by the code will be shown.
    Catches and logs errors that occur during execution.
    """
    try:
        exec_globals = {}
        exec(code, exec_globals)
    except Exception as e:
        logger.error(f"Error during code execution: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate pixel art Python code from a natural language prompt.")
    parser.add_argument("prompt", help="Description of the pixel art to generate code for")
    parser.add_argument("-x", "--execute", action="store_true",
                        help="Execute the generated code after retrieving it")
    parser.add_argument("-m", "--model", type=str, default="gpt-4o",
                        help="Model to use (default: gpt-4o). For o-series models (o1-/o3-), the system message is omitted.")
    parser.add_argument("-n", "--iterations", type=int, default=1, choices=range(1, 101),
                        help="Number of unique images to generate (1-100)")
    args = parser.parse_args()

    prompt = args.prompt

    for i in range(1, args.iterations + 1):
        file_name = f"img{i:03d}.png"
        logger.info(f"Generating image iteration {i} -> {file_name}")
        try:
            # Generate new code for each iteration, including the file_name in the prompt.
            code = generate_code(prompt, file_name, model=args.model)
        except Exception as e:
            logger.error(f"Could not generate code: {e}")
            sys.exit(1)

        # Replace any hardcoded "npc_sprite.png" with OUTPUT_FILENAME.
        code = re.sub(r'(["\'])npc_sprite\.png\1', "OUTPUT_FILENAME", code)

        # Prepend the assignment of OUTPUT_FILENAME to the generated code.
        code_to_run = f'OUTPUT_FILENAME = "{file_name}"\n' + code

        if args.execute:
            logger.info(f"Executing code for image {file_name}...")
            execute_code(code_to_run)
    logger.info("All iterations finished.")

if __name__ == "__main__":
    main()
