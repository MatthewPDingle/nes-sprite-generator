#!/usr/bin/env python3
import os
import sys
import logging
import re
import argparse
import base64
import time
from datetime import datetime

from openai import OpenAI
from PIL import Image  # For programmatic checks

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def log_api_call(model: str, messages: list, response: dict, call_label: str, files=None):
    """
    Append an API call record to a file named 'api_calls.log'.
    Includes the model, request messages (and any file attachments), and the raw response.
    """
    timestamp = datetime.now().isoformat()
    with open("api_calls.log", "a", encoding="utf-8") as f:
        f.write("====================================\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Call Label: {call_label}\n")
        f.write(f"Model: {model}\n")
        f.write("Messages Sent:\n")
        for i, msg in enumerate(messages, start=1):
            f.write(f"  Message {i}:\n")
            f.write(f"    Role: {msg.get('role', '')}\n")
            content = msg.get("content", "")
            if isinstance(content, list):
                for j, block in enumerate(content, start=1):
                    block_type = block.get("type", "text")
                    f.write(f"      Block {j} (type: {block_type}):\n")
                    if block_type == "text":
                        f.write(f"        Text: {block.get('text', '')}\n")
                    elif block_type == "image_url":
                        url = block.get("image_url", {}).get("url", "")
                        f.write(f"        Image URL: {url[:60]}...[truncated]\n")
            else:
                f.write(f"    Content: {content}\n")
            if "image" in msg and isinstance(msg["image"], str):
                img_data = msg.get("image", "")
                f.write(f"    Image: [base64 data, length={len(img_data)}]\n")
        if files:
            f.write("Files Attached:\n")
            for file in files:
                f.write(f"    Field: {file[0]}, Filename: {file[1][0]}, MIME Type: {file[1][2]}\n")
        f.write("Response:\n")
        f.write(str(response) + "\n")
        f.write("====================================\n\n")


def get_api_key() -> str:
    with open("apikey.txt", "r") as file:
        api_key = file.read().strip()
    if not api_key:
        raise RuntimeError("OpenAI API key not found in apikey.txt.")
    return api_key


def generate_code(prompt: str, file_name: str, model: str = "o3-mini", feedback: str = None) -> str:
    """
    Generate Python code for pixel art based on the given prompt using OpenAI's API.
    If feedback is provided from a previous iteration, it is formatted in a structured way.
    """
    api_key = get_api_key()
    client = OpenAI(api_key=api_key)

    messages = []
    # Construct the user message with all necessary instructions (no system messages)
    if feedback is None:
        user_message = (
            "As a Python coding assistant who outputs only code, your task is to write a complete, self-contained Python script "
            "using the Pillow library to generate pixel art based on the following description:\n"
            f"<description>\n{prompt}\n</description>\n"
        )
    else:
        user_message = (
            "You were originally tasked with writing a complete, self-contained Python script using the Pillow library "
            "to generate pixel art based on the following description:\n"
            f"<description>\n{prompt}\n</description>\n"
        )

    if feedback:
        user_message += (
            "\nAfter attempting this task, you have been given the following feedback:\n"
            f"<feedback>\n{feedback}\n</feedback>\n"
            "Try again."
        )

    user_message += (
        "\nThe script must create an image with the size specified in the description and with a transparent background. "
        "Ensure that the pixel art fills the entire image, uses a rich and varied color palette of at least 8 colors but no bigger than the size found in the description, "
        "and produces a detailed, high-quality sprite similar to those seen in classic NES/SNES RPGs. "
        f"Save the generated image to a file named '{file_name}'. "
        "Output only the code, with no additional commentary."
    )

    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(model=model, messages=messages)
        log_api_call(model, messages, response, "generate_code")
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
    Execute the given Python code string safely.
    """
    try:
        exec_globals = {}
        exec(code, exec_globals)
    except Exception as e:
        logger.error(f"Error during code execution: {e}")


def check_transparent_background(image_file: str, threshold: float = 0.10) -> str:
    """
    Programmatically check if the image has a transparent background by analyzing the border pixels.
    Returns "PASS" if at least `threshold` fraction of border pixels are fully transparent.
    """
    try:
        with Image.open(image_file) as img:
            if img.mode != "RGBA":
                return "Transparent Background check failed: Image mode is not RGBA."
            width, height = img.size
            pixels = img.load()
            border_coords = []

            # Top and bottom rows
            for x in range(width):
                border_coords.append((x, 0))
                border_coords.append((x, height - 1))
            # Left and right columns (excluding corners already added)
            for y in range(1, height - 1):
                border_coords.append((0, y))
                border_coords.append((width - 1, y))

            total = len(border_coords)
            transparent_count = 0
            for coord in border_coords:
                r, g, b, a = pixels[coord]
                if a == 0:
                    transparent_count += 1

            ratio = transparent_count / total
            if ratio >= threshold:
                return f"PASS (Transparent Background: {ratio*100:.1f}% border transparent)"
            else:
                return f"Transparent Background check failed: Only {ratio*100:.1f}% of border pixels are transparent."
    except Exception as e:
        return f"Transparent Background check error: {e}"


def check_color_palette(image_file: str, expected_count: int = 32) -> str:
    """
    Programmatically check the number of unique non-transparent colors in the image.
    Returns "PASS (Unique colors: X)" if the image uses at least 8 unique non-transparent colors,
    otherwise a failure message.
    """
    try:
        with Image.open(image_file) as img:
            img = img.convert("RGBA")
            pixels = list(img.getdata())
            unique_colors = set()
            for color in pixels:
                # Only count if not fully transparent.
                if color[3] != 0:
                    unique_colors.add(color[:3])
            count = len(unique_colors)
            if count >= 8:
                return f"PASS (Unique colors: {count})"
            else:
                return f"Color Palette check failed: Expected at least 8 unique colors, found {count}."
    except Exception as e:
        return f"Color Palette check error: {e}"


def overall_status(vision_feedback: str, transparency_result: str, palette_result: str) -> str:
    """
    Determine overall status.
    Overall PASS is granted only if:
      - The programmatic transparent background check passes.
      - The programmatic color palette check passes.
      - The GPT-4 art critic evaluation (vision_feedback) ends with a line that is exactly 'PASS'.
    Otherwise, overall status is FAIL.
    """
    if not transparency_result.startswith("PASS"):
        return "FAIL"
    if not palette_result.startswith("PASS"):
        return "FAIL"
    lines = [line.strip() for line in vision_feedback.splitlines() if line.strip()]
    if lines and lines[-1] == "PASS":
        return "PASS"
    return "FAIL"


def criteria_met(feedback: str) -> bool:
    """
    Check if the combined feedback indicates that the image meets all criteria.
    This function looks for a line starting with 'Overall:' and checks if it equals 'Overall: PASS'.
    """
    for line in feedback.splitlines():
        if line.strip().startswith("Overall:"):
            return line.strip() == "Overall: PASS"
    return False


def evaluate_image(image_file: str, original_prompt: str, model: str = "gpt-4o") -> str:
    """
    Evaluate the generated image using GPT-4 Vision.
    The image is passed as a base64 data URI embedded in the message content.
    
    The art critic (GPT-4) should evaluate only the following 4 criteria:
      1. Image Composition: Is the subject both horizontally and vertically centered and does it use the full canvas?
      2. Prompt Adherence: Does the image correctly represent the requested content? The original prompt is provided below.
         (Ignore any instructions regarding upscaling or color palette; focus solely on the subject described.)
      3. Image Detail: Does the image have sufficient pixel-level detail?
      4. Subjective Quality: Is the image visually appealing and well-executed?
    
    If the image fails any criteria, please detail what is wrong.
    If all criteria are met, simply respond with 'PASS' on a separate line at the end.
    Provide only the evaluation and nothing else.
    """
    api_key = get_api_key()
    client = OpenAI(api_key=api_key)

    try:
        with open(image_file, "rb") as img_file:
            image_bytes = img_file.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        # Create a data URI string for a PNG image
        data_uri = f"data:image/png;base64,{base64_image}"
    except Exception as e:
        logger.error(f"Failed to read image {image_file}: {e}")
        return "Error reading image."

    user_message_text = (
        "As a model with vision capabilities and an expert pixel art critic, please evaluate the attached image. "
        "Evaluate the image based on the following 4 criteria (do not evaluate background transparency or color palette):\n\n"
        "1. Image Composition: Is the subject both horizontally and vertically centered and does it use the full width and height of the canvas?\n"
        "2. Prompt Adherence: Does the image correctly represent the requested content? The original prompt is as follows:\n"
        f"   {original_prompt}\n"
        "   (Ignore any instructions regarding upscaling or color palette; focus solely on the subject described.)\n"
        "3. Image Detail: Does the image have sufficient pixel-level detail?\n"
        "4. Subjective Quality: Is the image visually appealing and well-executed?\n\n"
        "If the image fails any criteria, please detail what is wrong. "
        "If all criteria are met, simply respond with 'PASS' on a separate line at the end. "
        "Provide only the evaluation and nothing else."
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": user_message_text},
            {"type": "image_url", "image_url": {"url": data_uri}}
        ]
    }]

    try:
        response = client.chat.completions.create(model=model, messages=messages)
        log_api_call(model, messages, response, "evaluate_image")
        vision_feedback = response.choices[0].message.content
    except Exception as e:
        logger.error(f"Image evaluation failed: {e}")
        vision_feedback = "Evaluation failed."
    return vision_feedback


def main():
    # Models:
    # chatgpt-4o-latest
    # gpt-4o
    # gpt-4o-mini
    # o1
    # o1-preview
    # o1-mini
    # o3-mini
    # gpt-4.5-preview
    parser = argparse.ArgumentParser(description="Iterative Pixel Art Refinement with GPT-4 Vision Feedback.")
    parser.add_argument("prompt", help="Description of the pixel art to generate code for")
    parser.add_argument("-x", "--execute", action="store_true",
                        help="Execute the generated code after retrieving it")
    parser.add_argument("-m", "--model", type=str, default="gpt-4.5-preview",
                        help="Model to use (default: gpt-4o).")
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="Maximum number of refinement iterations (default: 10)")
    args = parser.parse_args()

    original_prompt = args.prompt
    max_iterations = args.max_iterations
    current_prompt = original_prompt
    feedback = None
    iteration = 1

    while iteration <= max_iterations:
        file_name = f"img_iter_{iteration:02d}.png"
        logger.info(f"Iteration {iteration}: Generating image -> {file_name}")
        try:
            code = generate_code(current_prompt, file_name, model=args.model, feedback=feedback)
        except Exception as e:
            logger.error(f"Could not generate code: {e}")
            sys.exit(1)

        # Replace any hardcoded "npc_sprite.png" with OUTPUT_FILENAME.
        code = re.sub(r'(["\'])npc_sprite\.png\1', "OUTPUT_FILENAME", code)
        code_to_run = f'OUTPUT_FILENAME = "{file_name}"\n' + code

        if args.execute:
            logger.info(f"Executing code for image {file_name}...")
            execute_code(code_to_run)
            time.sleep(1)
        else:
            logger.info("Execution flag not set; skipping code execution.")

        # Error handling: Check if the image file was created and is a valid image.
        if not os.path.exists(file_name):
            error_message = f"Generated code did not produce an image file: {file_name} does not exist."
            logger.error(error_message)
            feedback = "Image generation error: " + error_message
            iteration += 1
            continue
        try:
            with Image.open(file_name) as img:
                img.verify()  # Verify that it's a valid image
        except Exception as e:
            error_message = f"Generated code produced an invalid image file: {e}"
            logger.error(error_message)
            feedback = "Image generation error: " + error_message
            iteration += 1
            continue

        # Perform programmatic checks
        transparency_result = check_transparent_background(file_name)
        palette_result = check_color_palette(file_name, expected_count=32)
        logger.info(f"Transparent Background check: {transparency_result}")
        logger.info(f"Color Palette check: {palette_result}")

        # Evaluate the generated image using GPT-4 Vision (art critic for 4 criteria)
        logger.info("Evaluating generated image with GPT-4 Vision...")
        vision_feedback = evaluate_image(file_name, original_prompt, model=args.model)

        # Determine overall status based on programmatic checks and GPT-4 feedback
        overall = overall_status(vision_feedback, transparency_result, palette_result)
        combined_feedback = (
            vision_feedback + "\n" +
            transparency_result + "\n" +
            palette_result + "\n" +
            "Overall: " + overall
        )

        logger.info(f"Feedback from evaluation (Iteration {iteration}):\n{combined_feedback}")

        # Save the feedback to a text file alongside the image
        feedback_file = f"img_iter_{iteration:02d}_feedback.txt"
        with open(feedback_file, "w", encoding="utf-8") as fb_file:
            fb_file.write(combined_feedback)

        # Check if the image meets all criteria based on our overall status
        if criteria_met(combined_feedback):
            logger.info(f"Iteration {iteration}: Image meets all criteria. Stopping early.")
            break

        # Prepare new prompt by appending feedback to the original description
        current_prompt = original_prompt + "\nRefinement Feedback: " + combined_feedback
        feedback = combined_feedback
        iteration += 1

    logger.info("Final image generated: " + file_name)


if __name__ == "__main__":
    main()
