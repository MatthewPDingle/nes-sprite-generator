from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from ..api import generate_sprite, list_available_models
import os
import time
import logging
import json
import uuid
import requests
import base64
from io import BytesIO
from PIL import Image
import threading
import numpy as np

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Storage for sprite sheet generation sessions
sprite_sheet_sessions = {}

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/spritesheet')
def spritesheet():
    return render_template('spritesheet.html')

@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.json
    
    # Get number of versions
    versions = int(data.get('versions', 1))
    
    # Generate unique output filenames for each version
    timestamp = int(time.time())
    output_dir = os.path.join(app.static_folder, 'generated')
    os.makedirs(output_dir, exist_ok=True)
    
    image_urls = []
    all_successful = True
    error_message = None
    
    try:
        for i in range(versions):
            version_suffix = f"_{i+1}" if versions > 1 else ""
            output_file = os.path.join(output_dir, f"sprite_{timestamp}{version_suffix}.png")
            
            # Call the API function with post-processing enabled by default
            result = generate_sprite(
                prompt=data.get('prompt'),
                width=int(data.get('width', 16)),
                height=int(data.get('height', 24)),
                colors=int(data.get('colors', 32)),
                model=data.get('model', 'gemini-2.0-flash-exp'),
                output=output_file,
                post_process=True,  # Enable post-processing by default
                resize_method="bilinear"  # Use bilinear for smoother resizing
            )
            
            if result['success']:
                image_urls.append(f"/static/generated/sprite_{timestamp}{version_suffix}.png")
            else:
                all_successful = False
                error_message = result.get('error', 'Unknown error')
                image_urls.append(None)  # Add None for failed generation
        
        # Return JSON response
        return jsonify({
            'success': True,
            'image_urls': image_urls,
            'explanation': result['pixel_data'].get('explanation', 'No explanation provided')
        })
    except Exception as e:
        logger.error(f"Error generating sprite: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models')
def api_models():
    models = list_available_models()
    return jsonify(models)

@app.route('/api/generate-spritesheet', methods=['POST'])
def api_generate_spritesheet():
    """
    Initiate a sprite sheet generation session.
    Expects JSON data with:
    - poses: list of pose descriptions
    - description: character description
    - image_data: base64 encoded image data (optional)
    - image_url: URL to reference image (optional)
    """
    try:
        data = request.json
        poses = data.get('poses', [])
        description = data.get('description', '')
        image_data = data.get('image_data')
        image_url = data.get('image_url')
        
        if not poses:
            return jsonify({'success': False, 'error': 'No poses specified'})
        
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        
        # Initialize the session data
        sprite_sheet_sessions[session_id] = {
            'status': 'starting',
            'description': description,
            'poses': poses,
            'image_data': image_data,
            'image_url': image_url,
            'generated_poses': [{'pose': pose, 'status': 'pending', 'image_url': None} for pose in poses],
            'completed': 0,
            'total': len(poses),
            'sprite_sheet_url': None,
            'error': None,
            'debug_images': {}
        }
        
        # Start a background thread to process the sprite sheet
        thread = threading.Thread(
            target=process_sprite_sheet, 
            args=(session_id, poses, description, image_data, image_url)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': f'Sprite sheet generation started with {len(poses)} poses'
        })
        
    except Exception as e:
        logger.error(f"Error starting sprite sheet generation: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/spritesheet-progress')
def api_spritesheet_progress():
    """Check the progress of a sprite sheet generation session"""
    session_id = request.args.get('session_id')
    
    if not session_id or session_id not in sprite_sheet_sessions:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session ID'
        }), 404
    
    # Get session data
    session = sprite_sheet_sessions[session_id]
    
    return jsonify({
        'success': True,
        'status': session['status'],
        'completed': session['completed'],
        'total': session['total'],
        'generated_poses': session['generated_poses'],
        'sprite_sheet_url': session['sprite_sheet_url'],
        'error': session['error']
    })

def process_sprite_sheet(session_id, poses, description, image_data, image_url):
    """
    Background thread to process sprite sheet generation.
    
    Args:
        session_id: Unique session identifier
        poses: List of pose descriptions to generate
        description: Character description
        image_data: Base64 encoded image data (optional)
        image_url: URL to reference image (optional)
    """
    session = sprite_sheet_sessions[session_id]
    session['status'] = 'in_progress'
    
    try:
        # Get the reference image (if any)
        reference_image = None
        if image_data:
            # Convert base64 to image
            try:
                # Remove data URL prefix if present
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                image_binary = base64.b64decode(image_data)
                reference_image = Image.open(BytesIO(image_binary))
                logger.info(f"Loaded reference image from uploaded data: {reference_image.size}")
            except Exception as e:
                logger.error(f"Error loading reference image from data: {e}")
        elif image_url:
            # Download image from URL
            try:
                # If the URL is relative, make it absolute
                if image_url.startswith('/'):
                    # Construct absolute URL based on host
                    image_url = request.host_url.rstrip('/') + image_url
                
                # If it's a local static file, load directly
                if image_url.startswith(request.host_url) and '/static/' in image_url:
                    local_path = image_url.split('/static/')[1]
                    img_path = os.path.join(app.static_folder, local_path)
                    if os.path.exists(img_path):
                        reference_image = Image.open(img_path)
                        logger.info(f"Loaded reference image from local file: {reference_image.size}")
                    else:
                        logger.error(f"Local file not found: {img_path}")
                else:
                    # Download from external URL
                    response = requests.get(image_url, stream=True)
                    if response.status_code == 200:
                        reference_image = Image.open(BytesIO(response.content))
                        logger.info(f"Loaded reference image from URL: {reference_image.size}")
                    else:
                        logger.error(f"Failed to download image: {response.status_code}")
            except Exception as e:
                logger.error(f"Error downloading reference image: {e}")
        
        # Extract the color palette from the reference image if available
        reference_palette = None
        if reference_image:
            try:
                from ..image_utils import image_to_pixel_grid
                pixel_grid, palette = image_to_pixel_grid(reference_image)
                reference_palette = palette
                logger.info(f"Extracted palette with {len(palette)} colors")
                
                # Save reference image dimensions
                reference_width, reference_height = reference_image.size
            except Exception as e:
                logger.error(f"Error extracting palette: {e}")
        
        # Create output directory for sprite sheet
        timestamp = int(time.time())
        output_dir = os.path.join(app.static_folder, 'generated')
        debug_dir = os.path.join(output_dir, f'spritesheet_{timestamp}')
        os.makedirs(debug_dir, exist_ok=True)
        
        # Process each pose
        for i, pose in enumerate(poses):
            try:
                # Update status
                session['generated_poses'][i]['status'] = 'generating'
                
                # Create the prompt for the pose
                pose_prompt = f"{description} in a {pose} pose with a white background."
                
                # Generate the sprite using the gemini-2.0-flash-exp model
                # First, set output filename for this pose
                pose_filename = f"pose_{timestamp}_{i+1}.png"
                pose_output_path = os.path.join(output_dir, pose_filename)
                
                # Set up the request params
                request_params = {
                    'prompt': pose_prompt,
                    'model': 'gemini-2.0-flash-exp',
                    'width': reference_width if reference_image else 16, 
                    'height': reference_height if reference_image else 24,
                    'colors': len(reference_palette) if reference_palette else 32,
                    'output': pose_output_path,
                    'post_process': True
                }
                
                # If we have a reference image, pass it to the model
                if reference_image:
                    # Save reference image to a temp file
                    ref_temp_path = os.path.join(debug_dir, f"reference_{i}.png")
                    reference_image.save(ref_temp_path)
                    request_params['reference_image'] = ref_temp_path
                
                # Generate the sprite
                logger.info(f"Generating pose {i+1}/{len(poses)}: {pose}")
                result = generate_sprite(**request_params)
                
                if result['success']:
                    # Store results
                    pose_url = f"/static/generated/{pose_filename}"
                    session['generated_poses'][i] = {
                        'pose': pose,
                        'status': 'completed',
                        'image_url': pose_url
                    }
                    
                    # Debug: save intermediate steps if available
                    if 'debug_images' in result:
                        for step, step_image in result['debug_images'].items():
                            step_filename = f"pose_{i+1}_{step}.png"
                            step_path = os.path.join(debug_dir, step_filename)
                            step_image.save(step_path)
                            
                            # Add to session for debugging
                            if 'debug_images' not in session:
                                session['debug_images'] = {}
                            if i not in session['debug_images']:
                                session['debug_images'][i] = {}
                            session['debug_images'][i][step] = f"/static/generated/spritesheet_{timestamp}/{step_filename}"
                else:
                    # Handle failure
                    session['generated_poses'][i] = {
                        'pose': pose,
                        'status': 'failed',
                        'image_url': None,
                        'error': result.get('error', 'Unknown error')
                    }
                    logger.error(f"Failed to generate pose {i+1}: {result.get('error')}")
            
            except Exception as e:
                # Handle any exceptions during generation
                session['generated_poses'][i] = {
                    'pose': pose,
                    'status': 'failed',
                    'image_url': None,
                    'error': str(e)
                }
                logger.error(f"Exception generating pose {i+1}: {e}")
            
            # Update completion count
            session['completed'] += 1
        
        # Create the sprite sheet once all poses are generated
        try:
            # Get all successfully generated poses
            successful_poses = [pose for pose in session['generated_poses'] 
                              if pose['status'] == 'completed' and pose['image_url']]
            
            if successful_poses:
                spritesheet_path = create_sprite_sheet(successful_poses, output_dir, timestamp, debug_dir)
                session['sprite_sheet_url'] = f"/static/generated/spritesheet_{timestamp}.png"
                session['status'] = 'completed'
                logger.info(f"Sprite sheet created successfully: {spritesheet_path}")
            else:
                session['status'] = 'failed'
                session['error'] = "No poses were successfully generated"
                logger.error("Failed to create sprite sheet: no successful poses")
        
        except Exception as e:
            session['status'] = 'failed'
            session['error'] = f"Error creating sprite sheet: {str(e)}"
            logger.error(f"Error creating sprite sheet: {e}")
    
    except Exception as e:
        # Handle any exceptions during the overall process
        session['status'] = 'failed'
        session['error'] = str(e)
        logger.error(f"Error in sprite sheet generation process: {e}")

def create_sprite_sheet(poses, output_dir, timestamp, debug_dir):
    """
    Create a sprite sheet from the generated poses.
    
    Args:
        poses: List of pose data dictionaries
        output_dir: Directory to save the output
        timestamp: Timestamp for unique filenames
        debug_dir: Directory for debug images
        
    Returns:
        Path to the created sprite sheet
    """
    logger.info(f"Creating sprite sheet with {len(poses)} poses")
    
    # Load all the pose images
    pose_images = []
    for pose in poses:
        try:
            # Get local path from URL
            url_path = pose['image_url'].split('/static/')[1]
            img_path = os.path.join(app.static_folder, url_path)
            
            # Load the image
            img = Image.open(img_path)
            pose_images.append(img)
            
            # Save a copy to debug dir
            debug_path = os.path.join(debug_dir, os.path.basename(img_path))
            img.copy().save(debug_path)
        except Exception as e:
            logger.error(f"Error loading pose image {pose['pose']}: {e}")
    
    if not pose_images:
        raise ValueError("No valid pose images to create sprite sheet")
    
    # Determine the grid layout
    grid_size = int(np.ceil(np.sqrt(len(pose_images))))
    rows = grid_size
    cols = grid_size
    
    # Make sure we have enough cells
    while rows * cols < len(pose_images):
        cols += 1
    
    # Get the maximum dimensions
    max_width = max(img.width for img in pose_images)
    max_height = max(img.height for img in pose_images)
    
    # Create the sprite sheet canvas
    sheet_width = cols * max_width
    sheet_height = rows * max_height
    
    # Create with transparent background
    sprite_sheet = Image.new('RGBA', (sheet_width, sheet_height), (0, 0, 0, 0))
    
    # Place each image on the grid
    for i, img in enumerate(pose_images):
        row = i // cols
        col = i % cols
        
        # Calculate position (centered in the cell)
        x = col * max_width + (max_width - img.width) // 2
        y = row * max_height + (max_height - img.height) // 2
        
        # Paste the image onto the sprite sheet
        sprite_sheet.paste(img, (x, y), img)
    
    # Save the sprite sheet
    output_path = os.path.join(output_dir, f"spritesheet_{timestamp}.png")
    sprite_sheet.save(output_path)
    
    # Save a copy to debug dir
    debug_path = os.path.join(debug_dir, f"spritesheet_{timestamp}.png")
    sprite_sheet.save(debug_path)
    
    return output_path

def run_webapp(host='0.0.0.0', port=5000, debug=False):
    """Start the Flask web application"""
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_webapp(debug=True)