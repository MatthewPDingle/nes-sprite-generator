from flask import Flask, render_template, request, jsonify, send_from_directory
from ..api import generate_sprite, list_available_models
import os
import time
import logging

logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

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
                model=data.get('model', 'gpt-4o'),
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

def run_webapp(host='0.0.0.0', port=5000, debug=False):
    """Start the Flask web application"""
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_webapp(debug=True)