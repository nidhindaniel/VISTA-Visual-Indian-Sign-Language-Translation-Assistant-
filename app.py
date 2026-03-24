import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from advanced_text_to_gloss import AdvancedGlossTranslator
from sign_language_player import SignLanguagePlayer

app = Flask(__name__)

# Initialize VISTA Engines
gloss_engine = AdvancedGlossTranslator()

# Video directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
player = SignLanguagePlayer(video_dir=VIDEO_DIR)

# Ensure the app can serve videos from the videos folder
# We will create a route for this or assume they are static if moved. 
# Better: serve from a specific route.

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory(VIDEO_DIR, filename)

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # 1. Convert to Gloss
        glosses = gloss_engine.translate(text)
        
        # 2. Generate Video
        # Create a unique filename to avoid collision? 
        # For simplicity in this demo, 'output.mp4' or 'timestamp.mp4'
        # Let's use 'output.mp4' and overwrite for single user demo
        output_filename = "output.mp4" 
        video_path = player.generate_video(glosses, output_filename=output_filename)
        
        if video_path and os.path.exists(video_path):
            video_url = f"/videos/{output_filename}?t={os.path.getmtime(video_path)}" # Anti-cache
            return jsonify({
                'text': text,
                'gloss': glosses,
                'video_url': video_url
            })
        else:
            return jsonify({
                'text': text,
                'gloss': glosses,
                'video_url': None,
                'error': 'Video generation failed or no assets found.'
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=7860)
