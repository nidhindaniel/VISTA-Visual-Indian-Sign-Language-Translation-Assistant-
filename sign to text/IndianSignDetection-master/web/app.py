"""
VISTA - Web Application Server
Flask + WebSocket for real-time ISL sign recognition
and text/audio-to-sign-language animation.
"""

import os
import sys
import base64
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from flask import Flask, render_template, send_from_directory, request, jsonify
from flask_socketio import SocketIO, emit

# ---- Path Setup ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PARENT_DIR, "Data")
MODEL_DIR = os.path.join(DATA_DIR, "Model")
MODEL_PATH = os.path.join(MODEL_DIR, "sign_model_landmarks.pth")

# VISTA project root (for text-to-animation modules)
VISTA_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..'))
if VISTA_ROOT not in sys.path:
    sys.path.insert(0, VISTA_ROOT)

# Video directory for sign language animations
VIDEO_DIR = os.path.join(VISTA_ROOT, "videos")

# ---- Flask App ----
app = Flask(__name__)
app.config['SECRET_KEY'] = 'vista-isl-2025'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ---- Text-to-Animation Engines ----
try:
    from advanced_text_to_gloss import AdvancedGlossTranslator
    from sign_language_player import SignLanguagePlayer
    gloss_engine = AdvancedGlossTranslator()
    sign_player = SignLanguagePlayer(video_dir=VIDEO_DIR)
    ANIMATION_AVAILABLE = True
    print(f"[VISTA] Text-to-Animation engine loaded. {len(sign_player.available_files)} videos indexed.")
except Exception as e:
    ANIMATION_AVAILABLE = False
    gloss_engine = None
    sign_player = None
    print(f"[VISTA] Text-to-Animation not available: {e}")

# ---- Model Architecture (must match Train.py) ----
class LandmarkClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


# ---- Load Model ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[VISTA] Loading model from {MODEL_PATH}...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
class_names = checkpoint['class_names']

model = LandmarkClassifier(checkpoint['num_features'], checkpoint['num_classes'])
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()
print(f"[VISTA] Model loaded! {checkpoint['num_classes']} classes, device: {DEVICE}")

# ---- MediaPipe ----
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)


def normalize_landmarks(hand_landmarks):
    """Extract and normalize hand landmarks (must match ExtractLandmarks.py)."""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    landmarks = np.array(landmarks, dtype=np.float32)

    # Normalize relative to wrist
    wrist = landmarks[:3].copy()
    for i in range(21):
        landmarks[i * 3] -= wrist[0]
        landmarks[i * 3 + 1] -= wrist[1]
        landmarks[i * 3 + 2] -= wrist[2]

    # Scale normalization
    distances = []
    for i in range(21):
        dx = landmarks[i * 3]
        dy = landmarks[i * 3 + 1]
        dist = np.sqrt(dx**2 + dy**2)
        distances.append(dist)

    max_dist = max(distances) if max(distances) > 0 else 1.0
    landmarks /= max_dist

    return landmarks


def get_raw_landmarks(hand_landmarks):
    """Get raw landmark positions for drawing on client canvas."""
    return [[lm.x, lm.y] for lm in hand_landmarks.landmark]


# ---- Routes ----
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/translate')
def translate():
    return render_template('translate.html')


@app.route('/animate')
def animate():
    return render_template('animate.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/signs')
def signs():
    sign_names = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d != 'Model'
    )
    return render_template('signs.html', signs=sign_names)


@app.route('/sign-image/<sign_name>')
def sign_image(sign_name):
    """Serve a sample image for a given sign."""
    sign_dir = os.path.join(DATA_DIR, sign_name)
    if os.path.isdir(sign_dir):
        images = [f for f in os.listdir(sign_dir)
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if images:
            return send_from_directory(sign_dir, images[0])
    return '', 404


@app.route('/videos/<path:filename>')
def serve_video(filename):
    """Serve sign language animation videos."""
    return send_from_directory(VIDEO_DIR, filename)


@app.route('/api/animate', methods=['POST'])
def api_animate():
    """Translate text to ISL gloss and generate sign language animation video."""
    if not ANIMATION_AVAILABLE:
        return jsonify({'error': 'Text-to-animation engine not available.'}), 503

    try:
        data = request.json
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided.'}), 400

        # 1. Convert text to ISL glosses
        glosses = gloss_engine.translate(text)

        # 2. Generate stitched video
        output_filename = 'output.mp4'
        video_path = sign_player.generate_video(glosses, output_filename=output_filename)

        if video_path and os.path.exists(video_path):
            # Anti-cache query param
            video_url = f"/videos/{output_filename}?t={os.path.getmtime(video_path)}"
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
                'error': 'Video generation failed or no matching sign assets found.'
            })

    except Exception as e:
        print(f"[VISTA] Animation error: {e}")
        return jsonify({'error': str(e)}), 500


# ---- WebSocket Events ----
@socketio.on('frame')
def handle_frame(data):
    """Process incoming webcam frame and return prediction."""
    try:
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            emit('prediction', {'hand_detected': False})
            return

        # Run MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            emit('prediction', {'hand_detected': False})
            return

        hand_lm = results.multi_hand_landmarks[0]

        # Get raw landmarks for drawing
        raw_landmarks = get_raw_landmarks(hand_lm)

        # Normalize for model input
        features = normalize_landmarks(hand_lm)

        # Predict
        with torch.no_grad():
            tensor = torch.FloatTensor(features).unsqueeze(0).to(DEVICE)
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            confidence, class_idx = torch.max(probs, 0)

        label = str(class_names[class_idx.item()])
        conf = float(confidence.item())

        emit('prediction', {
            'hand_detected': True,
            'label': label,
            'confidence': conf,
            'landmarks': raw_landmarks
        })

    except Exception as e:
        print(f"[VISTA] Error: {e}")
        emit('prediction', {'hand_detected': False})


# ---- Run ----
if __name__ == '__main__':
    use_ssl = '--ssl' in sys.argv
    ssl_args = {'ssl_context': 'adhoc'} if use_ssl else {}
    port = int(os.environ.get('PORT', 7860))
    
    print("\n" + "=" * 50)
    print("  VISTA - ISL Sign Language Translator")
    if use_ssl:
        print(f"  Running with Adhoc SSL (HTTPS enabled)")
        print(f"  Open: https://localhost:{port} or https://<your-ip>:{port}")
        print("  Note: Your browser may warn you about the connection not being private. Proceed anyway.")
    else:
        print(f"  Open: http://localhost:{port}")
        print("  Note: If testing on another device, you must start the server with 'python app.py --ssl' to enable camera access.")
    print("=" * 50 + "\n")
    
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True, **ssl_args)
