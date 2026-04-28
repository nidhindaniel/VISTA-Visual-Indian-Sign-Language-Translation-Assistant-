"""
ISL Sign Language Recognition - Real-time Inference (MediaPipe Landmarks)
Uses MediaPipe hand landmarks + trained landmark model for recognition.
Draws hand skeleton overlay and shows predictions in real-time.
"""

import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import mediapipe as mp

# Configuration
MODEL_PATH = "Data/Model/sign_model_landmarks.pth"
LABELS_PATH = "Data/Model/labels.txt"
CONFIDENCE_THRESHOLD = 0.6

# Auto-detect device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class LandmarkClassifier(nn.Module):
    """Must match the architecture used during training."""

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


def normalize_landmarks(hand_landmarks):
    """Extract and normalize landmarks from MediaPipe hand result.
    
    Returns 63-feature array normalized relative to wrist position.
    Must match the normalization used during training (ExtractLandmarks.py).
    """
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    landmarks = np.array(landmarks, dtype=np.float32)

    # Normalize relative to wrist (landmark 0)
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


class SignRecognizer:
    def __init__(self):
        """Initialize the sign recognizer with landmark model."""
        print("Loading landmark model...")

        # Load checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        num_features = checkpoint['num_features']
        num_classes = checkpoint['num_classes']
        self.class_names = checkpoint['class_names']

        # Build model and load weights
        self.model = LandmarkClassifier(num_features, num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(DEVICE)
        self.model.eval()
        print(f"Model loaded! ({num_classes} classes, device: {DEVICE})")

    def predict(self, landmarks_array):
        """Run prediction on normalized landmark features."""
        with torch.no_grad():
            tensor = torch.FloatTensor(landmarks_array).unsqueeze(0).to(DEVICE)
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, class_idx = torch.max(probabilities, 0)
            return self.class_names[class_idx.item()], confidence.item()

    def run(self):
        """Main loop for real-time recognition."""
        print("\n" + "=" * 50)
        print("ISL Sign Language Recognition (MediaPipe)")
        print("=" * 50)
        print("Show your hand sign to the camera")
        print("Press 'q' to quit")
        print("Press 'a' for auto-predict mode (toggle)")
        print("Press 'c' to capture and predict")
        print("=" * 50 + "\n")

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        auto_predict = True  # Auto-predict on by default
        last_prediction = ""
        last_confidence = 0

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:

            while True:
                success, img = cap.read()
                if not success:
                    print("Failed to capture frame")
                    break

                # Flip for mirror effect
                img = cv2.flip(img, 1)

                # Convert to RGB for MediaPipe
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                hand_detected = False

                if results.multi_hand_landmarks:
                    hand_detected = True
                    hand_landmarks = results.multi_hand_landmarks[0]

                    # Draw hand skeleton
                    mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Predict
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('a'):
                        auto_predict = not auto_predict
                        print(f"Auto-predict: {'ON' if auto_predict else 'OFF'}")

                    if auto_predict or key == ord('c'):
                        landmarks = normalize_landmarks(hand_landmarks)
                        label, confidence = self.predict(landmarks)
                        last_prediction = label
                        last_confidence = confidence
                else:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('a'):
                        auto_predict = not auto_predict
                        print(f"Auto-predict: {'ON' if auto_predict else 'OFF'}")

                # Display prediction
                if last_prediction and hand_detected:
                    color = (0, 255, 0) if last_confidence >= CONFIDENCE_THRESHOLD else (0, 0, 255)

                    # Semi-transparent background
                    overlay = img.copy()
                    cv2.rectangle(overlay, (10, 10), (300, 80), (0, 0, 0), cv2.FILLED)
                    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

                    # Prediction text
                    display_text = f"{last_prediction} ({last_confidence:.0%})"
                    cv2.putText(img, display_text,
                                (20, 55),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, color, 3)

                # Status bar at bottom
                h = img.shape[0]
                status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
                hand_status = "Hand Detected" if hand_detected else "No Hand Detected - Show your hand"
                mode_text = f"[AUTO] {hand_status}" if auto_predict else f"[MANUAL] {hand_status}"

                overlay2 = img.copy()
                cv2.rectangle(overlay2, (0, h - 40), (img.shape[1], h), (0, 0, 0), cv2.FILLED)
                img = cv2.addWeighted(overlay2, 0.6, img, 0.4, 0)
                cv2.putText(img, mode_text, (10, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                # Display
                cv2.imshow("ISL Sign Recognition", img)

                if key == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Run ExtractLandmarks.py then Train.py first.")
        return

    recognizer = SignRecognizer()
    recognizer.run()


if __name__ == "__main__":
    main()
