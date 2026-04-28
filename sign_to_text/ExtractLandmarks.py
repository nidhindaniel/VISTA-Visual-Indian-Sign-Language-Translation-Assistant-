"""
ISL Sign Language - Landmark Extraction Script
Extracts MediaPipe hand landmarks from all training images.
Output: Data/Model/landmarks_data.npz (features + labels)
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import time

# Configuration
DATA_DIR = "Data"
MODEL_DIR = os.path.join(DATA_DIR, "Model")
OUTPUT_FILE = os.path.join(MODEL_DIR, "landmarks_data.npz")

# MediaPipe Hands setup
mp_hands = mp.solutions.hands


def get_class_names():
    """Get sorted list of class names from Data directory."""
    classes = []
    for item in os.listdir(DATA_DIR):
        item_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(item_path) and item != "Model":
            classes.append(item)
    return sorted(classes)


def extract_landmarks(image_path, hands):
    """Extract hand landmarks from an image using MediaPipe.
    
    Returns normalized landmark array (63 features) or None if no hand detected.
    Landmarks are normalized relative to wrist position for position invariance.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Convert BGR to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None

    # Take the first detected hand
    hand_landmarks = results.multi_hand_landmarks[0]

    # Extract all 21 landmarks (x, y, z)
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    landmarks = np.array(landmarks, dtype=np.float32)

    # Normalize relative to wrist (landmark 0) for position invariance
    wrist = landmarks[:3].copy()  # x, y, z of wrist
    for i in range(21):
        landmarks[i * 3] -= wrist[0]      # x
        landmarks[i * 3 + 1] -= wrist[1]  # y
        # Keep z relative too
        landmarks[i * 3 + 2] -= wrist[2]  # z

    # Scale normalization - normalize by max distance from wrist
    distances = []
    for i in range(21):
        dx = landmarks[i * 3]
        dy = landmarks[i * 3 + 1]
        dist = np.sqrt(dx**2 + dy**2)
        distances.append(dist)
    
    max_dist = max(distances) if max(distances) > 0 else 1.0
    landmarks /= max_dist

    return landmarks


def main():
    print("=" * 60)
    print("ISL Sign Language - Landmark Extraction")
    print("=" * 60)

    class_names = get_class_names()
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    all_features = []
    all_labels = []
    skipped = 0
    total_processed = 0

    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3
    ) as hands:

        start_time = time.time()

        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(DATA_DIR, class_name)
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            class_detected = 0

            for img_name in images:
                img_path = os.path.join(class_dir, img_name)
                landmarks = extract_landmarks(img_path, hands)

                if landmarks is not None:
                    all_features.append(landmarks)
                    all_labels.append(class_idx)
                    class_detected += 1
                else:
                    skipped += 1

                total_processed += 1

            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            print(
                f"  [{class_idx+1:2d}/{num_classes}] {class_name:>2s}: "
                f"{class_detected}/{len(images)} detected "
                f"({total_processed} total, {rate:.0f} img/s)"
            )

    # Convert to numpy arrays
    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Extraction complete in {elapsed:.1f}s")
    print(f"Total images processed: {total_processed}")
    print(f"Landmarks extracted: {len(features)} ({len(features)/total_processed*100:.1f}%)")
    print(f"Skipped (no hand detected): {skipped}")
    print(f"Feature shape: {features.shape}")

    # Save to npz
    np.savez(OUTPUT_FILE,
             features=features,
             labels=labels,
             class_names=class_names)
    
    file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\nSaved to: {OUTPUT_FILE} ({file_size:.1f} MB)")

    # Save labels.txt
    labels_path = os.path.join(MODEL_DIR, "labels.txt")
    with open(labels_path, 'w') as f:
        for i, name in enumerate(class_names):
            f.write(f"{i} {name}\n")
    print(f"Labels saved to: {labels_path}")


if __name__ == "__main__":
    main()
