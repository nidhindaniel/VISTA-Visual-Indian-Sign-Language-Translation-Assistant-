---
title: VISTA - Visual Indian Sign Language Translation Assistant
emoji: 🌟
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

<div align="center">
  <h1>  VISTA </h1>
  <h3>Visual Indian Sign Language Translation Assistant</h3>

  <p align="center">
    <strong>Breaking down communication barriers with real-time, two-way Indian Sign Language translation powered by Deep Learning.</strong>
  </p>

  <p align="center">
    <a href="#features">Features</a> •
    <a href="#how-it-works">How It Works</a> •
    <a href="#tech-stack">Tech Stack</a> •
    <a href="#installation">Installation</a> •
    <a href="#project-structure">Structure</a>
  </p>
</div>

---

## 📖 Overview

**VISTA** (Visual Indian Sign Language Translation Assistant) is an advanced, end-to-end web application that provides seamless, real-time two-way translation between spoken/written language and Indian Sign Language (ISL). 

By leveraging cutting-edge deep learning, computer vision, and natural language processing, VISTA allows users to translate text or voice inputs into highly accurate ISL gloss animations, and can detect real-time ISL gestures via webcam to convert them back into readable text.

---

## ✨ Features

- **Real-Time Sign-to-Text Recognition:** Uses your webcam to capture hand gestures, extracts precise landmarks via MediaPipe, and classifies them into text using a custom-trained, lightweight PyTorch Neural Network.
- **Text/Audio-to-Sign Animation:** Translates English text or voice input into grammatically correct ISL Gloss structures using advanced NLP algorithms, then concatenates pre-recorded sign videos into a seamless, fluid animation.
- **High-Performance Architecture:** Inference is heavily optimized using PyTorch. Designed for smooth, low-latency translation even on consumer-grade hardware.
- **Premium Web Interface:** Beautiful, responsive UI built with Vanilla CSS, dynamic micro-animations, and a responsive glassmorphism aesthetic.
- **Hugging Face Ready:** Configured and ready to be deployed as a Hugging Face Space for widespread accessibility.

---

## 🛠 Tech Stack

### Core Technologies
- **Python 3.x**
- **Flask**: Robust backend framework serving API endpoints and web templates.

### Machine Learning & Computer Vision
- **PyTorch**: Framework for defining, training, and running the lightweight fully-connected neural network.
- **MediaPipe**: For high-accuracy, real-time hand landmark extraction.
- **OpenCV**: Handles video capturing, processing, and output rendering.
- **NumPy & Pandas**: Data manipulation and feature extraction.

### Frontend
- **HTML5 & Vanilla CSS**: Custom-tailored design system with modern typography and animations.
- **JavaScript (ES6)**: Real-time interactions, API handling, and asynchronous video rendering.

---

## 🚀 Installation & Setup

### Prerequisites
Make sure you have **Python 3.8+** installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/nidhindaniel/VISTA-Visual-Indian-Sign-Language-Translation-Assistant-.git
cd VISTA-Visual-Indian-Sign-Language-Translation-Assistant-
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note: Make sure PyTorch, OpenCV, Flask, and MediaPipe are installed if a `requirements.txt` is missing)*

### 3. Run the Application
VISTA includes a modular environment. Depending on what you want to launch:

**To launch the complete application (Text to Sign Video):**
```bash
python app.py
```
*The app will run on `http://localhost:7860/`.*

**To launch the Live Sign Detection interface:**
```bash
cd "sign_to_text/web"
python app.py
```
*The app will run on `http://localhost:5000/`.*

---

## 📁 Project Structure

```text
VISTA/
├── app.py                      # Main entry for Text-to-Gloss Video generation API
├── advanced_text_to_gloss.py   # NLP algorithms for translating English to ISL Gloss
├── text_to_gloss.py            # Baseline gloss engine
├── sign_language_player.py     # Video concatenation and rendering engine
├── sign_to_text/
│   ├── Train.py            # PyTorch Model training script for Landmark classifier
│   ├── SignRecognition.py  # Inference script for real-time sign detection
│   └── web/
│       ├── app.py          # Flask interface for live Sign-to-Text detection
│       ├── static/         # CSS, JS, and Images for the web app
│       └── templates/      # HTML views (index.html, translate.html, etc.)
├── videos/                     # Asset folder containing dictionary of ISL sign videos
└── Seamless_Looping_Idle_Animation_Creation.mp4 # Example Idle Animation
```

---

## How the AI Works

### Sign Detection (Vision -> Text)
1. **Feature Extraction:** A user gestures into the webcam. `MediaPipe` calculates 21 3D landmarks for the hand.
2. **Preprocessing:** Coordinates are normalized and transformed into a 63-dimensional feature array (x, y, z for each point).
3. **Inference:** A 4-layer fully connected network (`LandmarkClassifier` built in PyTorch) runs inference, identifying the most probable sign with an accuracy upward of 91%.

### Text/Speech Translation (Text -> Vision)
1. **NLP Processing:** User provides text/audio. `advanced_text_to_gloss.py` strips stop words, normalizes stems, and rearranges English (SVO grammar) to match the SOV (Subject-Object-Verb) grammatical constraints of Indian Sign Language.
2. **Video Assembly:** `sign_language_player.py` references the generated gloss array, searches the `videos/` database, and concatenates the corresponding ISL sign clips into a cohesive animation stream delivered to the frontend.

---

<div align="center">
  <b>Built with ❤️ for an accessible future.</b>
</div>
