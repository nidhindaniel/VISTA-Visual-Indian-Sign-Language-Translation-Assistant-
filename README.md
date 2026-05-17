<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0F2027,50:203A43,100:2C5364&height=260&section=header&text=VISTA&fontSize=80&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Visual%20Indian%20Sign%20Language%20Translation%20Assistant&descAlignY=60&descSize=18"/>

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Poppins&weight=600&size=28&pause=1000&color=00D9FF&center=true&vCenter=true&width=900&lines=Real-Time+Indian+Sign+Language+Translation;AI+Powered+Gesture+Recognition;Deep+Learning+%2B+Computer+Vision+%2B+NLP;Breaking+Communication+Barriers+with+AI" />
</p>

<p align="center">
  <a href="#">
    <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge"/>
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python"/>
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Framework-Flask-black?style=for-the-badge&logo=flask"/>
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/DeepLearning-PyTorch-red?style=for-the-badge&logo=pytorch"/>
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/ComputerVision-OpenCV-green?style=for-the-badge&logo=opencv"/>
  </a>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-ai-pipeline">AI Pipeline</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-project-structure">Structure</a>
</p>

</div>

---

# 🌟 Overview

## VISTA — Visual Indian Sign Language Translation Assistant

VISTA is an advanced AI-powered system designed to bridge communication gaps through real-time Indian Sign Language (ISL) translation.

The project combines:

- 🧠 Deep Learning
- 👁️ Computer Vision
- 🎙️ Natural Language Processing
- ⚡ Real-Time Inference

to create a seamless two-way communication platform between spoken/written language and Indian Sign Language.

---

# 💡 Why VISTA Matters

Millions of people rely on sign language as their primary mode of communication, yet digital accessibility tools remain limited — especially for Indian Sign Language (ISL).

VISTA aims to create a more inclusive future by enabling:

- 🗣️ Speech/Text → ISL Translation
- ✋ ISL Gesture → Text Recognition
- ⚡ Real-time communication assistance
- 🌍 Accessible AI-driven interaction systems

---

# ✨ Features

<div align="center">

| Feature | Description |
|---|---|
| 🎥 Real-Time Sign Recognition | Detects hand gestures live through webcam input |
| 🧠 Deep Learning Powered | Custom PyTorch neural network for gesture classification |
| 👁️ MediaPipe Landmark Tracking | Extracts precise 3D hand landmarks |
| 🔊 Speech/Text to Sign | Converts text/audio into ISL gloss animations |
| 🎬 Dynamic Video Stitching | Combines ISL sign clips into smooth animations |
| ⚡ Low Latency Inference | Optimized for real-time interaction |
| 🎨 Modern UI | Responsive frontend with smooth animations |
| 🌐 Deployment Ready | Hugging Face compatible architecture |

</div>

---

# 🧠 AI Pipeline

# ✋ Sign → Text Translation

```text
Webcam Feed
      ↓
MediaPipe Hand Tracking
      ↓
21 Hand Landmark Extraction
      ↓
Feature Vector Generation
      ↓
PyTorch Neural Network
      ↓
Predicted Sign Output
      ↓
Readable Text
```

---

# 🗣️ Text/Speech → Sign Translation

```text
Text / Voice Input
        ↓
Speech Recognition (Optional)
        ↓
NLP Processing
        ↓
ISL Gloss Conversion
        ↓
Video Matching Engine
        ↓
Video Stitching
        ↓
Animated Sign Language Output
```

---

# ⚙️ Tech Stack

<div align="center">

## 🚀 Core Technologies

<img src="https://skillicons.dev/icons?i=python,flask,js,html,css"/>

---

## 🧠 AI / Machine Learning

<img src="https://skillicons.dev/icons?i=pytorch,tensorflow"/>

<p>
  <img src="https://img.shields.io/badge/MediaPipe-FF6F00?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas"/>
</p>

---

## 🎨 Frontend

<img src="https://skillicons.dev/icons?i=html,css,js"/>

</div>

---

# 🚀 Installation

## 📋 Prerequisites

Make sure the following are installed:

- Python 3.8+
- pip
- Git

---

## 1️⃣ Clone Repository

```bash
git clone https://github.com/nidhindaniel/VISTA-Visual-Indian-Sign-Language-Translation-Assistant-.git

cd VISTA-Visual-Indian-Sign-Language-Translation-Assistant-
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Run Main Application

```bash
python app.py
```

Application runs on:

```text
http://localhost:7860/
```

---

## 4️⃣ Run Live Sign Detection Module

```bash
cd sign_to_text/web

python app.py
```

Application runs on:

```text
http://localhost:5000/
```

---

# 📂 Project Structure

```text
VISTA/
│
├── app.py
├── advanced_text_to_gloss.py
├── text_to_gloss.py
├── sign_language_player.py
│
├── sign_to_text/
│   ├── Train.py
│   ├── SignRecognition.py
│   └── web/
│       ├── app.py
│       ├── static/
│       └── templates/
│
├── videos/
│
└── Seamless_Looping_Idle_Animation_Creation.mp4
```

---

# 🔬 How The AI Works

## ✋ Sign Recognition System

### Step 1 — Landmark Detection
MediaPipe extracts 21 three-dimensional hand landmarks from live webcam frames.

### Step 2 — Feature Engineering
Coordinates are normalized into a structured feature vector.

### Step 3 — Neural Network Inference
A lightweight fully connected neural network predicts the most probable sign in real-time.

### Step 4 — Text Generation
The predicted gesture is converted into readable text.

---

## 🗣️ NLP Translation System

### Step 1 — Input Processing
Text or speech input is captured and normalized.

### Step 2 — ISL Grammar Transformation
English grammar is converted into Indian Sign Language Gloss structure.

### Step 3 — Video Retrieval
Relevant ISL sign clips are retrieved from the video database.

### Step 4 — Animation Generation
Videos are concatenated dynamically into a seamless sign animation.

---

# 📈 Future Roadmap

- [ ] Sentence-level ISL translation
- [ ] Transformer-based gesture recognition
- [ ] Mobile app deployment
- [ ] Multi-language support
- [ ] Hugging Face deployment
- [ ] Real-time conversational assistant
- [ ] 3D avatar-based sign rendering

---

# 🌍 Potential Applications

- 🏫 Educational Platforms
- 🏥 Healthcare Communication
- 🏢 Public Service Accessibility
- 🎥 Real-Time Translation Systems
- 📱 Accessibility Mobile Apps
- 🤖 Human-AI Interaction Systems

---

# 📜 License

This project is intended for educational and research purposes.

---

# 🌟 Support The Project

If you found this project useful:

⭐ Star the repository  
🍴 Fork the project  
🧠 Contribute ideas and improvements  

---

<div align="center">

## 💙 Built for an Accessible Future

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2C5364,50:203A43,100:0F2027&height=120&section=footer"/>

</div>
