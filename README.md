# 🐱 MoodPaws

MoodPaws is a real-time emotion detection application that uses AI to detect a person's facial emotion and displays a cat image matching the detected mood.

The project combines facial expression recognition with a fun visual response — when your mood changes, your cat companion changes too! 🐾

## ✨ Features

- 🎭 Real-time facial emotion detection
- 🐱 Emotion-based cat reactions
- 📷 Live webcam support
- 😊 Supports 6 emotions:
  - Happy
  - Sad
  - Angry
  - Fear
  - Surprise
  - Neutral
- 📊 Displays detected emotion and confidence
- 🔄 Emotion smoothing for more stable predictions
- 🖼️ Multiple cat images for each emotion
- ⚡ Processes frames periodically for better performance

## 🛠️ Tech Stack

- **Python**
- **OpenCV** – webcam and face detection
- **Hugging Face Transformers** – emotion classification
- **ViT Face Expression Model** – facial emotion recognition
- **NumPy** – image processing
- **Pillow (PIL)** – image conversion and processing

## 🧠 How It Works

```text
Webcam
   ↓
Face Detection
   ↓
Face Cropping
   ↓
Emotion Recognition Model
   ↓
Emotion Smoothing
   ↓
Detected Emotion
   ↓
Matching Cat Image
   ↓
MoodPaws Display