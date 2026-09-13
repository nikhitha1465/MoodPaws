from transformers import pipeline
from PIL import Image

print("Loading emotion model...")

emotion_model = pipeline(
    "image-classification",
    model="trpakov/vit-face-expression"
)

print("Model loaded successfully!")
print(emotion_model)