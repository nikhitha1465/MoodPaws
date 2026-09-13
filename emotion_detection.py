import cv2
from fer import FER

# Create emotion detector
detector = FER(mtcnn=False)

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Could not open webcam")
    exit()

print("✅ Emotion detection started!")
print("Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("❌ Could not read frame")
        break

    # Mirror the camera
    frame = cv2.flip(frame, 1)

    # Detect emotions
    results = detector.detect_emotions(frame)

    for result in results:
        x, y, w, h = result["box"]
        emotions = result["emotions"]

        # Get strongest emotion
        emotion = max(emotions, key=emotions.get)
        confidence = emotions[emotion]

        # Draw face box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Display emotion
        text = f"{emotion.upper()} ({confidence * 100:.1f}%)"

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("MoodPaws - Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()