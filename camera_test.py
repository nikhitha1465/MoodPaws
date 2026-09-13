import cv2

# Open the webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Could not open webcam")
    exit()

print("✅ Webcam started!")
print("Press Q to quit.")

while True:
    success, frame = camera.read()
    frame = cv2.flip(frame, 1)

    if not success:
        print("❌ Could not read frame")
        break

    cv2.imshow("MoodPaws - Camera Test", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()