import cv2
import os
import random
import numpy as np
from transformers import pipeline
from PIL import Image
from collections import Counter


# ============================================================
# 1. LOAD EMOTION MODEL
# ============================================================

print("Loading emotion model...")

emotion_model = pipeline(
    "image-classification",
    model="trpakov/vit-face-expression"
)

print("✅ Emotion model loaded!")


# ============================================================
# 2. FACE DETECTOR
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ============================================================
# 3. FIND CATS FOLDER
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CAT_FOLDER = os.path.join(
    BASE_DIR,
    "cats"
)


# ============================================================
# 4. EMOTION CATEGORIES
# ============================================================

cat_images = {
    "happy": [],
    "sad": [],
    "angry": [],
    "fear": [],
    "surprise": [],
    "neutral": []
}


# ============================================================
# 5. LOAD CAT IMAGES
# ============================================================

print("\n🐱 Loading cat images...")

for emotion in cat_images:

    folder = os.path.join(
        CAT_FOLDER,
        emotion
    )

    if not os.path.exists(folder):

        print(
            f"⚠️ Folder missing: {folder}"
        )

        continue

    for file in os.listdir(folder):

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):

            image_path = os.path.join(
                folder,
                file
            )

            cat_images[emotion].append(
                image_path
            )


# ============================================================
# 6. SHOW LOADED IMAGES
# ============================================================

print("\n==============================")
print("🐱 CAT IMAGES")
print("==============================")

for emotion, images in cat_images.items():

    print(
        f"{emotion:<10} : {len(images)} image(s)"
    )

print("==============================\n")


# ============================================================
# 7. LOAD CAT IMAGE
# ============================================================

def load_cat_image(path):

    if path is None:
        return None

    image = cv2.imread(path)

    if image is None:

        print(
            f"❌ Could not read: {path}"
        )

        return None

    image = cv2.resize(
        image,
        (400, 400)
    )

    return image


# ============================================================
# 8. START CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("❌ Could not open webcam")
    exit()


camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


print("📷 Camera started!")
print("Press Q to quit.")


# ============================================================
# 9. SETTINGS
# ============================================================

PROCESS_EVERY = 5

HISTORY_SIZE = 6

emotion_history = []

frame_count = 0

current_emotion = None

current_confidence = 0.0

current_cat_path = None

cat_image = None


# ============================================================
# 10. MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:

        print("❌ Could not read frame")
        break


    # Selfie view
    frame = cv2.flip(
        frame,
        1
    )


    # ========================================================
    # FACE DETECTION
    # ========================================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )


    frame_count += 1


    # ========================================================
    # PROCESS FACE
    # ========================================================

    for (x, y, w, h) in faces:

        # Draw face box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # Only run AI every few frames
        if frame_count % PROCESS_EVERY != 0:

            continue


        # ----------------------------------------------------
        # CROP FACE
        # ----------------------------------------------------

        face = frame[
            y:y + h,
            x:x + w
        ]

        if face.size == 0:

            continue


        # Resize
        face = cv2.resize(
            face,
            (224, 224)
        )


        # BGR → RGB
        face_rgb = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )


        # PIL image
        pil_image = Image.fromarray(
            face_rgb
        )


        # ====================================================
        # EMOTION PREDICTION
        # ====================================================

        predictions = emotion_model(
            pil_image
        )


        predictions = sorted(
            predictions,
            key=lambda p: p["score"],
            reverse=True
        )


        best = predictions[0]

        detected_emotion = (
            best["label"]
            .strip()
            .lower()
        )

        detected_confidence = (
            best["score"]
        )


        # ====================================================
        # EMOTION SMOOTHING
        # ====================================================

        emotion_history.append(
            detected_emotion
        )


        if len(emotion_history) > HISTORY_SIZE:

            emotion_history.pop(0)


        counts = Counter(
            emotion_history
        )


        stable_emotion = (
            counts.most_common(1)[0][0]
        )


        # ====================================================
        # UPDATE EMOTION
        # ====================================================

        current_confidence = (
            detected_confidence
        )


        if stable_emotion != current_emotion:

            current_emotion = stable_emotion

            print(
                f"\n🐾 MOOD CHANGED → "
                f"{current_emotion.upper()}"
            )


            # =================================================
            # FIND MATCHING CAT
            # =================================================

            available_cats = cat_images.get(
                current_emotion,
                []
            )


            if available_cats:

                current_cat_path = random.choice(
                    available_cats
                )


                cat_image = load_cat_image(
                    current_cat_path
                )


                print(
                    f"🐱 Showing: "
                    f"{os.path.basename(current_cat_path)}"
                )

            else:

                current_cat_path = None

                cat_image = None

                print(
                    f"⚠️ No cat images for "
                    f"{current_emotion}"
                )


    # ========================================================
    # CAMERA PANEL
    # ========================================================

    cv2.putText(
        frame,
        "MOODPAWS",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )


    if current_emotion is not None:

        emotion_text = (
            f"{current_emotion.upper()} "
            f"{current_confidence * 100:.1f}%"
        )

        cv2.putText(
            frame,
            emotion_text,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    # ========================================================
    # CAT PANEL
    # ========================================================

    cat_panel = np.zeros(
        (480, 480, 3),
        dtype=np.uint8
    )


    # Title
    cv2.putText(
        cat_panel,
        "YOUR MOOD CAT",
        (95, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY CAT
    # ========================================================

    if cat_image is not None:

        display_cat = cv2.resize(
            cat_image,
            (400, 400)
        )


        cat_panel[
            55:455,
            40:440
        ] = display_cat


    else:

        cv2.putText(
            cat_panel,
            "Waiting for mood...",
            (100, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )


    # ========================================================
    # COMBINE CAMERA + CAT
    # ========================================================

    camera_panel = cv2.resize(
        frame,
        (640, 480)
    )


    combined = np.hstack(
        (
            camera_panel,
            cat_panel
        )
    )


    # ========================================================
    # SHOW WINDOW
    # ========================================================

    cv2.imshow(
        "MoodPaws - Emotion + Cat",
        combined
    )


    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# 11. CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

print("\n👋 MoodPaws stopped!")