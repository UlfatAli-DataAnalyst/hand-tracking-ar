import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# Download model file if not exists
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand model... please wait...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )
    print("Download complete!")

# Setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

COLORS = [
    (255, 0, 255),
    (0, 255, 255),
    (0, 255, 100),
    (100, 100, 255),
    (0, 165, 255),
]

TIPS = [4, 8, 12, 16, 20]

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Running! Press Q to quit.")

with HandLandmarker.create_from_options(options) as detector:
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = detector.detect(mp_image)
        num_hands = len(result.hand_landmarks)

        cv2.putText(frame, f"Hands Detected: {num_hands}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        all_tips = []

        for hand in result.hand_landmarks:
            pts = []
            for lm in hand:
                pts.append((int(lm.x * w), int(lm.y * h)))

            for a, b in CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (255, 255, 255), 1)

            for pt in pts:
                cv2.circle(frame, pt, 4, (200, 200, 200), -1)

            for i, tip in enumerate(TIPS):
                all_tips.append((pts[tip], COLORS[i]))

        for i, (p1, c1) in enumerate(all_tips):
            for j, (p2, c2) in enumerate(all_tips):
                if i != j:
                    cv2.line(frame, p1, p2, c1, 2)

        for tip_pt, color in all_tips:
            cv2.circle(frame, tip_pt, 10, color, -1)
            cv2.circle(frame, tip_pt, 15, color, 1)

        cv2.imshow("Hand Tracking AR | Press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Stopped.")