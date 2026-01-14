# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# model_path = '/Users/kaganakcelik/dev/ai/aero/hand_landmarker.task'
import cv2
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

cap = cv2.VideoCapture(0)

start_time = time.time()
last_ts = -1

# Create a hand landmarker instance with the video mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)
with HandLandmarker.create_from_options(options) as landmarker:
    # The landmarker is initialized. Use it here.
    # ...
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        timestamp_ms = int((time.time() - start_time) * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                for landmark in hand_landmarks:
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)

        cv2.imshow('Hand Landmarks', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()        