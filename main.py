import cv2
import mediapipe as mp

# Create the hand landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

# Open webcam
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Camera not detected")
            break

        # Convert BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert OpenCV image to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # Detect hand
        result = landmarker.detect(mp_image)

        # Draw landmarks
        if result.hand_landmarks:

            for hand in result.hand_landmarks:

                h, w, _ = frame.shape

                for id, landmark in enumerate(hand):

                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    if id == 8:

                        cv2.circle(
                            frame,
                            (x, y),
                            10,
                            (0, 0, 255),
                            -1
                        )

                    else:

                        cv2.circle(
                            frame,
                            (x, y),
                            5,
                            (0, 255, 0),
                            -1
                        )

        cv2.imshow("AirCanvas AI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()