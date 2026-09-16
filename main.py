import cv2
import mediapipe as mp
import numpy as np
# -----------------------------
# MediaPipe setup
# -----------------------------

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

# -----------------------------
# Webcam
# -----------------------------

cap = cv2.VideoCapture(0)

# -----------------------------
# Drawing variables
# -----------------------------

prev_x = None
prev_y = None

# -----------------------------
# Start hand tracking
# -----------------------------

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Camera not detected")
            break

        # Mirror the camera
        frame = cv2.flip(frame, 1)

        # Get frame size
        h, w, _ = frame.shape

        # Create white canvas
        canvas = 255 * np.ones(
          (h, w, 3),
      dtype="uint8"
    )

        # Convert BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # Detect hand
        result = landmarker.detect(mp_image)

        # -----------------------------
        # If hand detected
        # -----------------------------

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # Index fingertip = landmark 8
            fingertip = hand[8]

            # Convert normalized coordinates → pixels
            x = int(fingertip.x * w)
            y = int(fingertip.y * h)

            # Draw red fingertip
            cv2.circle(
                frame,
                (x, y),
                10,
                (0, 0, 255),
                -1
            )

            # Draw line from previous position
            if prev_x is not None and prev_y is not None:

                cv2.line(
                    canvas,
                    (prev_x, prev_y),
                    (x, y),
                    (255, 0, 0),
                    5
                )

            # Update previous position
            prev_x = x
            prev_y = y

        else:

            # No hand → stop connecting old and new positions
            prev_x = None
            prev_y = None

        # Show webcam
        cv2.imshow("AirCanvas Camera", frame)

        # Show drawing
        cv2.imshow("AirCanvas Drawing", canvas)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()