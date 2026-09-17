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
# Read first frame
# -----------------------------

ret, frame = cap.read()

if not ret:
    print("Camera not detected")
    cap.release()
    exit()

frame = cv2.flip(frame, 1)

h, w, _ = frame.shape

# -----------------------------
# White drawing canvas
# -----------------------------

canvas = 255 * np.ones(
    (h, w, 3),
    dtype="uint8"
)

# -----------------------------
# Drawing variables
# -----------------------------

prev_x = None
prev_y = None

# Current drawing color
current_color = (255, 0, 0)

# -----------------------------
# Color palette
# -----------------------------

colors = [
    ((0, 0, 255), "RED"),
    ((0, 255, 0), "GREEN"),
    ((255, 0, 0), "BLUE"),
    ((0, 255, 255), "YELLOW")
]

# -----------------------------
# Start hand tracking
# -----------------------------

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Camera not detected")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # -----------------------------
        # Draw color palette
        # -----------------------------

        box_width = 100
        box_height = 70

        for i, (color, name) in enumerate(colors):

            x1 = i * box_width
            y1 = 0

            x2 = x1 + box_width
            y2 = box_height

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                -1
            )

            cv2.putText(
                frame,
                name,
                (x1 + 10, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

        # -----------------------------
        # Convert BGR → RGB
        # -----------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # -----------------------------
        # Detect hand
        # -----------------------------

        result = landmarker.detect(mp_image)

        # -----------------------------
        # Hand detected
        # -----------------------------

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # Index finger
            index_up = hand[8].y < hand[6].y

            fingertip = hand[8]

            x = int(fingertip.x * w)
            y = int(fingertip.y * h)

            # -----------------------------
            # Color selection
            # -----------------------------

            if y < box_height:

                color_index = x // box_width

                if 0 <= color_index < len(colors):

                    current_color = colors[color_index][0]

                    prev_x = None
                    prev_y = None

            # -----------------------------
            # Draw fingertip
            # -----------------------------

            cv2.circle(
                frame,
                (x, y),
                10,
                (0, 0, 0),
                -1
            )

            # -----------------------------
            # Drawing
            # -----------------------------

            if index_up and y > box_height:

                cv2.putText(
                    frame,
                    "DRAWING",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    current_color,
                    2
                )

                if prev_x is not None and prev_y is not None:

                    cv2.line(
                        canvas,
                        (prev_x, prev_y),
                        (x, y),
                        current_color,
                        5
                    )

                prev_x = x
                prev_y = y

            else:

                cv2.putText(
                    frame,
                    "NOT DRAWING",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2
                )

                prev_x = None
                prev_y = None

        else:

            prev_x = None
            prev_y = None

        # -----------------------------
        # Show current color
        # -----------------------------

        cv2.putText(
            frame,
            "Current Color",
            (450, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

        cv2.rectangle(
            frame,
            (600, 10),
            (680, 55),
            current_color,
            -1
        )

        # -----------------------------
        # Show windows
        # -----------------------------

        cv2.imshow(
            "AirCanvas Camera",
            frame
        )

        cv2.imshow(
            "AirCanvas Drawing",
            canvas
        )

        # -----------------------------
        # Press Q to quit
        # -----------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()