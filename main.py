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

# -----------------------------
# Undo / Redo history
# -----------------------------

undo_stack = []
redo_stack = []

# -----------------------------
# Current drawing color
# -----------------------------

current_color = (255, 0, 0)

# -----------------------------
# Eraser mode
# -----------------------------

eraser_mode = False

# -----------------------------
# Color palette
# -----------------------------

colors = [
    ((0, 0, 255), "RED"),
    ((0, 255, 0), "GREEN"),
    ((255, 0, 0), "BLUE"),
    ((0, 255, 255), "YELLOW"),
    ((100, 100, 100), "ERASER"),
    ((0, 0, 0), "CLEAR")
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
                (x1 + 5, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
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

            # -----------------------------
            # Index finger detection
            # -----------------------------

            index_up = hand[8].y < hand[6].y

            # -----------------------------
            # Fingertip position
            # -----------------------------

            fingertip = hand[8]

            x = int(fingertip.x * w)
            y = int(fingertip.y * h)

            # Keep fingertip inside screen

            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))

            # -----------------------------
            # Palette selection
            # -----------------------------

            if y < box_height and index_up:

                color_index = x // box_width

                if 0 <= color_index < len(colors):

                    # -----------------------------
                    # Eraser
                    # -----------------------------

                    if color_index == 4:

                        eraser_mode = True

                    # -----------------------------
                    # Clear
                    # -----------------------------

                    elif color_index == 5:

                        # Save current canvas
                        # before clearing

                        undo_stack.append(
                            canvas.copy()
                        )

                        # New action means
                        # redo history is cleared

                        redo_stack.clear()

                        # Clear canvas

                        canvas[:] = 255

                        eraser_mode = False

                    # -----------------------------
                    # Normal color
                    # -----------------------------

                    else:

                        eraser_mode = False

                        current_color = colors[
                            color_index
                        ][0]

                    # Stop drawing after
                    # selecting a tool

                    prev_x = None
                    prev_y = None

            # -----------------------------
            # Drawing
            # -----------------------------

            if index_up and y > box_height:

                # -----------------------------
                # New stroke
                # -----------------------------

                if prev_x is None or prev_y is None:

                    # Save canvas before
                    # starting the stroke

                    undo_stack.append(
                        canvas.copy()
                    )

                    # New drawing after undo
                    # removes redo history

                    redo_stack.clear()

                # -----------------------------
                # Show ERASER
                # -----------------------------

                if eraser_mode:

                    cv2.putText(
                        frame,
                        "ERASER",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (100, 100, 100),
                        2
                    )

                # -----------------------------
                # Show DRAWING
                # -----------------------------

                else:

                    cv2.putText(
                        frame,
                        "DRAWING",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        current_color,
                        2
                    )

                # -----------------------------
                # Draw line
                # -----------------------------

                if prev_x is not None and prev_y is not None:

                    if eraser_mode:

                        cv2.line(
                            canvas,
                            (prev_x, prev_y),
                            (x, y),
                            (255, 255, 255),
                            20
                        )

                    else:

                        cv2.line(
                            canvas,
                            (prev_x, prev_y),
                            (x, y),
                            current_color,
                            5
                        )

                # Update previous position

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

        else:

            # No hand detected

            prev_x = None
            prev_y = None

        # -----------------------------
        # Current color display
        # -----------------------------

        cv2.putText(
            frame,
            "Current Color",
            (650, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

        # -----------------------------
        # Show current color
        # -----------------------------

        if eraser_mode:

            cv2.rectangle(
                frame,
                (800, 10),
                (850, 55),
                (100, 100, 100),
                -1
            )

        else:

            cv2.rectangle(
                frame,
                (800, 10),
                (850, 55),
                current_color,
                -1
            )

        # -----------------------------
        # Show keyboard instructions
        # -----------------------------

        cv2.putText(
            frame,
            "Z: Undo   Y: Redo   Q: Quit",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
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
        # Keyboard controls
        # -----------------------------

        key = cv2.waitKey(1) & 0xFF

        # -----------------------------
        # Undo
        # -----------------------------

        if key == ord("z"):

            if len(undo_stack) > 0:

                redo_stack.append(
                    canvas.copy()
                )

                canvas = undo_stack.pop()

            prev_x = None
            prev_y = None

        # -----------------------------
        # Redo
        # -----------------------------

        elif key == ord("y"):

            if len(redo_stack) > 0:

                undo_stack.append(
                    canvas.copy()
                )

                canvas = redo_stack.pop()

            prev_x = None
            prev_y = None

        # -----------------------------
        # Quit
        # -----------------------------

        elif key == ord("q"):

            break

# -----------------------------
# Release resources
# -----------------------------

cap.release()
cv2.destroyAllWindows()