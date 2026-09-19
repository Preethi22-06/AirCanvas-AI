import cv2
import mediapipe as mp
import numpy as np
import math

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
# Shape points
# -----------------------------

points = []

# -----------------------------
# Shape mode
# -----------------------------

shape_mode = False

# -----------------------------
# Undo / Redo
# -----------------------------

undo_stack = []
redo_stack = []

# -----------------------------
# Current color
# -----------------------------

current_color = (255, 0, 0)

# -----------------------------
# Eraser
# -----------------------------

eraser_mode = False

# -----------------------------
# Last recognized shape
# -----------------------------

shape_name = ""

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


# ============================================================
# SHAPE RECOGNITION FUNCTIONS
# ============================================================

# -----------------------------
# Calculate distance
# -----------------------------

def distance(p1, p2):

    return math.sqrt(
        (p2[0] - p1[0]) ** 2 +
        (p2[1] - p1[1]) ** 2
    )


# -----------------------------
# Recognize shape
# -----------------------------

def recognize_shape(points):

    if len(points) < 10:
        return "Unknown", None

    # Convert points to NumPy array

    pts = np.array(
        points,
        dtype=np.int32
    )

    # -----------------------------
    # Approximate contour
    # -----------------------------

    perimeter = cv2.arcLength(
        pts.reshape((-1, 1, 2)),
        False
    )

    if perimeter == 0:
        return "Unknown", None

    epsilon = 0.04 * perimeter

    approx = cv2.approxPolyDP(
        pts.reshape((-1, 1, 2)),
        epsilon,
        False
    )

    # -----------------------------
    # Bounding rectangle
    # -----------------------------

    x, y, width, height = cv2.boundingRect(
        pts
    )

    if width == 0 or height == 0:
        return "Unknown", None

    # -----------------------------
    # Check if stroke is closed
    # -----------------------------

    start_point = points[0]
    end_point = points[-1]

    closing_distance = distance(
        start_point,
        end_point
    )

    # -----------------------------
    # Shape recognition
    # -----------------------------

    number_of_corners = len(approx)

    # -----------------------------
    # Triangle
    # -----------------------------

    if (
        number_of_corners == 3
        and closing_distance < max(width, height) * 0.35
    ):

        return "Triangle", approx

    # -----------------------------
    # Rectangle
    # -----------------------------

    if (
        number_of_corners == 4
        and closing_distance < max(width, height) * 0.35
    ):

        return "Rectangle", approx

    # -----------------------------
    # Circle
    # -----------------------------

    center_x = x + width // 2
    center_y = y + height // 2

    radius = (width + height) // 4

    if radius > 0:

        center_distances = []

        for point in points:

            d = distance(
                point,
                (center_x, center_y)
            )

            center_distances.append(d)

        average_radius = np.mean(
            center_distances
        )

        radius_error = np.mean(
            np.abs(
                np.array(center_distances)
                - average_radius
            )
        )

        # A circle should have
        # similar width and height

        aspect_ratio = width / height

        if (
            0.75 <= aspect_ratio <= 1.25
            and average_radius > 0
            and radius_error / average_radius < 0.30
            and closing_distance < max(width, height) * 0.35
        ):

            center = (
                center_x,
                center_y
            )

            return "Circle", (
                center,
                int(average_radius)
            )

    # -----------------------------
    # Unknown
    # -----------------------------

    return "Unknown", None


# -----------------------------
# Draw recognized shape
# -----------------------------

def draw_recognized_shape(
    canvas,
    shape_name,
    shape_data,
    color
):

    # -----------------------------
    # Circle
    # -----------------------------

    if shape_name == "Circle":

        center, radius = shape_data

        cv2.circle(
            canvas,
            center,
            radius,
            color,
            5
        )

    # -----------------------------
    # Rectangle
    # -----------------------------

    elif shape_name == "Rectangle":

        polygon = shape_data

        cv2.polylines(
            canvas,
            [polygon],
            True,
            color,
            5
        )

    # -----------------------------
    # Triangle
    # -----------------------------

    elif shape_name == "Triangle":

        polygon = shape_data

        cv2.polylines(
            canvas,
            [polygon],
            True,
            color,
            5
        )


# ============================================================
# START HAND TRACKING
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:

            print("Camera not detected")
            break

        # -----------------------------
        # Mirror camera
        # -----------------------------

        frame = cv2.flip(
            frame,
            1
        )

        # -----------------------------
        # Draw palette
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
        # Convert BGR to RGB
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

        result = landmarker.detect(
            mp_image
        )

        # ====================================================
        # HAND DETECTED
        # ====================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # -----------------------------
            # Index finger
            # -----------------------------

            index_up = (
                hand[8].y < hand[6].y
            )

            # -----------------------------
            # Fingertip
            # -----------------------------

            fingertip = hand[8]

            x = int(
                fingertip.x * w
            )

            y = int(
                fingertip.y * h
            )

            # Keep inside screen

            x = max(
                0,
                min(x, w - 1)
            )

            y = max(
                0,
                min(y, h - 1)
            )

            # =================================================
            # PALETTE SELECTION
            # =================================================

            if (
                y < box_height
                and index_up
            ):

                color_index = (
                    x // box_width
                )

                if (
                    0 <= color_index
                    < len(colors)
                ):

                    # -----------------------------
                    # Eraser
                    # -----------------------------

                    if color_index == 4:

                        eraser_mode = True

                    # -----------------------------
                    # Clear
                    # -----------------------------

                    elif color_index == 5:

                        undo_stack.append(
                            canvas.copy()
                        )

                        redo_stack.clear()

                        canvas[:] = 255

                        points.clear()

                        shape_name = ""

                        eraser_mode = False

                    # -----------------------------
                    # Normal color
                    # -----------------------------

                    else:

                        eraser_mode = False

                        current_color = (
                            colors[color_index][0]
                        )

                    prev_x = None
                    prev_y = None

            # =================================================
            # DRAWING
            # =================================================

            if (
                index_up
                and y > box_height
            ):

                # -----------------------------
                # Start of new stroke
                # -----------------------------

                if (
                    prev_x is None
                    or prev_y is None
                ):

                    undo_stack.append(
                        canvas.copy()
                    )

                    redo_stack.clear()

                    # Start collecting points

                    points = []

                    shape_name = ""

                # -----------------------------
                # Store point
                # -----------------------------

                points.append(
                    (x, y)
                )

                # -----------------------------
                # Display mode
                # -----------------------------

                if shape_mode:

                    cv2.putText(
                        frame,
                        "SHAPE MODE",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

                elif eraser_mode:

                    cv2.putText(
                        frame,
                        "ERASER",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (100, 100, 100),
                        2
                    )

                else:

                    cv2.putText(
                        frame,
                        "DRAWING MODE",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        current_color,
                        2
                    )

                # -----------------------------
                # Draw line
                # -----------------------------

                if (
                    prev_x is not None
                    and prev_y is not None
                ):

                    if (
                        shape_mode
                        and not eraser_mode
                    ):

                        # Draw temporary
                        # shape stroke

                        cv2.line(
                            canvas,
                            (prev_x, prev_y),
                            (x, y),
                            current_color,
                            5
                        )

                    elif eraser_mode:

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

                prev_x = x
                prev_y = y

                # -----------------------------
                # Point count
                # -----------------------------

                cv2.putText(
                    frame,
                    "POINTS: "
                    + str(len(points)),
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    2
                )

            # =================================================
            # STOP DRAWING
            # =================================================

            else:

                # -----------------------------
                # Shape recognition
                # -----------------------------

                if (
                    shape_mode
                    and len(points) >= 10
                ):

                    recognized, data = (
                        recognize_shape(points)
                    )

                    # -----------------------------
                    # Replace rough shape
                    # -----------------------------

                    if recognized != "Unknown":

                        # Restore canvas to
                        # before rough stroke

                        if len(undo_stack) > 0:

                            canvas = (
                                undo_stack[-1].copy()
                            )

                        # Draw clean shape

                        draw_recognized_shape(
                            canvas,
                            recognized,
                            data,
                            current_color
                        )

                        shape_name = (
                            recognized
                        )

                    else:

                        shape_name = "Unknown"

                # -----------------------------
                # Reset drawing
                # -----------------------------

                prev_x = None
                prev_y = None

            # -----------------------------
            # Fingertip indicator
            # -----------------------------

            cv2.circle(
                frame,
                (x, y),
                10,
                (0, 0, 0),
                -1
            )

        else:

            # -----------------------------
            # No hand
            # -----------------------------

            prev_x = None
            prev_y = None

        # ====================================================
        # STATUS
        # ====================================================

        # -----------------------------
        # Shape result
        # -----------------------------

        if shape_name != "":

            cv2.putText(
                frame,
                "Detected: "
                + shape_name,
                (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2
            )

        # -----------------------------
        # Current color
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
        # Color indicator
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
        # Instructions
        # -----------------------------

        cv2.putText(
            frame,
            "D: Draw   S: Shape   Z: Undo   Y: Redo   Q: Quit",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2
        )

        # -----------------------------
        # Show camera
        # -----------------------------

        cv2.imshow(
            "AirCanvas Camera",
            frame
        )

        # -----------------------------
        # Show canvas
        # -----------------------------

        cv2.imshow(
            "AirCanvas Drawing",
            canvas
        )

        # ====================================================
        # KEYBOARD CONTROLS
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        # -----------------------------
        # Drawing mode
        # -----------------------------

        if key == ord("d"):

            shape_mode = False

            points.clear()

            shape_name = ""

        # -----------------------------
        # Shape mode
        # -----------------------------

        elif key == ord("s"):

            shape_mode = True

            points.clear()

            shape_name = ""

        # -----------------------------
        # Undo
        # -----------------------------

        elif key == ord("z"):

            if len(undo_stack) > 0:

                redo_stack.append(
                    canvas.copy()
                )

                canvas = undo_stack.pop()

            prev_x = None
            prev_y = None

            points.clear()

            shape_name = ""

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

            points.clear()

            shape_name = ""

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