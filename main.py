import cv2
import mediapipe as mp
import numpy as np
import math

# ============================================================
# MEDIAPIPE SETUP
# ============================================================

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

# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

if not ret:
    print("Camera not detected")
    cap.release()
    exit()

frame = cv2.flip(frame, 1)

h, w, _ = frame.shape

# ============================================================
# CANVAS
# ============================================================

canvas = 255 * np.ones(
    (h, w, 3),
    dtype="uint8"
)

# ============================================================
# DRAWING VARIABLES
# ============================================================

prev_x = None
prev_y = None

# ============================================================
# SHAPE VARIABLES
# ============================================================

points = []
shape_mode = False
shape_name = ""

# ============================================================
# UNDO / REDO
# ============================================================

undo_stack = []
redo_stack = []

# ============================================================
# COLOR
# ============================================================

current_color = (255, 0, 0)

# ============================================================
# ERASER
# ============================================================

eraser_mode = False

# ============================================================
# COLORS
# ============================================================

colors = [
    ((0, 0, 255), "RED"),
    ((0, 255, 0), "GREEN"),
    ((255, 0, 0), "BLUE"),
    ((0, 255, 255), "YELLOW"),
    ((100, 100, 100), "ERASER"),
    ((0, 0, 0), "CLEAR")
]


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p2[0] - p1[0]) ** 2 +
        (p2[1] - p1[1]) ** 2
    )


# ============================================================
# IMPROVED SHAPE RECOGNITION
# ============================================================

def recognize_shape(points):

    # Need enough points
    if len(points) < 15:
        return "Unknown", None

    # Convert points to NumPy array
    pts = np.array(
        points,
        dtype=np.int32
    )

    # --------------------------------------------------------
    # Bounding rectangle
    # --------------------------------------------------------

    x, y, width, height = cv2.boundingRect(pts)

    # Shape should have reasonable size
    if width < 30 or height < 30:
        return "Unknown", None

    # --------------------------------------------------------
    # Check whether shape is closed
    # --------------------------------------------------------

    start = points[0]
    end = points[-1]

    closing_distance = distance(
        start,
        end
    )

    diagonal = math.sqrt(
        width * width +
        height * height
    )

    if closing_distance > diagonal * 0.35:
        return "Unknown", None

    # --------------------------------------------------------
    # Create contour
    # --------------------------------------------------------

    contour = pts.reshape(
        (-1, 1, 2)
    )

    # --------------------------------------------------------
    # Perimeter
    # --------------------------------------------------------

    perimeter = cv2.arcLength(
        contour,
        True
    )

    if perimeter == 0:
        return "Unknown", None

    # --------------------------------------------------------
    # Approximate polygon
    # --------------------------------------------------------

    epsilon = 0.04 * perimeter

    approx = cv2.approxPolyDP(
        contour,
        epsilon,
        True
    )

    corners = len(approx)

    # ========================================================
    # TRIANGLE
    # ========================================================

    if corners == 3:

        return "Triangle", approx

    # ========================================================
    # RECTANGLE
    # ========================================================

    if corners == 4:

        area = cv2.contourArea(
            contour
        )

        if area <= 0:
            return "Unknown", None

        bounding_area = width * height

        fill_ratio = (
            area / bounding_area
        )

        if fill_ratio > 0.45:

            return "Rectangle", approx

    # ========================================================
    # CIRCLE
    # ========================================================

    aspect_ratio = width / height

    if 0.75 <= aspect_ratio <= 1.25:

        area = cv2.contourArea(
            contour
        )

        if area > 0:

            circularity = (
                4 * math.pi * area
                / (perimeter * perimeter)
            )

            if circularity > 0.60:

                center_x = (
                    x + width // 2
                )

                center_y = (
                    y + height // 2
                )

                radius = (
                    width + height
                ) // 4

                return "Circle", (
                    (center_x, center_y),
                    radius
                )

    # ========================================================
    # UNKNOWN
    # ========================================================

    return "Unknown", None


# ============================================================
# DRAW RECOGNIZED SHAPE
# ============================================================

def draw_recognized_shape(
    canvas,
    shape_name,
    shape_data,
    color
):

    # --------------------------------------------------------
    # Circle
    # --------------------------------------------------------

    if shape_name == "Circle":

        center, radius = shape_data

        cv2.circle(
            canvas,
            center,
            radius,
            color,
            5
        )

    # --------------------------------------------------------
    # Rectangle
    # --------------------------------------------------------

    elif shape_name == "Rectangle":

        polygon = shape_data

        cv2.polylines(
            canvas,
            [polygon],
            True,
            color,
            5
        )

    # --------------------------------------------------------
    # Triangle
    # --------------------------------------------------------

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
# START MEDIAPIPE
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        # ----------------------------------------------------
        # READ CAMERA
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            print("Camera not detected")
            break

        # Mirror camera
        frame = cv2.flip(
            frame,
            1
        )

        # ====================================================
        # DRAW COLOR PALETTE
        # ====================================================

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

        # ====================================================
        # MEDIAPIPE IMAGE
        # ====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # ====================================================
        # DETECT HAND
        # ====================================================

        result = landmarker.detect(
            mp_image
        )

        # ====================================================
        # HAND FOUND
        # ====================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # ------------------------------------------------
            # INDEX FINGER UP
            # ------------------------------------------------

            index_up = (
                hand[8].y < hand[6].y
            )

            # ------------------------------------------------
            # FINGERTIP
            # ------------------------------------------------

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

                    # -----------------------------------------
                    # ERASER
                    # -----------------------------------------

                    if color_index == 4:

                        eraser_mode = True

                    # -----------------------------------------
                    # CLEAR
                    # -----------------------------------------

                    elif color_index == 5:

                        undo_stack.append(
                            canvas.copy()
                        )

                        redo_stack.clear()

                        canvas[:] = 255

                        points.clear()

                        shape_name = ""

                        eraser_mode = False

                    # -----------------------------------------
                    # NORMAL COLOR
                    # -----------------------------------------

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

                # ---------------------------------------------
                # NEW STROKE
                # ---------------------------------------------

                if (
                    prev_x is None
                    or prev_y is None
                ):

                    undo_stack.append(
                        canvas.copy()
                    )

                    redo_stack.clear()

                    points = []

                    shape_name = ""

                # ---------------------------------------------
                # STORE POINT
                # ---------------------------------------------

                points.append(
                    (x, y)
                )

                # ---------------------------------------------
                # STATUS
                # ---------------------------------------------

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

                # ---------------------------------------------
                # DRAW LINE
                # ---------------------------------------------

                if (
                    prev_x is not None
                    and prev_y is not None
                ):

                    if (
                        shape_mode
                        and not eraser_mode
                    ):

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

                # ---------------------------------------------
                # POINT COUNT
                # ---------------------------------------------

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

                # ---------------------------------------------
                # RECOGNIZE SHAPE
                # ---------------------------------------------

                if (
                    shape_mode
                    and len(points) >= 15
                    and not eraser_mode
                ):

                    recognized, data = (
                        recognize_shape(points)
                    )

                    # -----------------------------------------
                    # RECOGNIZED
                    # -----------------------------------------

                    if recognized != "Unknown":

                        # Restore canvas to
                        # state before rough shape

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

                    # -----------------------------------------
                    # UNKNOWN
                    # -----------------------------------------

                    else:

                        shape_name = "Unknown"

                # Reset drawing coordinates

                prev_x = None
                prev_y = None

            # =================================================
            # FINGERTIP DOT
            # =================================================

            cv2.circle(
                frame,
                (x, y),
                10,
                (0, 0, 0),
                -1
            )

        # ====================================================
        # NO HAND
        # ====================================================

        else:

            prev_x = None
            prev_y = None

        # ====================================================
        # SHOW DETECTED SHAPE
        # ====================================================

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

        # ====================================================
        # CURRENT COLOR
        # ====================================================

        cv2.putText(
            frame,
            "Current Color",
            (650, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

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

        # ====================================================
        # CONTROLS
        # ====================================================

        cv2.putText(
            frame,
            "D: Draw   S: Shape   Z: Undo   Y: Redo   Q: Quit",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2
        )

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "AirCanvas Camera",
            frame
        )

        cv2.imshow(
            "AirCanvas Drawing",
            canvas
        )

        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        # ----------------------------------------------------
        # D = DRAWING MODE
        # ----------------------------------------------------

        if key == ord("d"):

            shape_mode = False

            points.clear()

            shape_name = ""

        # ----------------------------------------------------
        # S = SHAPE MODE
        # ----------------------------------------------------

        elif key == ord("s"):

            shape_mode = True

            points.clear()

            shape_name = ""

        # ----------------------------------------------------
        # Z = UNDO
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Y = REDO
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Q = QUIT
        # ----------------------------------------------------

        elif key == ord("q"):

            break


# ============================================================
# RELEASE
# ============================================================

cap.release()

cv2.destroyAllWindows()