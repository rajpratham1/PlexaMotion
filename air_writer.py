import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import datetime
import math
import os

# Constants
WIDTH, HEIGHT = 1280, 720
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# --- Color Palette ---
COLORS = {
    "GREEN": (0, 255, 0),
    "BLUE": (255, 0, 0),
    "RED": (0, 0, 255),
    "YELLOW": (0, 255, 255),
    "ORANGE": (0, 165, 255),
    "PURPLE": (128, 0, 128),
    "PINK": (203, 192, 255),
    "CYAN": (255, 255, 0),
}

# Global state
is_paused = False
current_background = None
canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

def set_background(background_path_or_color):
    """Sets the background for the canvas."""
    global canvas, current_background
    current_background = background_path_or_color
    if os.path.exists(str(current_background)):
        background_image = cv2.imread(str(current_background))
        if background_image is not None:
            canvas = cv2.resize(background_image, (WIDTH, HEIGHT))
        else:
            canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
            if str(current_background).lower() == 'whiteboard':
                canvas[:] = WHITE
            elif str(current_background).lower() == 'blackboard':
                canvas[:] = BLACK
    elif isinstance(current_background, str):
        if current_background.lower() == 'whiteboard':
            canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
            canvas[:] = WHITE
        elif current_background.lower() == 'blackboard':
            canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
            canvas[:] = BLACK
    else:
        canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

def get_colors():
    """Returns the available colors."""
    return COLORS

def main(background=None):
    """
    Main function to run the hand tracking and drawing application.
    """
    global is_paused, canvas
    set_background(background)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    # Setup camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    # Points deque for storing drawing coordinates
    points = deque(maxlen=512)
    
    # Undo/Redo stacks
    undo_stack = deque(maxlen=10)
    redo_stack = deque(maxlen=10)
    
    # Default color and thickness
    draw_color = COLORS["GREEN"]
    thickness = 10
    
    # Eraser
    eraser_color = BLACK

    save_message_timer = 0

    # Brush thickness slider
    slider_x, slider_y, slider_w, slider_h = 900, 10, 200, 40
    slider_pos = slider_x + int((thickness / 50) * slider_w)

    while True:
        if is_paused:
            yield canvas
            continue

        # Read frame from camera
        success, frame = cap.read()
        if not success:
            print("Error: Failed to capture frame.")
            break # Exit loop if frame capture fails

        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and find hands
        result = hands.process(rgb_frame)

        # Draw the hand annotations on the image.
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get coordinates of landmarks
                landmarks = hand_landmarks.landmark
                
                # Get coordinates of index finger tip and thumb tip
                index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                
                cx, cy = int(index_finger_tip.x * WIDTH), int(index_finger_tip.y * HEIGHT)
                tx, ty = int(thumb_tip.x * WIDTH), int(thumb_tip.y * HEIGHT)

                # Calculate distance between index finger and thumb
                pinch_distance = math.hypot(cx - tx, cy - ty)

                # Check if the user is selecting a color or action
                if cy < 60:
                    # Color selection
                    color_keys = list(COLORS.keys())
                    for i, key in enumerate(color_keys):
                        if 20 + i*100 < cx < 120 + i*100:
                            draw_color = COLORS[key]
                            break
                    
                    # Other actions
                    if 1060 < cx < 1160: # Eraser
                        draw_color = eraser_color
                    elif 1170 < cx < 1270: # Clear
                        undo_stack.append(canvas.copy())
                        canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
                        if current_background:
                            set_background(current_background)
                        points = deque(maxlen=512)

                    # Brush thickness slider interaction
                    if slider_x < cx < slider_x + slider_w:
                        slider_pos = cx
                        thickness = int(((slider_pos - slider_x) / slider_w) * 49) + 1

                # Gesture for drawing (index finger extended, others curled)
                index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
                ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
                pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
                pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP]

                # A more relaxed drawing gesture: index finger up, middle finger down.
                is_drawing = (index_tip.y < index_pip.y and
                              middle_tip.y > middle_pip.y)

                if is_drawing and cy > 60:
                    if len(points) > 0 and points[0] is None:
                        undo_stack.append(canvas.copy())
                        redo_stack.clear()
                    points.appendleft((cx, cy))
                else:
                    points.appendleft(None) # Add None to break the line

        # Draw on the canvas
        for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                continue
            cv2.line(canvas, points[i - 1], points[i], draw_color, thickness)

        # --- UI ---
        # Draw color selection boxes
        color_keys = list(COLORS.keys())
        for i, key in enumerate(color_keys):
            x1, y1 = 20 + i*100, 10
            x2, y2 = 120 + i*100, 50
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[key], -1)
            cv2.putText(frame, key, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE if key != "YELLOW" and key != "CYAN" else BLACK, 2)
            # Highlight selected color
            if draw_color == COLORS[key]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE, 3)

        # Draw eraser button
        cv2.rectangle(frame, (1060, 10), (1160, 50), WHITE, -1)
        cv2.putText(frame, "ERASE", (1065, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2)
        if draw_color == eraser_color:
            cv2.rectangle(frame, (1060, 10), (1160, 50), BLACK, 3)

        # Draw Clear button
        cv2.rectangle(frame, (1170, 10), (1270, 50), WHITE, -1)
        cv2.putText(frame, "CLEAR", (1175, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2)

        # Draw brush thickness slider
        cv2.rectangle(frame, (slider_x, slider_y), (slider_x + slider_w, slider_y + slider_h), WHITE, -1)
        cv2.putText(frame, f"Size: {thickness}", (slider_x + 5, slider_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2)
        cv2.circle(frame, (slider_pos, slider_y + slider_h // 2), 15, BLACK, -1)

        # Show save message
        if save_message_timer > 0:
            cv2.putText(frame, "Saved!", (550, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, GREEN, 3)
            save_message_timer -= 1

        # Combine frame and canvas
        # Create a mask of the canvas where it is not black
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(mask)
        
        # Black out the area of the drawing on the frame
        frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
        
        # Take only region of drawing from canvas
        drawing_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
        
        # Put drawing in ROI and add it to the main frame
        frame = cv2.add(frame_bg, drawing_fg)

        yield frame

    # Release resources
    cap.release()

def pause_drawing():
    global is_paused
    is_paused = True

def resume_drawing():
    global is_paused
    is_paused = False

if __name__ == '__main__':
    air_writer_generator = main()
    while True:
        try:
            frame = next(air_writer_generator)
            cv2.imshow("Air Writer", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                pause_drawing()
            elif key == ord('r'):
                resume_drawing()
        except StopIteration:
            break
    cv2.destroyAllWindows()