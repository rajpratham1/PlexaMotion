import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import math
import os

# Constants
WIDTH, HEIGHT = 1280, 720
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0) 
RED = (0, 0, 255) # Used for error message

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
eraser_color = BLACK 

def set_background(background_path_or_color):
    """Sets the background for the canvas and updates the eraser color."""
    global canvas, current_background, eraser_color
    
    current_background = str(background_path_or_color or '').lower()
    
    canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    eraser_color = BLACK 

    if current_background == 'whiteboard':
        canvas[:] = WHITE
        eraser_color = WHITE
    elif current_background == 'blackboard':
        canvas[:] = BLACK
        eraser_color = BLACK
    elif os.path.exists(current_background):
        background_image = cv2.imread(current_background)
        if background_image is not None:
            canvas = cv2.resize(background_image, (WIDTH, HEIGHT))
            eraser_color = WHITE 
        else:
            canvas[:] = WHITE
            eraser_color = WHITE
    else:
        canvas[:] = WHITE
        eraser_color = WHITE

def get_colors():
    """Returns the available colors."""
    return COLORS

def main(background=None):
    """
    Main function to run the hand tracking and drawing application.
    Implements a safety switch to bypass ML setup in cloud environments.
    """
    global is_paused, canvas, eraser_color
    set_background(background or 'whiteboard') 
    
    # --- SAFETY SWITCH CHECK ---
    # If this environment variable is set to '1' (e.g., on Render), 
    # we bypass camera/ML setup to save resources and prevent crashes.
    is_cloud_deploy = os.environ.get("IS_CLOUD_DEPLOY", "0") == "1"
    
    # Initialize variables
    cap = None
    hands = None
    mp_draw = None
    is_camera_available = False

    if not is_cloud_deploy:
        # Setup camera
        cap = cv2.VideoCapture(0)
        is_camera_available = cap.isOpened()
        
        if is_camera_available:
            # Initialize MediaPipe Hands only if camera is available
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
            mp_draw = mp.solutions.drawing_utils
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        else:
            print("Warning: Camera failed to open locally. Running static server mode.")
    else:
        print("Warning: Running in dedicated static server (cloud) mode.")
    
    # Points deque for storing drawing coordinates
    points = deque(maxlen=512)
    
    # Undo/Redo stacks
    undo_stack = deque(maxlen=10)
    redo_stack = deque(maxlen=10)
    
    # Default color and thickness
    draw_color = COLORS["GREEN"]
    thickness = 10
    
    save_message_timer = 0
    slider_x, slider_y, slider_w, slider_h = 900, 10, 200, 40

    while True:
        # Start the frame with the current canvas content
        frame = canvas.copy()
        
        if is_paused:
            yield frame 
            continue
            
        # --- Camera Handling and Gesture Detection (Only if camera/ML is ready) ---
        is_drawing = False
        
        if is_camera_available and cap and hands:
            success, cam_frame = cap.read()
            if not success:
                is_camera_available = False # Treat failure as non-available
                continue
            
            cam_frame = cv2.flip(cam_frame, 1)
            frame = cam_frame.copy() 

            rgb_frame = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    landmarks = hand_landmarks.landmark
                    index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                    
                    cx, cy = int(index_finger_tip.x * WIDTH), int(index_finger_tip.y * HEIGHT)
                    
                    # Selection/Control Zone Check (Top 60px)
                    if cy < 60:
                        color_keys = list(COLORS.keys())
                        for i, key in enumerate(color_keys):
                            if 20 + i*100 < cx < 120 + i*100:
                                draw_color = COLORS[key]
                                break
                        
                        if 1060 < cx < 1160: # Eraser
                            draw_color = eraser_color
                        elif 1170 < cx < 1270: # Clear
                            undo_stack.append(canvas.copy())
                            set_background(current_background) 
                            points = deque(maxlen=512)

                        # Brush thickness slider interaction
                        if slider_x < cx < slider_x + slider_w:
                            slider_pos = cx
                            thickness = int(((slider_pos - slider_x) / slider_w) * 49) + 1

                    # Drawing Gesture: index finger up, middle finger down
                    is_drawing = (index_finger_tip.y < index_pip.y and 
                                  middle_tip.y > middle_pip.y)

                    if is_drawing and cy > 60:
                        if len(points) > 0 and points[0] is None:
                            undo_stack.append(canvas.copy())
                            redo_stack.clear()
                        points.appendleft((cx, cy))
                    else:
                        points.appendleft(None) 
        
        # --- Drawing on Canvas ---
        for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                continue
            cv2.line(canvas, points[i - 1], points[i], draw_color, thickness)
            
        # --- Frame Combination ---
        if is_camera_available and cap:
            # Camera is available: overlay drawing onto the live camera frame
            gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
            
            drawing_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
            inv_mask = cv2.bitwise_not(mask)
            frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
            
            frame = cv2.add(frame_bg, drawing_fg)
        else:
            # Camera is NOT available (Server Mode): frame is the canvas
            frame = canvas.copy()
            # Display the helpful server message
            cv2.putText(frame, "Run Locally for Gesture Control", (250, HEIGHT // 2 - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, RED, 3)
            cv2.putText(frame, "(Server mode supports drawing on background only)", (200, HEIGHT // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)

        # --- UI Overlay (Drawn on the final frame) ---
        slider_pos = slider_x + int((thickness / 50) * slider_w)
        
        color_keys = list(COLORS.keys())
        for i, key in enumerate(color_keys):
            x1, y1 = 20 + i*100, 10
            x2, y2 = 120 + i*100, 50
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[key], -1)
            cv2.putText(frame, key, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE if key not in ["YELLOW", "CYAN"] else BLACK, 2)
            if draw_color == COLORS[key]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE, 3)

        cv2.rectangle(frame, (1060, 10), (1160, 50), WHITE, -1)
        cv2.putText(frame, "ERASE", (1065, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2)
        if draw_color == eraser_color:
            cv2.rectangle(frame, (1060, 10), (1160, 50), BLACK, 3)

        cv2.rectangle(frame, (1170, 10), (1270, 50), WHITE, -1)
        cv2.putText(frame, "CLEAR", (1175, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2)

        cv2.rectangle(frame, (slider_x, slider_y), (slider_x + slider_w, slider_y + slider_h), WHITE, -1)
        cv2.putText(frame, f"Size: {thickness}", (slider_x + 5, slider_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2)
        cv2.circle(frame, (slider_pos, slider_y + slider_h // 2), 15, BLACK, -1)
        
        if save_message_timer > 0:
            cv2.putText(frame, "Saved!", (550, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, GREEN, 3)
            save_message_timer -= 1
            
        yield frame

    if cap:
        cap.release()

# --- Functions exposed to app.py (Fixes ImportError) ---

def pause_drawing():
    """Pauses the drawing process."""
    global is_paused
    is_paused = True

def resume_drawing():
    """Resumes the drawing process."""
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
