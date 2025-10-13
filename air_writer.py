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
GREEN = (0, 255, 0) # Defined for 'Saved' message

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
    
    # Ensure background_path_or_color is a string for comparison
    current_background = str(background_path_or_color or '').lower()
    
    # Default to a blank canvas
    canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    eraser_color = BLACK # Default eraser color

    if current_background == 'whiteboard':
        canvas[:] = WHITE
        eraser_color = WHITE
    elif current_background == 'blackboard':
        canvas[:] = BLACK
        eraser_color = BLACK
    elif os.path.exists(current_background):
        # Handle image/PDF file paths
        background_image = cv2.imread(current_background)
        if background_image is not None:
            canvas = cv2.resize(background_image, (WIDTH, HEIGHT))
            # Simple approximation for eraser color on custom backgrounds
            eraser_color = WHITE 
        else:
            # Fallback if image load fails
            canvas[:] = WHITE
            eraser_color = WHITE
    else:
        # Default canvas state (e.g., initial load with no arg)
        canvas[:] = WHITE
        eraser_color = WHITE

def get_colors():
    """Returns the available colors."""
    return COLORS

def main(background=None):
    """
    Main function to run the hand tracking and drawing application.
    Checks for camera and runs the loop or yields a static error frame.
    """
    global is_paused, canvas, eraser_color
    set_background(background or 'whiteboard') # Default to whiteboard
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    # Setup camera
    cap = cv2.VideoCapture(0)
    is_camera_available = cap.isOpened()
    
    if is_camera_available:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    else:
        print("Warning: Could not open camera. Running in static background mode.")
    
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
        
        # Check if drawing is paused
        if is_paused:
            yield frame 
            continue
            
        # --- Camera Handling and Gesture Detection (Only if camera is available) ---
        is_drawing = False
        
        if is_camera_available:
            success, cam_frame = cap.read()
            if not success:
                is_camera_available = False
                continue
            
            cam_frame = cv2.flip(cam_frame, 1) # Flip for selfie view
            frame = cam_frame.copy() # Use the live camera feed as the base frame

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
                        # Color selection
                        color_keys = list(COLORS.keys())
                        for i, key in enumerate(color_keys):
                            if 20 + i*100 < cx < 120 + i*100:
                                draw_color = COLORS[key]
                                break
                        
                        # Other actions (Eraser, Clear)
                        if 1060 < cx < 1160: # Eraser
                            draw_color = eraser_color
                        elif 1170 < cx < 1270: # Clear
                            undo_stack.append(canvas.copy())
                            set_background(current_background) # Re-apply background
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
                        points.appendleft(None) # Break the line segment

        # --- Drawing on Canvas ---
        for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                continue
            cv2.line(canvas, points[i - 1], points[i], draw_color, thickness)
            
        # --- Frame Combination ---
        if is_camera_available:
            # Overlay drawing onto the live camera frame
            gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
            
            drawing_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
            
            inv_mask = cv2.bitwise_not(mask)
            frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
            
            frame = cv2.add(frame_bg, drawing_fg)
        else:
            # If camera is NOT available, the frame IS the canvas.
            frame = canvas.copy()
            # Display camera error message
            cv2.putText(frame, "Camera Not Available (Server)", (250, HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # --- UI Overlay (Drawn on the final frame for visibility) ---
        slider_pos = slider_x + int((thickness / 50) * slider_w)
        
        # Draw color selection boxes
        color_keys = list(COLORS.keys())
        for i, key in enumerate(color_keys):
            x1, y1 = 20 + i*100, 10
            x2, y2 = 120 + i*100, 50
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[key], -1)
            cv2.putText(frame, key, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE if key not in ["YELLOW", "CYAN"] else BLACK, 2)
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
        
        # Show save message (if needed)
        if save_message_timer > 0:
            cv2.putText(frame, "Saved!", (550, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, GREEN, 3)
            save_message_timer -= 1
            
        yield frame

    if is_camera_available:
        cap.release()

# --- Functions exposed to app.py ---

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
