# PlexaMotion Air Writer

Welcome to Air Writer, a web application that transforms your webcam into a virtual canvas! Use hand gestures to draw, annotate, and express your ideas in real-time.

## Features

- **Real-time Hand Tracking:** Draws on the screen by tracking your index finger.
- **Multiple Colors:** A palette of colors to choose from.
- **Adjustable Brush Size:** A slider to control the thickness of your brush.
- **Eraser & Clear:** Easily erase parts of your drawing or clear the entire canvas.
- **Custom Backgrounds:** 
    - Draw on a classic whiteboard or blackboard.
    - Upload your own images (JPG, PNG) or single-page PDFs to use as a background.

## Tech Stack

- **Backend:** Python, Flask
- **Computer Vision:** OpenCV, MediaPipe
- **Frontend:** HTML, CSS, JavaScript

## Project Structure

```
.project-py/
├── app.py              # Main Flask application, handles routing and server logic
├── air_writer.py       # Core computer vision script for hand tracking and drawing
├── templates/
│   ├── index.html      # The main landing and instruction page
│   └── writer.html     # The page with the drawing interface
├── uploads/            # Default directory for user-uploaded backgrounds
├── venv311/            # Python virtual environment
├── LICENSE             # MIT License file
└── README.md           # This file
```

## Setup and Installation

To get the project running on your local machine, follow these steps:

1.  **Activate Virtual Environment:**
    Open your terminal in the project directory and activate the virtual environment.
    ```sh
    venv311\Scripts\activate
    ```

2.  **Install Dependencies:**
    Make sure you have the required Python libraries installed. If not, you can install them using pip:
    ```sh
    pip install Flask opencv-python mediapipe PyMuPDF
    ```

3.  **Run the Application:**
    ```sh
    python app.py
    ```

4.  **Open in Browser:**
    Navigate to `http://127.0.0.1:5000` in your web browser.

## How to Use

Once you click "Start Writing", the application will activate your webcam.

- **Select Action:** Move your hand to the top of the screen and point your index finger at a color or button (Erase, Clear).
- **Draw:** Keep your middle finger down while raising your index finger to draw.
- **Change Brush Size:** Point your index finger at the "Size" slider at the top right to adjust the thickness.
- **Upload Background:** Use the controls at the bottom of the writer page to upload an image/PDF or switch to a whiteboard/blackboard.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
*Copyright (c) 2025 PlexaMotion*
