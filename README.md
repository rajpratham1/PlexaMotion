# PlexaMotion Air Writer

Welcome to Air Writer, a web application that transforms your webcam into a virtual canvas! Use hand gestures to draw, annotate, and express your ideas in real-time.

## Features

- **Real-time Hand Tracking:** Draws on the screen by tracking your index finger.
- **Color Palette:** A palette of vibrant colors to choose from for your drawings.
- **Adjustable Brush Size:** An interactive slider to control the thickness of your brush strokes.
- **Eraser & Clear:** Easily erase parts of your drawing or clear the entire canvas with a single action.
- **Save Drawing:** Save your creations as a PNG file to your local disk.
- **Web-Based Interface:** Access the tool from your browser without any complex software installation.

## Tech Stack

- **Backend:** Python, Flask
- **Computer Vision:** OpenCV, MediaPipe
- **Frontend:** HTML, CSS, JavaScript

## Project Structure

```
.project-py/
├── app.py              # Main Flask application, handles routing and server logic
├── air_writer.py       # Core computer vision script for hand tracking and drawing
├── requirements.txt    # A list of all required Python packages
├── templates/
│   ├── index.html      # The main landing and instruction page
│   └── writer.html     # The page with the drawing interface
├── uploads/            # Default directory for saved drawings
├── venv311/            # Python virtual environment (to be created)
├── LICENSE             # MIT License file
└── README.md           # This file
```

---

## Setup and Installation

Follow these steps carefully to set up and run the project on your local machine.

### 1. Prerequisite: Install Python 3.11

This project depends on the `mediapipe` library, which is not yet compatible with Python versions newer than 3.11. You **must** have Python 3.11 installed.

- You can download Python 3.11 from the [official Python website](https://www.python.org/downloads/release/python-3119/).
- To check if you have the correct version installed, open your terminal or command prompt and run:
  ```sh
  python --version
  ```
  The output should be `Python 3.11.x`.

### 2. Create a Virtual Environment

Once Python 3.11 is installed, navigate to the project directory (`project py`) in your terminal and create a virtual environment. This will isolate the project's dependencies.

```sh
python -m venv venv311
```

This command creates the `venv311` folder in your project directory.

### 3. Activate the Virtual Environment

Before installing packages, you need to activate the environment you just created.

- **On Windows:**
  ```powershell
  .\venv311\Scripts\Activate.ps1
  ```
  *(If you are using Command Prompt (cmd.exe), the command is `venv311\\Scripts\\activate`)*

- **On macOS and Linux:**
  ```sh
  source venv311/bin/activate
  ```

After activation, you will see `(venv311)` at the beginning of your terminal prompt.

### 4. Install Dependencies

With the virtual environment active, install all the required libraries from the `requirements.txt` file.

```sh
pip install -r requirements.txt
```

---

## How to Run the Application

1.  **Ensure your virtual environment is active.** (You should see `(venv311)` in your prompt).
2.  Run the Flask application:
    ```sh
    python app.py
    ```
3.  You will see output in your terminal indicating that the server is running, something like:
    ```
     * Running on http://127.0.0.1:5000
    ```
4.  Open your web browser and navigate to **`http://127.0.0.1:5000`**.

## How to Use

1.  From the home page, click **"Start Writing"** to go to the virtual canvas. The application will request access to your webcam.
2.  **To Draw:** Raise your index finger while keeping your other fingers down. Move your hand to draw on the canvas.
3.  **To Select an Action:** Move your hand to the top of the screen and point your index finger at a color in the palette, the "ERASE" button, or the "CLEAR" button.
4.  **To Change Brush Size:** Point your index finger at the "Size" slider at the top right and move your hand left or right.
5.  **To Save:** Click the "Save" button. Your drawing will be saved as a PNG file in the `uploads` folder.
6.  **To Exit:** Click the "Exit" button. This will stop the camera and return you to the home page.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
*Copyright (c) 2025 PlexaMotion*
