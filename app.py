from flask import Flask, render_template, Response, jsonify
import cv2
import os
from air_writer import main as air_writer_main, get_colors, save_canvas, release_camera

app = Flask(__name__)

# Initialize the generator
air_writer_generator = None

def generate_frames():
    """Generator function to yield frames for the video stream."""
    global air_writer_generator
    if air_writer_generator is None:
        air_writer_generator = air_writer_main()

    while True:
        try:
            frame = next(air_writer_generator)
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            # Yield the frame in the byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except StopIteration:
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/writer')
def writer():
    return render_template('writer.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/colors')
def colors():
    return jsonify(get_colors())

@app.route('/stop_video')
def stop_video():
    release_camera()
    return "stopped"

if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible from the network
    app.run(host='0.0.0.0', port=5000, debug=True)
