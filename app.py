from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import fitz # PyMuPDF
# Import fixed functions
from air_writer import main as air_writer_main, pause_drawing, resume_drawing, set_background as set_air_writer_background, get_colors

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global state to persist the background setting across requests
current_bg_setting = 'whiteboard' 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/writer')
def writer():
    return render_template('writer.html')

def generate_frames():
    """Generator function to yield frames from the air_writer.py script."""
    # Pass the current global background setting
    for frame in air_writer_main(background=current_bg_setting):
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_bg_setting
    if 'file' not in request.files:
        return jsonify(error='No file part'), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error='No selected file'), 400
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        final_filepath = filepath

        if filename.lower().endswith('.pdf'):
            # Convert first page of PDF to an image
            try:
                pdf_document = fitz.open(filepath)
                page = pdf_document.load_page(0)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
                output_filename = f"{os.path.splitext(filename)[0]}.png"
                output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
                pix.save(output_filepath)
                pdf_document.close()
                final_filepath = output_filepath
            except Exception as e:
                print(f"PDF conversion failed: {e}")
                return jsonify(error='PDF conversion failed'), 500
        
        # Update the global state and the background for the generator
        current_bg_setting = final_filepath
        set_air_writer_background(final_filepath)
        
        return jsonify(filepath=final_filepath)

@app.route('/background', methods=['POST'])
def set_background():
    global current_bg_setting
    data = request.get_json()
    background = data.get('background')
    
    # Update the global state and the background for the generator
    current_bg_setting = background
    set_air_writer_background(background)
    
    return jsonify(status="background updated")

@app.route('/colors')
def colors():
    return jsonify(get_colors())

@app.route('/pause', methods=['POST'])
def pause():
    pause_drawing()
    return jsonify(status="paused")

@app.route('/resume', methods=['POST'])
def resume():
    resume_drawing()
    return jsonify(status="resumed")

if __name__ == '__main__':
    # Initialize the background upon startup
    set_air_writer_background(current_bg_setting) 
    app.run(debug=True) # This line will only run during local development/testing.
