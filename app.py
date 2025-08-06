"""
Flask web application for crop disease detection system.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import uuid
from werkzeug.utils import secure_filename
from model import CropDiseaseModel
from disease_database import get_disease_info
import json

app = Flask(__name__)
app.secret_key = 'crop_disease_detection_secret_key'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'mp4', 'avi', 'mov', 'mkv'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the production ML model
model = CropDiseaseModel()

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_video_file(filename):
    """Check if file is a video."""
    video_extensions = {'mp4', 'avi', 'mov', 'mkv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in video_extensions

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and disease detection."""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Process the file
                if is_video_file(filename):
                    # Process video
                    predictions = model.process_video(filepath)
                    if isinstance(predictions, dict) and 'error' in predictions:
                        flash(f'Error processing video: {predictions["error"]}')
                        return redirect(url_for('upload_file'))
                    
                    # Get most common prediction from video frames
                    disease_counts = {}
                    for pred in predictions:
                        if 'disease' in pred:
                            disease = pred['disease']
                            disease_counts[disease] = disease_counts.get(disease, 0) + 1
                    
                    if disease_counts:
                        most_common_disease = max(disease_counts, key=disease_counts.get)
                        avg_confidence = sum(p.get('confidence', 0) for p in predictions if 'confidence' in p) / len([p for p in predictions if 'confidence' in p])
                        
                        result = {
                            'disease': most_common_disease,
                            'confidence': avg_confidence,
                            'frame_count': len(predictions),
                            'type': 'video'
                        }
                    else:
                        flash('Could not detect disease in video')
                        return redirect(url_for('upload_file'))
                else:
                    # Process image
                    result = model.predict_disease(filepath)
                    if 'error' in result:
                        flash(f'Error processing image: {result["error"]}')
                        return redirect(url_for('upload_file'))
                    result['type'] = 'image'
                
                # Get disease information
                disease_info = get_disease_info(result['disease'])
                
                # Clean up uploaded file
                os.remove(filepath)
                
                return render_template('results.html', 
                                     prediction=result, 
                                     disease_info=disease_info,
                                     filename=file.filename)
                
            except Exception as e:
                # Clean up uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)
                flash(f'Error processing file: {str(e)}')
                return redirect(url_for('upload_file'))
        else:
            flash('Invalid file type. Please upload an image or video file.')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for disease prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save file temporarily
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        if is_video_file(filename):
            predictions = model.process_video(filepath)
            if isinstance(predictions, dict) and 'error' in predictions:
                return jsonify(predictions), 500
            
            # Get most common prediction
            disease_counts = {}
            for pred in predictions:
                if 'disease' in pred:
                    disease = pred['disease']
                    disease_counts[disease] = disease_counts.get(disease, 0) + 1
            
            if disease_counts:
                most_common_disease = max(disease_counts, key=disease_counts.get)
                avg_confidence = sum(p.get('confidence', 0) for p in predictions if 'confidence' in p) / len([p for p in predictions if 'confidence' in p])
                
                result = {
                    'disease': most_common_disease,
                    'confidence': avg_confidence,
                    'frame_count': len(predictions),
                    'type': 'video'
                }
            else:
                return jsonify({'error': 'Could not detect disease in video'}), 500
        else:
            result = model.predict_disease(filepath)
            if 'error' in result:
                return jsonify(result), 500
            result['type'] = 'image'
        
        # Get disease information
        disease_info = get_disease_info(result['disease'])
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'prediction': result,
            'disease_info': disease_info
        })
        
    except Exception as e:
        # Clean up
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """About page."""
    return render_template('about.html')

@app.route('/diseases')
def diseases():
    """Disease information page."""
    from disease_database import DISEASE_DATABASE
    return render_template('diseases.html', diseases=DISEASE_DATABASE)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash('File is too large. Maximum size is 16MB.')
    return redirect(url_for('upload_file'))

if __name__ == '__main__':
    print("Starting Crop Disease Detection System...")
    print("Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
