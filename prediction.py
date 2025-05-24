"""
Updated prediction routes with fixed Grad-CAM implementation
"""
from flask import Blueprint, request, jsonify, current_app, session
import os
import uuid
import numpy as np
import cv2
from datetime import datetime
from werkzeug.utils import secure_filename
import base64
import io
from PIL import Image

from src.models.cnn_model import FruitQualityClassifier
from src.models.gradcam import generate_gradcam
from src.routes.auth import login_required, add_to_history

# Create blueprint
prediction_bp = Blueprint('prediction', __name__)

# Initialize classifier
classifier = None

def get_classifier():
    """Get or initialize the classifier"""
    global classifier
    if classifier is None:
        classifier = FruitQualityClassifier()
        model_path = os.path.join(current_app.root_path, 'models', 'fruit_quality_model.h5')
        if os.path.exists(model_path):
            classifier.load_model(model_path)
        else:
            # If no model exists, build a default one
            classifier.build_model()
    return classifier

def allowed_file(filename):
    """Check if file has an allowed extension"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@prediction_bp.route('/predict', methods=['POST'])
@login_required
def predict():
    """Predict fruit quality from uploaded image"""
    # Check if the post request has the file part
    if 'file' not in request.files and 'image_data' not in request.json:
        return jsonify({'error': 'No file or image data provided'}), 400
    
    # Create upload directory if it doesn't exist
    upload_dir = os.path.join(current_app.root_path, 'static', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    image_path = None
    
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            # Save the file with a secure filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            image_path = os.path.join(upload_dir, unique_filename)
            file.save(image_path)
    else:
        # Handle base64 encoded image data
        try:
            image_data = request.json.get('image_data', '')
            if image_data.startswith('data:image'):
                # Remove the data URL prefix
                image_data = image_data.split(',')[1]
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Save the image
            unique_filename = f"{uuid.uuid4()}.jpg"
            image_path = os.path.join(upload_dir, unique_filename)
            image.save(image_path)
        except Exception as e:
            return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
    
    # Get the classifier
    clf = get_classifier()
    
    try:
        # Make prediction
        result = clf.predict(image_path)
        
        # Generate Grad-CAM visualization using the fixed implementation
        try:
            # Preprocess the image for the model
            processed_image = clf.preprocess_image(image_path)
            
            # Generate Grad-CAM using the fixed implementation
            _, heatmap = generate_gradcam(clf.model, processed_image, image_path)
            
            # Save heatmap
            heatmap_filename = f"heatmap_{unique_filename}"
            heatmap_path = os.path.join(upload_dir, heatmap_filename)
            cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
            
            # Add heatmap path to result
            result['heatmap_url'] = f"/static/uploads/{heatmap_filename}"
        except Exception as e:
            current_app.logger.error(f"Error generating heatmap: {str(e)}")
            result['heatmap_url'] = None
        
        # Add image URL to result
        result['image_url'] = f"/static/uploads/{unique_filename}"
        
        # Add timestamp
        result['timestamp'] = datetime.now().isoformat()
        
        # Add to user history
        username = session.get('user_id')
        if username:
            add_to_history(username, result)
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@prediction_bp.route('/webcam_predict', methods=['POST'])
@login_required
def webcam_predict():
    """Predict fruit quality from webcam image"""
    # This is just a wrapper around the predict endpoint
    # The frontend will handle capturing the webcam image and sending it as base64
    return predict()
