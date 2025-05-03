from flask import Flask, request, render_template, jsonify
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import os
import base64
import cv2
import numpy as np
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Set up file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load models
def get_model(model_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "ocr":
        model = VisionEncoderDecoderModel.from_pretrained("../models/trocr_license_plate").to(device)
        processor = TrOCRProcessor.from_pretrained("../models/trocr_license_plate")
        return model, processor, device
    elif model_type == "yolo":
        model = YOLO("../models/best.pt").to(device)
        return model, None, device
    return None, None, None

# Process image for OCR (Text Extraction)
def process_ocr(image_path):
    model, processor, device = get_model("ocr")
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    
    # Generate text prediction
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

    # Ensure we return a valid string
    return generated_text[0] if generated_text else "No text detected"

# Process image for YOLO-based License Plate Detection (with Bounding Boxes)
def process_yolo(image_path):
    model, _, device = get_model("yolo")
    
    # Read image
    image = cv2.imread(image_path)
    
    # Perform object detection
    results = model(image)
    
    detected = False  # Flag to check if detection occurred
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = map(int, box)
            detected = True

            # Draw bounding box on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (155, 89, 182), 3)  # Purple for YOLO
            cv2.putText(image, f"Plate {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (155, 89, 182), 2)

    # Convert processed image to Base64
    _, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return img_str if detected else None, "High" if detected else None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    model_type = request.form.get('model_type', 'ocr')  # Default to OCR if not provided

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            if model_type == "ocr":
                result = process_ocr(file_path)
                confidence = "High"
            elif model_type == "yolo":
                result, confidence = process_yolo(file_path)
                if not result:
                    return jsonify({'error': 'No license plate detected'})

            # Convert original image to Base64 for display
            with open(file_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')

            return jsonify({
                'success': True,
                'model_used': model_type,
                'license_plate': result if model_type == "ocr" else "License Plate Detected",
                'confidence': confidence,
                'image': f'data:image/{file.filename.split(".")[-1]};base64,{img_data}',
                'plate_image': f'data:image/jpeg;base64,{result}' if model_type == "yolo" else None
            })
        except Exception as e:
            return jsonify({'error': str(e)})

    return jsonify({'error': 'File type not allowed'})

if __name__ == '__main__':
    app.run(debug=True)
