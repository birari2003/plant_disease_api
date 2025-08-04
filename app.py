# app.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS


# Load your trained model
model = tf.keras.models.load_model("plant_disease_model.h5")
IMG_SIZE = 224
labels = sorted(os.listdir("dataset"))

# Create Flask app
app = Flask(__name__)
CORS(app) 
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index]
    return labels[class_index], float(confidence)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    disease, confidence = predict_image(filepath)

    remedy = {
        "Tomato___Late_blight": "Remove infected leaves. Use copper fungicides.",
        "Tomato___Leaf_Mold": "Improve air circulation and use sulfur-based fungicides.",
        "Potato___Early_blight": "Apply neem oil. Rotate crops and avoid overhead watering.",
        # Add more as needed
    }.get(disease, "No remedy found.")

    return jsonify({
        'disease': disease,
        'confidence': f"{confidence * 100:.2f}%",
        'remedy': remedy
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
