# predict.py
import tensorflow as tf
import numpy as np
import cv2
import os

IMG_SIZE = 224
MODEL_PATH = "plant_disease_model.h5"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
labels = sorted(os.listdir("dataset"))  # same order as in training

# Prediction function
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index]

    return labels[class_index], confidence

# Example usage
if __name__ == "__main__":
    path = "image.png" 
    result, conf = predict_image(path)
    print(f"Disease: {result}\nConfidence: {conf*100:.2f}%")
