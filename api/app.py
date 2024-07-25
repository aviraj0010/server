from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
import psutil
import logging

app = Flask(__name__)

# Load your trained model once at startup
model_path = os.path.join(os.path.dirname(__file__), 'model', 'final5.h5')
model = load_model(model_path)

def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust the size to match your model input size
    image = np.array(image) / 255.0
    if image.shape[-1] != 3:
        image = np.stack((image,) * 3, axis=-1)  # Add the channel dimension if it is missing
    image = np.expand_dims(image, axis=0)
    return image

def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    app.logger.info(f"Memory Usage: RSS={memory_info.rss / 1024 ** 2:.2f} MB")

@app.route('/api/predict', methods=['POST'])
def predict():
    log_memory_usage()
    if 'file' not in request.files:
        app.logger.error("No file part")
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        app.logger.error("No selected file")
        return jsonify({'error': 'No selected file'})
    if file:
        try:
            image = Image.open(io.BytesIO(file.read()))
            app.logger.info("Image received, processing...")
            processed_image = preprocess_image(image)
            app.logger.info("Image processed, making prediction...")
            prediction = model.predict(processed_image)
            result = np.argmax(prediction, axis=1)[0]
            app.logger.info(f"Prediction result: {result}")
            log_memory_usage()
            return jsonify({'prediction': int(result)})
        except Exception as e:
            app.logger.error(f"Error: {e}")
            return jsonify({'error': str(e)})

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({'message': 'API is working'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
