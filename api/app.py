from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load your trained model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'final5.h5')
model = load_model(model_path)
logging.info("Model loaded successfully.")

def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust the size to match your model input size
    image = np.array(image) / 255.0
    if image.shape[-1] != 3:
        image = np.stack((image,) * 3, axis=-1)  # Add the channel dimension if it is missing
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        logging.error("No file part in the request.")
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        logging.error("No selected file.")
        return jsonify({'error': 'No selected file'})
    if file:
        try:
            image = Image.open(io.BytesIO(file.read()))
            logging.info("Image received, processing...")
            processed_image = preprocess_image(image)
            logging.info("Image processed, making prediction...")
            prediction = model.predict(processed_image)
            result = np.argmax(prediction, axis=1)[0]
            logging.info(f"Prediction result: {result}")
            return jsonify({'prediction': int(result)})
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return jsonify({'error': str(e)})

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({'message': 'API is working'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
