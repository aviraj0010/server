from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load your trained model
model = load_model('final5.h5')

def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust the size to match your model input size
    image = np.array(image) / 255.0
    if image.shape[-1] != 3:
        image = np.stack((image,) * 3, axis=-1)  # Add the channel dimension if it is missing
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        result = np.argmax(prediction, axis=1)[0]
        return jsonify({'prediction': int(result)})

if __name__ == '__main__':
    app.run(debug=True)
