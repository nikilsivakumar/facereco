from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load the pre-trained model from Google Cloud Storage (you need to replace with your bucket path)
#MODEL_PATH = "gs://facial-recognition-bucket-nk/fine_tuned_facial_recognition_model.h5"
MODEL_PATH = "C:\\Users\\nikil\\Documents\\Project\\DL Projects\\FaceRec1\\fine_tuned_facial_recognition_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Function to load and preprocess the image for the model
def load_image_from_file(image_path):
    """Load image, convert to grayscale, resize, and normalize it."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
    image = cv2.resize(image, (50, 50))  # Resize the image to 50x50
    image = np.repeat(image[..., np.newaxis], 3, axis=-1)  # Convert grayscale to 3 channels
    image = image / 255.0  # Normalize to [0, 1]
    return image.reshape(1, 50, 50, 3)  # Reshape the image to fit the model

# Function to match faces using the model
def match_faces(id_image, webcam_image):
    """Compare two images and return the similarity score."""
    # Normalize images
    id_image = id_image / 255.0
    webcam_image = webcam_image / 255.0
    
    # Predict embeddings (features) for both images
    id_embedding = model.predict(id_image)
    webcam_embedding = model.predict(webcam_image)
    
    # Calculate cosine similarity between the embeddings
    similarity = np.dot(id_embedding, webcam_embedding.T) / (
        np.linalg.norm(id_embedding) * np.linalg.norm(webcam_embedding)
    )
    
    return similarity.item() * 100  # Return percentage match

# Endpoint to verify the match between the ID and webcam images
@app.route('/verify', methods=['POST'])
def verify():
    """Endpoint for verifying faces from the uploaded ID and webcam images."""
    # Ensure both files are included in the request
    if 'id_image' not in request.files or 'test_image' not in request.files:
        return jsonify({"error": "Please upload both 'id_image' and 'test_image'."}), 400
    
    # Save the uploaded ID image
    id_file = request.files['id_image']
    id_image_path = 'id_image.jpg'
    id_file.save(id_image_path)
    
    # Save the uploaded test image
    test_file = request.files['test_image']
    test_image_path = 'test_image.jpg'
    test_file.save(test_image_path)
    
    # Load and preprocess both images
    id_image = load_image_from_file(id_image_path)
    webcam_image = load_image_from_file(test_image_path)
    
    # Calculate the match percentage
    match_percentage = match_faces(id_image, webcam_image)
    
    # Return the result
    if match_percentage > 85:
        result = "Match"
    else:
        result = "No Match"
    
    return jsonify({
        'match_percentage': match_percentage,
        'result': result
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
