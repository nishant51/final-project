from flask import Flask, render_template, request, jsonify 
from flask import  send_file
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io
 
app = Flask(__name__)
 
# Load the trained model
# Load the trained model
model_path = 'inception_epoch10_batch16_accuracy92.h5'
model = load_model(model_path)

# Route to serve the model file
@app.route('/model')
def download_model():
    return send_file(model_path, as_attachment=True) 

# Define a function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # rescale pixel values
    return img_array
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/classify', methods=['POST'])
def classify():
    # Check if image data is present in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
 
    # Read the image from the request
    img_data = request.files['image'].read()
    img = Image.open(io.BytesIO(img_data))
 
    # Preprocess the image
    img_array = preprocess_image(img)
 
    # Make predictions
    predictions = model.predict(img_array)
    predictions = np.array(predictions)
    predicted_class = np.argmax(predictions)   
    # Map predicted class index to class name
    class_names = ['Basmati', 'Invalid-Rice', 'Jeera Maseeno', 'Mansuli', 'Pokhrali']
    predicted_class_name = class_names[predicted_class]
   
    return jsonify({'predicted_class': predicted_class_name})
 
 
if __name__ == '__main__':
    app.run(debug=True)