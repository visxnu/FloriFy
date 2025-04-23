import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load your trained model
model = load_model('flower_classifier.h5')

# Define class labels (order should match your training)
class_labels = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return class_labels[predicted_class].capitalize()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            prediction = predict_image(file_path)
            return render_template('index.html', prediction=prediction, image_path=file_path)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
