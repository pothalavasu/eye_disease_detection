import numpy as np
import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# Load the trained model
model = load_model("evgg.h5")

# Initialize Flask app
app = Flask(__name__)

# Render Home Page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template("index.html")

@app.route('/inp')
def inp():
    return render_template("img_input.html")

# Image Prediction Route
@app.route('/predict', methods=["POST"])
def res():
    if request.method == "POST":
        f = request.files['image']
        
        # Save uploaded file
        basepath = os.path.dirname(__file__)  
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)

        # Load & Preprocess Image
        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)

        # Model Prediction
        prediction = np.argmax(model.predict(img_data), axis=1)
        index = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
        result = str(index[prediction[0]])

        print(result)
        return render_template('output.html', prediction=result)

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)