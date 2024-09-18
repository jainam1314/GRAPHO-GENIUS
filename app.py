from flask import Flask, render_template, request, flash
from flask.helpers import url_for
from werkzeug.utils import redirect, secure_filename
import json
import os
import numpy as np
from keras.models import load_model
from PIL import Image
import cv2
from scripts import test

with open('config.json', 'r') as c:
    params  = json.load(c)["params"]
app = Flask(__name__)
app.config['SECRET_KEY'] = "dcc0e3e40867ccaa0d9afc35"
app.config['UPLOAD_FOLDER'] = params['upload_location']

# Load your trained model
model = load_model("type_identification_A.h5")
# model = load_model("type_identification_G.h5")
# model = load_model("type_identification_B.h5")
# model = load_model("type_identification_C.h5")
# model = load_model("type_identification_D.h5")
# model = load_model("type_identification_E.h5")
# model = load_model("type_identification_F.h5")
# model = load_model("type_identification_G.h5")
# model = load_model("type_identification_H.h5")
# model = load_model("type_identification_I.h5")


# Set the upload folder and allowed file extensions
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Define class labels for categories
class_labels = ['type1', 'type2', 'type3']

# Define traints for associated categories
traits_mapping = {
    'type1': "Feeling of lassitude may arise when discussing topics related to home (not just house). Individuals of this type tend to experience varying levels of energy depletion or weariness in such conversations. They might find it challenging to maintain enthusiasm or interest when discussing personal matters. It's important for them to find ways to rejuvenate and regain energy.",
    'type2': "Individuals of this type often feel pressured or stressed when dealing with matters related to their home environment. They may experience a sense of urgency or tension in situations involving home responsibilities or discussions. It's common for them to seek ways to alleviate this pressure and create a more relaxed atmosphere at home.",
    'type3': "Feelings of frustration may surface for individuals of this type when engaging in conversations about home life. They might encounter obstacles or difficulties that lead to irritation or dissatisfaction with their living situation. Finding effective coping mechanisms and problem-solving strategies can help them manage these feelings and foster a more positive outlook."
}


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict" , methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        try:
            f = request.files['file1']
            if f.filename == '':
                flash("No file selected for upload!", category="danger")
                return redirect(url_for('index'))
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
            f.save(file_path)
            predictions = test.predict(file_path)
            os.remove(file_path)
            return render_template("predict.html", predictions=predictions)
        except Exception as e:
            flash(f"An error occurred during file upload: {e}", category="danger")
            return redirect(url_for('index'))
    return redirect(url_for('index'))


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load the uploaded image and process it for classification
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        image = cv2.resize(image, (28, 28))  # Resize to (28x28)
        image = image / 255.0  # Normalize pixel values

        # Use your model for classification
        result = model.predict(np.expand_dims(image, axis=0))
        category_index = np.argmax(result)
        category = class_labels[category_index]
        traits = traits_mapping.get(category, "No traits found")  # Default message if no traits found


        # return render_template('result.html', category=category)
        return render_template('result.html', category=category, traits=traits)

if __name__ == '__main__':
    app.run(debug=True)
