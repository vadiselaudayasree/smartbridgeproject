# app.py
import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('model/cleantech_vgg.h5')
classes = ['biodegradable', 'recyclable', 'trash']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_url = None
    if request.method == 'POST':
        img = request.files['image']
        path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
        img.save(path)

        # Preprocess image
        img_data = image.load_img(path, target_size=(224, 224))
        img_arr = image.img_to_array(img_data)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = preprocess_input(img_arr)

        result = model.predict(img_arr)
        prediction = classes[np.argmax(result)]
        img_url = path

    return render_template('index.html', prediction=prediction, img_url=img_url)

if __name__ == '__main__':
    app.run(debug=True)
