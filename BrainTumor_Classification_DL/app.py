import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

# Build the model
base_model = VGG19(include_top=False, input_shape=(240, 240, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(4608, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(1152, activation='relu')(x)
output = Dense(2, activation='softmax')(x)
model_03 = Model(inputs=base_model.input, outputs=output)

# Load pre-trained weights
# model_03.save_weights('vgg_unfrozen.weights.h5')
model_03.load_weights('vgg_unfrozen.weights.h5')

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print('Model loaded. Check http://127.0.0.1:5051/')

def get_className(classNo):
    return "No Brain Tumor" if classNo == 0 else "Yes Brain Tumor"

def getResult(img_path):
    image = cv2.imread(img_path)
    image = Image.fromarray(image, 'RGB').resize((240, 240))
    input_img = np.expand_dims(np.array(image), axis=0)
    prediction = model_03.predict(input_img)
    return np.argmax(prediction, axis=1)[0]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400

    f = request.files['file']
    if f.filename == '':
        return "No selected file", 400

    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
    f.save(file_path)

    result = get_className(getResult(file_path))
    return result

if __name__ == '__main__':
    app.run(debug=True, port=5052)
