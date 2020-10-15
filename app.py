# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 2020

@author: Pavan Naik
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf

#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# Keras
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
#MODEL_PATH = 'InceptionV3.h5'

# Load your trained model
#model = load_model('InceptionV3.h5')


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = x / 255
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "Bacterial Spot"
    elif preds == 1:
        preds = "Early Blight"
    elif preds == 2:
        preds = "Healthy"
    elif preds == 2:
        preds = "Late Blight"
    elif preds == 2:
        preds = "Leaf Mold"
    elif preds == 2:
        preds = "Septorial Leaf Spot"
    elif preds == 2:
        preds = "Spider Mites Two Spotted Spider mint"
    elif preds == 2:
        preds = "Target Spot"
    elif preds == 2:
        preds = "Tomoto Mosai virus"
    else:
        preds = "Tomoto Yellow Leaf Curl Virus"

    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post requestmode
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        model = load_model('InceptionV3.h5')
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001, debug=True)
