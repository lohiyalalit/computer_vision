#!/usr/bin/env python
# coding: utf-8
from flask import Flask,render_template,request
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

my_model = load_model('keras_model.h5')
app=Flask(__name__)

@app.route("/",methods = ['GET', 'POST'])
def welcome():
    return render_template('index.html')

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        # Replace this with the path to your image
        image = Image.open(img_path)
        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        #turn the image into a numpy array
        image_array = np.asarray(image)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # Load the image into the array
        input_array = normalized_image_array.reshape(1,224,224,3)

        # run the inference
        prediction = my_model.predict(input_array)

        if np.argmax(prediction) == 1:
          prediction = 'dog'
        else:
          prediction = 'cat'
        
    return render_template("index.html",img_path=img_path, prediction=prediction)

if __name__ == '__main__':
    app.run(host='localhost',debug=True)