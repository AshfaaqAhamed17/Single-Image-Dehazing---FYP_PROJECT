# flask run --debugger --reload

import os
import io
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from tensorflow.image import psnr, ssim
import glob
import random
import datetime
import shutil
from flask import Flask, request, url_for, render_template, jsonify, send_file
from keras.models import load_model

# Evaluation
import generator

UPLOAD_FOLDER = 'myenv/static/inputs'
remove_directories = ['./static/inputs',
                      './static/outputs', './static/inputsResized']

generator = generator.build_generator()

# model = load_model(model_path)
app = Flask(__name__)
app.config['DEBUG'] = True                      # Turn off for production
app.config['TEMPLATES_AUTO_RELOAD'] = True      # Turn off for production
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def landing():
    return render_template('index.html')


@app.route('/main')
def main():
    print(' ')
    clear_directory('myenv\static\inputs')
    print(' ')
    print('CLEAR INPUTS DIRECTORY')
    print(' ')
    clear_directory('myenv\static\outputs')
    print(' ')
    print('CLEAR OUTPUTS DIRECTORY')
    print(' ')
    clear_directory('myenv\static\inputsResized')
    print(' ')
    print('CLEAR RESIZED INPUTS DIRECTORY')
    print(' ')
    print(generator)
    return render_template('upload.html')

# ---------------------------------------------------------------------------------------


@app.route('/contactus')
def contactus():
    return render_template('contactus.html')


@app.route('/loading', methods=['POST', 'GET'])
def loading():

    if 'image' not in request.files:
        return 'No image file provided', 400

    image = request.files['image']
    image.filename = image.filename.replace(' ', '')
    if image.filename == '':
        return 'No selected image file', 400

    if not allowed_file(image.filename):
        return 'Invalid file type', 400

    image_type = request.form['image_type']
    print("IMAGE TYPE ------------------> ", image_type)

    # Save the image to the static folder
    image_url = url_for('static', filename='inputs/' + image.filename, )
    image.save(app.root_path + image_url)

    start_time = datetime.datetime.now()

    result_path = evaluation('static/inputs/', image.filename, image_type)

    response = send_file(result_path, mimetype='image/jpeg')

    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    str_elapsed_time = str(elapsed_time.total_seconds())

    input_r = result_path["resized"]
    output = result_path["output"]

    print("oooooooooooooooo---->>>>>>>output:", output)

    print(">>>>>>>>>>>>>>>>>>>>>>>>Elapsed time:", elapsed_time)
    print(" ")
    print(">>>>>>>>>>>>>>>>>>>>>>>>STR Elapsed time:", str_elapsed_time)
    print(" ")

    data = {
        "input": input_r,
        "output": output,
        "time": str_elapsed_time,
        "message": "Image successfully dehazed",
        "status": 200
    }
    return jsonify(data)


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    filename = filename.replace(' ', '')
    print(filename)
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def evaluation(img_root_path, file_name, image_type):

    print("FILE NAME ------------------> ", file_name)
    image_path = (img_root_path + file_name)
    print("IMAGE PATH ------------------> ", image_path)

    # Load a hazy image
    input_image_path = image_path
    hazy_image = Image.open(input_image_path)

    # Preprocess the hazy image
    # Resize the image to match the generator input size
    hazy_image = hazy_image.resize((256, 256))

    hazy_image_url = url_for(
        'static', filename='inputsResized/' + file_name)
    hazy_image.save(app.root_path + hazy_image_url)

    # Scale pixel values to the range [-1, 1]
    hazy_image = np.array(hazy_image) / 127.5 - 1.0
    hazy_image = np.expand_dims(hazy_image, axis=0)  # Add a batch dimension

    if image_type == "0":
        # OUTDOOR LOAD MODEL
        print('----------------------------------------------')
        print("++++++++------LOAD OUTDOOR MODEL------++++++++")
        print('----------------------------------------------')
        # generator.load_weights(
        #     'TrainedModel\OutdoorModel\outdoor_generator_0050.h5')
        generator.load_weights(
            'TrainedModel/OutdoorModel/new_outdoor_generator_0060.h5')
        # Pass the preprocessed image through the generator
        dehazed_image = generator.predict(hazy_image)
    elif image_type == "1":
        # INDOOR LOAD MODEL
        print('---------------------------------------------')
        print("++++++++------LOAD INDOOR MODEL------++++++++")
        print('---------------------------------------------')

        # generator.load_weights(
        #     'TrainedModel\IndoorModel\indoor_generator_0050.h5')
        generator.load_weights(
            'TrainedModel/IndoorModel/new_indoor_generator_0050.h5')

        # Pass the preprocessed image through the generator
        dehazed_image = generator.predict(hazy_image)

    # Post-process the dehazed image
    dehazed_image = np.squeeze(dehazed_image)  # Remove the batch dimension
    # Rescale pixel values to the range [0, 255]
    dehazed_image = (dehazed_image + 1.0) * 127.5
    dehazed_image = np.clip(dehazed_image, 0, 255).astype(
        np.uint8)  # Clip values and convert to uint8

    # Display or save the dehazed image
    output_image_path = "static/outputs/dehazed_" + file_name
    Image.fromarray(dehazed_image).save(output_image_path)

    # Load the hazy and dehazed images as arrays
    hazy_image = np.array(Image.open(input_image_path).resize((256, 256)))
    dehazed_image = np.array(Image.open(output_image_path))

    data = {
        "resized": hazy_image_url,
        "output": output_image_path,
    }
    return data

# ---------------------------------------------------------------------------------------


def clear_directory(directory_path):

    print("-------------   Clearing directory:", directory_path)
    for directory in remove_directories:
        for file in os.listdir(directory):
            full_path = os.path.join(directory, file)
            try:
                if os.path.isfile(full_path) or os.path.islink(full_path):
                    os.unlink(full_path)
                elif os.path.isdir(full_path):
                    shutil.rmtree(full_path)
            except Exception as e:
                print('Failed to delete %s. due to: %s' % (full_path, e))

# ---------------------------------------------------------------------------------------


@app.route('/save_images', methods=['POST'])
def save_images():
    print(" ")
    print("SAVE CLICKED")

    if 'hazy_image' not in request.files:
        return 'No image file provided', 400

    if 'dehazed_image' not in request.files:
        return 'No image file provided', 400

    hazy_image = request.files['hazy_image']
    dehazed_image = request.files['dehazed_image']

    print(" ")
    print("HAZY IMAGE to save :", hazy_image.filename)
    print("DEHAZED IMAGE to save :", dehazed_image.filename)
    print(" ")

    # Save the image to the static folder
    hazy_image_url = url_for(
        'static', filename='dataset/hazy_images/' + hazy_image.filename)
    hazy_image.save(app.root_path + hazy_image_url)

    dehazed_image_url = url_for(
        'static', filename='dataset/dehazed_images/' + dehazed_image.filename)
    dehazed_image.save(app.root_path + dehazed_image_url)

    return "Image saved successfully to the server. Thank you!"

# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    return render_template('loading.html')

# ---------------------------------------------------------------------------------------


# @app.route('/output', methods=['POST', 'GET'])
# def output():

@app.route('/output/<filename>', methods=['POST', 'GET'])
def output(filename):
    # return render_template('output.html')
    filename = filename.replace(' ', '')
    return render_template('output.html', image_url=url_for('static', filename='inputs/' + filename))
