# flask run --debugger --reload

import os
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import glob
import random
import datetime
import shutil
from PIL import Image
from flask import Flask, request, url_for, render_template, jsonify, send_file
from keras.models import load_model

UPLOAD_FOLDER = 'myenv/static/inputs'
remove_directories = ['./static/inputs', './static/outputs']

# model_path = 'myenv\model\saved_model.pb'  # replace with your model path
# model = tf.saved_model.load(model_path)

dehaze_model = tf.keras.models.load_model('model', compile=False)

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
    print('CLEAR OUTPUTS DIRECTORY')
    print(' ')
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

    if image.filename == '':
        return 'No selected image file', 400

    if not allowed_file(image.filename):
        return 'Invalid file type', 400

    # Save the image to the static folder
    image_url = url_for('static', filename='inputs/' + image.filename)
    image.save(app.root_path + image_url)

    start_time = datetime.datetime.now()

    result_path = evaluation(
        dehaze_model, 'myenv\static\inputs', image.filename)

    response = send_file(result_path, mimetype='image/jpeg')

    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    str_elapsed_time = str(elapsed_time.total_seconds())

    output = result_path["output"]
    psnr = result_path["psnr"]
    ssim = result_path["ssim"]

    print("oooooooooooooooo---->>>>>>>output:", output)
    print("ppppssssnnnnrrrr---->>>>>>>psnr:", psnr)
    print("sssssssssiiiiiim---->>>>>>>ssim:", ssim)

    print("RRRRRRRRRRRRRRRRR---->>>>>>>result_path:", result_path)

    print(">>>>>>>>>>>>>>>>>>>>>>>>Elapsed time:", elapsed_time)
    print(" ")
    print(">>>>>>>>>>>>>>>>>>>>>>>>STR Elapsed time:", str_elapsed_time)
    print(" ")

    data = {
        "input": image_url,
        "output": output,
        "time": str_elapsed_time,
        "psnr": psnr,
        "ssim": ssim,

        "message": "Image successfully dehazed",
        "status": 200
        # "version tf": tf.__version__,
        # "version np": np.__version__
    }
    return jsonify(data)

    # print(' ')
    # clear_directory('myenv\static\inputs')
    # print(' ')
    # print('CLEAR INPUTS DIRECTORY')
    # print(' ')
    # clear_directory('myenv\static\outputs')
    # print('CLEAR OUTPUTS DIRECTORY')
    # print(' ')

    # return (response)


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def evaluation(net, test_img_path, file_name):

    # test_img = glob.glob(test_img_path + '/*.jpg')
    test_img = glob.glob('./static/inputs/' + file_name)
    random.shuffle(test_img)

    print("Number of test images:", len(test_img))

    psnr_values = []
    ssim_values = []

    for img_path in test_img:

        print("Image path:", img_path)

        img = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(img, channels=3)

        img = tf.image.resize(img, size=(412, 548), antialias=True)
        img = img / 255
        # transform input image from 3D to 4D
        img = tf.expand_dims(img, axis=0)

        dehaze = net(img, training=False)

        my_array = dehaze[0].numpy()

        # Convert the NumPy array to an image
        generated_image = image.array_to_img(my_array)
        print(" ")
        print(" ")
        print("++++++++++++++++")
        print(" ")
        print(generated_image)
        print(" ")
        print(" ")

        # Save the image to a file
        output_image_path = "static/outputs/dehazed_" + file_name
        image.save_img(output_image_path, generated_image)

        # Calculate PSNR and SSIM
        psnr = tf.image.psnr(dehaze[0], img[0], max_val=1.0)
        ssim = tf.image.ssim(dehaze[0], img[0], max_val=1.0)

        # Append PSNR and SSIM values to lists
        psnr_values.append(psnr.numpy())
        ssim_values.append(ssim.numpy())

    # Compute and print average PSNR and SSIM
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    print("Average PSNR:", avg_psnr)
    print(" ")
    print("Average SSIM:", avg_ssim)
    print("Evaluation complete")

    # return output_image_path
    # , avg_psnr, avg_ssim
    data = {
        "output": output_image_path,
        "psnr": avg_psnr,
        "ssim": avg_ssim
    }
    return data


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

    # Save the dehazed image to the static folder
    # output_image = dehaze[0]
    # print(output_image)
    # output_image_name = "output_67"
    # output_image_url = url_for(
    #     'static', filename='outputs/' + output_image_name)
    # output_image.save(app.root_path + output_image_url)
    # return output_image_name
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
    return render_template('output.html', image_url=url_for('static', filename='inputs/' + filename))
