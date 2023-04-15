# flask run --debugger --reload

import os
import io
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from tensorflow.image import psnr, ssim
# from skimage.metrics import structural_similarity as ssim
import glob
import random
import datetime
import shutil
from flask import Flask, request, url_for, render_template, jsonify, send_file
from keras.models import load_model

# Evaluation
import generator


UPLOAD_FOLDER = 'myenv/static/inputs'
remove_directories = ['./static/inputs', './static/outputs']

# model_path = 'myenv\model\saved_model.pb'  # replace with your model path
# model = tf.saved_model.load(model_path)

generator = generator.build_generator()
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

    # input_r = 'static/inputsNew/' + image.filename
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

    print("FILE NAME ------------------> ", file_name)
    generator.load_weights('TrainedModel\generator_0005.h5')
    image_path = ('./static/inputs/' + file_name)
    print("IMAGE PATH ------------------> ", image_path)

    # Load a hazy image
    input_image_path = image_path
    hazy_image = Image.open(input_image_path)

    # Preprocess the hazy image
    # Resize the image to match the generator input size
    hazy_image = hazy_image.resize((256, 256))
    # Scale pixel values to the range [-1, 1]
    hazy_image = np.array(hazy_image) / 127.5 - 1.0
    hazy_image = np.expand_dims(hazy_image, axis=0)  # Add a batch dimension

    # output_folder = "static/inputsNew/"
    # output_filename = file_name  # Change the filename as needed
    # output_path = os.path.join(output_folder, output_filename)
    # resized_hazy_image = Image.fromarray(
    #     (hazy_image[0] * 127.5 + 1.0).astype(np.uint8))
    # resized_hazy_image.save(output_path)

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

    # Calculate the PSNR and SSIM scores
    # psnr = peak_signal_noise_ratio(hazy_image, dehazed_image)

    hazy_image = np.array(Image.open(input_image_path).resize((256, 256)))
    dehazed_image = np.array(Image.open(output_image_path))

    hazy_image = np.array(Image.open(input_image_path).resize((256, 256)))
    dehazed_image = np.array(Image.open(output_image_path))

    hazy_image = hazy_image.astype(np.float32) / 255.0
    dehazed_image = dehazed_image.astype(np.float32) / 255.0

    psnr = tf.image.psnr(tf.expand_dims(
        hazy_image, axis=0), tf.expand_dims(dehazed_image, axis=0), max_val=1.0)
    ssim = tf.image.ssim(tf.expand_dims(hazy_image, axis=0),
                         tf.expand_dims(dehazed_image, axis=0), max_val=1.0)

    psnr = psnr.numpy()[0]
    ssim = ssim.numpy()[0]

    # ssim = structural_similarity(hazy_image, dehazed_image, multichannel=True)

    # # Print the scores
    print('PSNR ++++++++++++>>>>>: ', psnr)
    print('SSIM ============>>>>>: ', ssim)

    # =================================================================================================

    # test_img = glob.glob(test_img_path + '/*.jpg')
    # test_img = glob.glob('./static/inputs/' + file_name)
    # random.shuffle(test_img)

    # print("Number of test images:", len(test_img))

    # psnr_values = []
    # ssim_values = []

    # for img_path in test_img:

    #     print("Image path:", img_path)

    #     img = tf.io.read_file(img_path)
    #     img = tf.io.decode_jpeg(img, channels=3)

    #     img = tf.image.resize(img, size=(412, 548), antialias=True)
    #     img = img / 255
    #     # transform input image from 3D to 4D
    #     img = tf.expand_dims(img, axis=0)

    #     dehaze = net(img, training=False)

    #     my_array = dehaze[0].numpy()

    #     # Convert the NumPy array to an image
    #     generated_image = image.array_to_img(my_array)
    #     print(" ")
    #     print(" ")
    #     print("++++++++++++++++")
    #     print(" ")
    #     print(generated_image)
    #     print(" ")
    #     print(" ")

    #     # Save the image to a file
    #     output_image_path = "static/outputs/dehazed_" + file_name
    #     image.save_img(output_image_path, generated_image)

    #     # Calculate PSNR and SSIM
    #     psnr = tf.image.psnr(dehaze[0], img[0], max_val=1.0)
    #     ssim = tf.image.ssim(dehaze[0], img[0], max_val=1.0)

    #     # Append PSNR and SSIM values to lists
    #     psnr_values.append(psnr.numpy())
    #     ssim_values.append(ssim.numpy())

    # # Compute and print average PSNR and SSIM
    # avg_psnr = sum(psnr_values) / len(psnr_values)
    # avg_ssim = sum(ssim_values) / len(ssim_values)
    # print("Average PSNR:", avg_psnr)
    # print(" ")
    # print("Average SSIM:", avg_ssim)
    # print("Evaluation complete")

    # return output_image_path
    # , avg_psnr, avg_ssim
    data = {
        "output": output_image_path,
        "psnr": psnr.item(),
        "ssim": ssim.item()
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
