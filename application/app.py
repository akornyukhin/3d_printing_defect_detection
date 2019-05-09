import cv2
import os
import h5py
import scipy.misc

import pandas as pd
import numpy as np
import tensorflow as tf

import sklearn.metrics as metrics

from flask import Flask, render_template, request
from werkzeug import secure_filename

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import keras.backend as K

UPLOAD_FOLDER = './static/uploaded/'
ALLOWED_EXTENSIONS = set(['jpg'])
RANDOM_STATE = 42

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/load_model_secret_api')
def load_model_secret_api():
    K.clear_session()
    load_model()
    return 'Model loaded', 200

@app.route('/calculate', methods = ['POST'])
def get_data():
    os.system("rm -rf ./static/uploaded/*")
    os.system("rm -rf ./static/img/*")
    
    if request.method == 'POST':
        # Get the name of the uploaded files
        uploaded_files = request.files.getlist("file")
        filenames = []
        for file in uploaded_files:
            # Check if the file is one of the allowed types/extensions
            if file and allowed_file(file.filename):
                # Make the filename safe, remove unsupported chars
                filename = secure_filename(file.filename)
                # Move the file form the temporal folder to the upload
                # folder we setup
                # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                file.save(os.path.join(UPLOAD_FOLDER, filename))
                # Save the filename into a list, we'll use it later
                filenames.append(filename)

    data = []
    for file in os.listdir(UPLOAD_FOLDER):
        if file.endswith(".jpg"):
            img = cv2.imread(UPLOAD_FOLDER+file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_resize = cv2.resize(gray, (256, 256))
            
            data.append(gray_resize/255.)

    data = np.reshape(data, (len(data), 256, 256, 1))

    data_proxy = np.reshape(data, (len(data), 256, 256))
    inputs= []
    i = 0
    for image in data_proxy:
        scipy.misc.imsave('./static/img/input_' + str(i) + '.jpg', image*255.)
        inputs.append('static/img/input_' + str(i) + '.jpg')
        i+=1

    with graph.as_default():
        encoded_imgs_256_cnn = encoder_256_cnn.predict(data[:, :, :, ])
        decoded_imgs_256_cnn = autoencoder_256_cnn.predict(data[:, :, :, :])

    filters = { 'filter1': [],
                'filter2': [],
                'filter3': [],
                'filter4': [],
                'filter5': [],
                'filter6': [],
                'filter7': [],
                'filter8': [],
                'filter9': [],
                'filter10': [],
                'filter11': [],
                'filter12': [],
                'filter13': [],
                'filter14': [],
                'filter15': [],
                'filter16': []}

    for i in range(encoded_imgs_256_cnn.shape[0]):
        for j in range(encoded_imgs_256_cnn[i, :, :, :].shape[2]):
            scipy.misc.imsave('./static/img/image_' + str(i) + '_filter_' + str(j) + '.jpg', cv2.resize(encoded_imgs_256_cnn[i, :, :, j]*255., (256, 256)))
            filters['filter'+str(j+1)].append('./static/img/image_' + str(i) + '_filter_' + str(j) + '.jpg')

    output_imgs = np.reshape(decoded_imgs_256_cnn, (len(decoded_imgs_256_cnn), 256, 256))
    outputs = []
    i = 0
    for image in output_imgs:
        scipy.misc.imsave('./static/img/output_' + str(i) + '.jpg', image*255.)
        outputs.append('./static/img/output_' + str(i) + '.jpg')
        i+=1
    
    anomalies = []
    difference = []
    for i in range(data.shape[0]):
        # diff_1 = (data[i, :, :, ]*255. - decoded_imgs_256_cnn[i]*255.).mean()
        diff_2 = np.linalg.norm(data[i, :, :, ]*255. - decoded_imgs_256_cnn[i]*255.)
        # diff_3 = metrics.mean_squared_error(np.reshape(decoded_imgs_256_cnn[i]*255., (256, 256)),
        #                               np.reshape(data[i, :, :, ]*255., (256, 256)))

        # diff_total = diff_1*.33 + diff_2*.33 + diff_3*.33
        difference.append(np.round(diff_2, 2))
        if diff_2 > 2745:
            anomalies.append(1)
        else:
            anomalies.append(0)

   
    print('--------')
    print(np.mean(difference))
    print(np.median(difference))
    print(anomalies)
    print(np.mean(anomalies)) 

    result = []
    result = zip(anomalies, difference, inputs, filters['filter1'], filters['filter2'],
                filters['filter3'], filters['filter4'], filters['filter5'],
                filters['filter6'], filters['filter7'], filters['filter8'],
                filters['filter9'], filters['filter10'], filters['filter11'],
                filters['filter12'], filters['filter13'], filters['filter14'],
                filters['filter15'], filters['filter16'], outputs)

    return render_template('./results.html', result=result, mean=np.mean(difference), med=np.median(difference), defect_percent=np.round(np.mean(anomalies), 2)*100)

def load_model():
    global autoencoder_256_cnn
    global encoder_256_cnn
    global graph
    graph = tf.get_default_graph()

    #Random seed
    np.random.seed(42)

    # Network architecture
    input_image_256_cnn = Input(shape=(256, 256, 1))

    # Encoding
    encoded_256_cnn = Conv2D(32, (5, 5), activation='relu', padding='same')(input_image_256_cnn)
    encoded_256_cnn = MaxPooling2D((2, 2), padding='same')(encoded_256_cnn)

    encoded_256_cnn = Conv2D(32, (5, 5), activation='relu', padding='same')(encoded_256_cnn)
    encoded_256_cnn = MaxPooling2D((2, 2), padding='same')(encoded_256_cnn)

    encoded_256_cnn = Conv2D(16, (5, 5), activation='relu', padding='same')(encoded_256_cnn)
    encoded_256_cnn = MaxPooling2D((2, 2), padding='same')(encoded_256_cnn)


    # Decoding
    decoded_256_cnn = Conv2D(16, (5, 5), activation='relu', padding='same')(encoded_256_cnn)
    decoded_256_cnn = UpSampling2D((2, 2))(decoded_256_cnn)

    decoded_256_cnn = Conv2D(32, (5, 5), activation='relu', padding='same')(decoded_256_cnn)
    decoded_256_cnn = UpSampling2D((2, 2))(decoded_256_cnn)

    decoded_256_cnn = Conv2D(32, (5, 5), activation='relu', padding='same')(decoded_256_cnn)
    decoded_256_cnn = UpSampling2D((2, 2))(decoded_256_cnn)

    decoded_256_cnn = Conv2D(1, (5, 5), activation='relu', padding='same')(decoded_256_cnn)
    
    # Autoencoder
    autoencoder_256_cnn = Model(input_image_256_cnn, decoded_256_cnn)

    # Encoder
    encoder_256_cnn = Model(input_image_256_cnn, encoded_256_cnn)

    autoencoder_256_cnn.compile(optimizer='adadelta', loss=euclidean_distance_loss)

    # load weights into new model
    autoencoder_256_cnn.load_weights("./static/index/autoencoder_cnn_256_deep_euclidian.h5")

    return 

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true)))

if __name__ == '__main__':
    # port = 8008

    # if os.getenv('PORT') != "":
    #     port = int(os.getenv('PORT'))

    app.run(host='0.0.0.0',
            port = 8008,
            debug=True)