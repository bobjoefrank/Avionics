# Mute tensorflow debugging information on console
import sys
if not hasattr(sys, 'argv'):
    sys.argv  = ['']
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from scipy.misc import imsave, imread, imresize
import numpy as np
import argparse
from keras.models import model_from_yaml
import re
import base64
import pickle
import json
import shutil
import io

import tensorflow as tf

graph = tf.get_default_graph()

def load_model(bin_dir):

    # load YAML and create model
    yaml_file = open('%s/model.yaml' % bin_dir, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)

    # load weights into new model
    model.load_weights('%s/model.h5' % bin_dir)
    return model

def test(image,model,mapping):

    # #model and weights location
    # bin_dir = '/Users/phillip/virtualenv/EMNIST/bin'
    #
    # # load model
    # model = load_model(bin_dir)
    # mapping = pickle.load(open('%s/mapping.p' % bin_dir, 'rb'))
    # print(mapping)
    # # read parsed image back in 8-bit, black and white mode (L)
    # x = imread('../pictures/saved_ocr.jpg', mode='L')
    #
    # # Visualize new array
    # imsave('../pictures/saved_ocr.png', x)
    x = imread(io.BytesIO(image), mode='L')

    x = imresize(x,(28,28))
    # reshape image data for use in neural network
    x = x.reshape(1, 28, 28, 1)

    # Convert type to float32
    x = x.astype('float32')

    # Normalize to prevent issues with model
    x /= 255

    # Predict from model
    with graph.as_default():
        out = model.predict(x)

    # Generate response
    response = {'prediction': chr(mapping[(int(np.argmax(out, axis=1)[0]))]),
                'confidence': str(max(out[0]) * 100)[:6]}
    #print result
    print(json.dumps(response))
    sys.stdout.flush()
    return response

if __name__ == '__main__':    #model and weights location
    bin_dir = '/Users/phillip/virtualenv/EMNIST/bin'

    #load model
    model = load_model(bin_dir)
    mapping = pickle.load(open('%s/mapping.p' % bin_dir, 'rb'))
    with open("1.jpg", 'rb') as f:
        b = bytearray(f.read())
    test(b, model, mapping)
