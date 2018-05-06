# Mute tensorflow debugging information on console
import sys
if not hasattr(sys, 'argv'):
    sys.argv  = ['']
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import scipy
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

def load_model(bin_dir):

    # load YAML and create model
    yaml_file = open('%s/model.yaml' % bin_dir, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)

    # load weights into new model
    model.load_weights('%s/model.h5' % bin_dir)
    return model

def test():

    #model and weights location
    bin_dir = '/Users/phillip/Desktop/projects/Avionics/Object_Detection/models'

    # load model
    model = load_model(bin_dir)
    mapping = pickle.load(open('%s/mapping.p' % bin_dir, 'rb'))

    # read parsed image back in 8-bit, black and white mode (L)
    x = imread('../pictures/saved_ocr.jpg', mode='L')

    x = imresize(x,(28,28))
    # reshape image data for use in neural network
    x = x.reshape(1, 28, 28, 1)

    # Convert type to float32
    x = x.astype('float32')

    # Normalize to prevent issues with model
    x /= 255

    # Predict from model
    out = model.predict(x)

    # Generate response
    response = {'prediction': chr(mapping[(int(np.argmax(out, axis=1)[0]))]),
                'confidence': str(max(out[0]) * 100)[:6]}
                
    #print result
    print(json.dumps(response))

if __name__ == '__main__':
    test()
