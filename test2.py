import argparse
import os

import wget

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
import cv2
import numpy as np
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import display_images
from utils import predict as depth_estimation
from matplotlib import pyplot as plt
# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

current_directory = os.getcwd()
print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)
os.chdir(current_directory)

print('\nModel loaded ({0}).'.format(args.model))


# Input images
# inputs = load_images(glob.glob(args.input))

def predict(cv2_image):
    global model
    # inputs = load_image("examples/470_image.png")
    x = np.clip(
        np.asarray(cv2.cvtColor(cv2.resize(cv2_image, (640, 480)), cv2.COLOR_BGR2RGB), dtype=float) / 255, 0, 1)
    inputs = np.stack([x], axis=0)
    del x
    print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

    # Compute results
    outputs = depth_estimation(model, inputs)
    model._make_predict_function()
    outputs = (np.squeeze(outputs) * 255).astype(np.uint8)
    return outputs
    # exit()
    #
    # # matplotlib problem on ubuntu terminal fix
    # # matplotlib.use('TkAgg')
    #
    # # Display results
    # viz = display_images(outputs.copy(), inputs.copy())
    # plt.figure(figsize=(10, 5))
    # plt.imshow(viz)
    # plt.savefig('test.png')
    # plt.show()
