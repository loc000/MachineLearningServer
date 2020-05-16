from __future__ import division
from __future__ import print_function
import os
import sys
import math
import argparse
import tarfile

import skimage.transform
import matplotlib.pyplot as plt
import wget
from six.moves import cPickle as pickle
import numpy as np
from scipy import ndimage
from PIL import Image
import tensorflow as tf
from ShowAttendAndTellModel.core.utils import load_coco_data, decode_captions
from ShowAttendAndTellModel.core.model import CaptionGenerator

# inference on a batch of images
class CaptionInference(object):
    def __init__(self, sess, model_path, use_inception):

        path_prefix = os.path.dirname(os.path.realpath(__file__))
        # word to index mapping
        with open(os.path.join(path_prefix, 'data/train/word_to_idx.pkl'), "rb") as f:
            self.word_to_idx = pickle.load(f)

        if use_inception:
            L = 64
            D = 2048
            cnn_model_path = os.path.join(path_prefix, 'data/inception_v3.ckpt')
            if not os.path.isfile(cnn_model_path):
                print("{} not found, downloading....".format(cnn_model_path))
                cwd = os.getcwd()
                os.chdir(os.path.dirname(os.path.abspath(__file__)))
                wget.download("http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz","inception_v3_2016_08_28.tar.gz")
                tar = tarfile.open("inception_v3_2016_08_28.tar.gz", "r:gz")
                tar.extractall()
                tar.close()
                os.remove("inception_v3_2016_08_28.tar.gz")
                os.rename("inception_v3.ckpt","data/inception_v3.ckpt")
                print(os.getcwd())
                os.chdir(cwd)
        else:
            L = 196
            D = 512
            cnn_model_path = os.path.join(path_prefix, './data/imagenet-vgg-verydeep-19.mat')
        self.batch_size = 128
        self.sess = sess
        self.use_inception = use_inception
        print("Creating model...")
        self.model = CaptionGenerator(self.word_to_idx, dim_feature=[L, D], dim_embed=512,
                                 dim_hidden=1800, n_time_step=16, prev2out=True, 
                                 ctx2out=True, alpha_c=5.0, selector=True, dropout=True, 
                                 use_cnn = "inception" if use_inception else "vgg",
                                 cnn_model_path = cnn_model_path)

        print("Loading CNN weights...")
        self.model.cnn.load_weights(sess)
        print("Building sampler...")
        self.alphas, self.betas, self.generated_captions = self.model.build_sampler(max_len=20)

        # initialize model and load weights
        print("Loading LSTM weights...")
        # tf.global_variables_initializer().run()
        saver = tf.train.Saver(self.model.sampler_vars)
        cwd = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isfile(model_path+".data-00000-of-00001"):
            print("{} not found, downloading....".format(model_path+".data-00000-of-00001"))
            wget.download("http://jaina.cs.ucdavis.edu/datasets/adv/captioning/attention-model-best.tar.bz2","attention-model-best.tar.bz2")
            tar = tarfile.open("attention-model-best.tar.bz2", "r:bz2")
            tar.extractall()
            tar.close()
            os.remove("attention-model-best.tar.bz2")
            # os.rename("model_best",os.path.dirname(os.path.dirname(model_path)))
            print(os.getcwd())
        saver.restore(sess, model_path)
        os.chdir(cwd)

    def inference_np(self, images):
        nimgs = images.shape[0]
        print("Running inference on {} images...".format(nimgs))
        nbatches = int(math.ceil(nimgs / self.batch_size))
        all_decoded = []
        all_alphas = None
        all_betas = None
        for i in range(nbatches):
            start = i * self.batch_size
            end = (i+1) * self.batch_size
            end = nimgs if end >= nimgs else end
            batch_images = images[start:end]
            print("processing {} images ({} to {})".format(batch_images.shape[0], start + 1, end))
            batch_alphas, batch_betas, batch_gen_cap = self.sess.run([self.alphas, self.betas, self.generated_captions], feed_dict = {self.model.images: batch_images})
            # batch_gen_cap = self.sess.run(self.generated_captions, feed_dict = {self.model.images: batch_images})
            batch_decoded = decode_captions(batch_gen_cap, self.model.idx_to_word)
            all_decoded.extend(batch_decoded)
            all_alphas = np.concatenate([all_alphas, batch_alphas]) if all_alphas is not None else batch_alphas
            all_betas = np.concatenate([all_betas, batch_betas]) if all_betas is not None else batch_betas
        return all_alphas, all_betas, all_decoded

    @staticmethod
    def resize_image(image, image_size):
        width, height = image.size
        if width > height:
            left = (width - height) / 2
            right = width - left
            top = 0
            bottom = height
        else:
            top = (height - width) / 2
            bottom = height - top
            left = 0
            right = width
        image = image.crop((left, top, right, bottom))
        image = image.resize([image_size, image_size], Image.ANTIALIAS)
        return image
    
    def preprocess_file(self, file_name):
        print("preprocess", file_name)
        if os.path.splitext(file_name)[1] == ".npy":
            return np.squeeze(np.load(file_name))
        else:
            img_np = np.array(self.resize_image(Image.open(file_name), self.model.cnn.image_size)).astype(np.float32)
            # convert grey scale image to 3-channel
            if self.use_inception:
                img_np /= 255.0
                img_np -= 0.5
                img_np *= 2.0
            if img_np.ndim == 2:
                img_np = np.stack((img_np,)*3, axis = -1)
            return img_np


    def inference_files(self, image_files):
        print("processing {} images...".format(len(image_files)))
        image_batch = np.array([self.preprocess_file(x) for x in image_files])
        return self.inference_np(image_batch)
        

# visualize attention
def visualize(alpha, beta, caption, filename, use_inception = False):
    if use_inception:
        att_shape = 8
        upscale = 37.375
        img_size = 299
    else:
        att_shape = 14
        upscale = 16
        img_size = 224

    # Plot original image
    if os.path.splitext(filename)[1] == ".npy":
        img = np.squeeze(np.load(filename))
        img /= 2.0
        img += 0.5
    else:
        img = np.array(CaptionInference.resize_image(Image.open(filename), img_size))
        # img = ndimage.imread(filename)

    # Plot images with attention weights
    words = caption.split(" ")
    nrows = 4
    ncols = 4
    plt.subplot(nrows, ncols, 1)
    plt.imshow(img, interpolation='mitchell')
    plt.axis('off')

    for t in range(len(words)):
        if t > nrows * ncols - 2:
            break
        plt.subplot(nrows, ncols, t+2)
        plt.text(0, 1, '%s(%.2f)'%(words[t], beta[t]) , color='black', backgroundcolor='white', fontsize=8)
        plt.imshow(img, interpolation='mitchell')
        alp_curr = alpha[t,:].reshape(att_shape,att_shape)
        alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=upscale, sigma=20)
        plt.imshow(alp_img, alpha=0.85, cmap='gray', interpolation='mitchell')
        plt.axis('off')
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(os.path.splitext(os.path.splitext(os.path.basename(filename))[0])[0] + '_att.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_inception", action="store_true", help="use inception network image size (299 * 299)")
    parser.add_argument("--model_path", default="model_best/model-best", help="lstm model path")
    parser.add_argument("--visualize", action="store_true", help="save attention image")
    parser.add_argument("--output", default="", help="save output to a file")
    parser.add_argument("image", type=str, nargs='+', help="image file paths")
    args = parser.parse_args()

    image_files = sorted(args.image)
    if args.output:
        out_file = open(args.output, "w")
    else:
        out_file = sys.stdout
    with tf.Session() as sess:
        cap_infer = CaptionInference(sess, args.model_path, args.use_inception)
        alphas, betas, captions = cap_infer.inference_files(image_files)
        for fname, alpha, beta, caption in zip(image_files, alphas, betas, captions):
            out_file.write("{}\t{}\n".format(fname, caption))
            if args.visualize:
                visualize(alpha, beta, caption, fname, args.use_inception)
    if args.output:
        out_file.close()
        print("results saved to {}".format(args.output))

