import tensorflow as tf
import scipy.io
from ShowAttendAndTellModel.core.inception_core import inception_v3

class InceptionV3(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.image_size = 299
        self.L = 64
        self.D = 2048
        self.inception_variables = []

    def build_inputs(self):
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], 'images')

    def load_weights(self, sess):
        saver = tf.train.Saver(self.inception_variables)
        tf.logging.info("Restoring Inception variables from checkpoint file %s",
                        self.model_path)
        saver.restore(sess, self.model_path)

    def build_model(self):
        self.features = inception_v3(
            self.images,
            trainable=False,
            is_training=False)
        self.inception_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

    def build(self):
        self.build_inputs()
        self.build_model()
