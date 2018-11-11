import tensorflow as tf
from Models.Helpers.helper import load_image

tf.enable_eager_execution()

class DatasetGeneratorNoLabels(object):
    def __init__(self, encode_train, batch_size=16):
        self.dataset = tf.data.Dataset.from_tensor_slices(
                                        encode_train).map(load_image).batch(batch_size)