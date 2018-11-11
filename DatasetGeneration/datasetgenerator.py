import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

class DatasetGenerator(object):
    def __init__(self, img_name_train, cap_train, batch_size=16, buffer_size=1000, num_parallel_calls=8):
        BATCH_SIZE = batch_size
        BUFFER_SIZE = buffer_size

        self.dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
        
        self.dataset = self.dataset.map(lambda item1, item2: tf.py_func(
          self.map_func, [item1, item2], [tf.float32, tf.int32]), num_parallel_calls=8)
        
        self.dataset = self.dataset.shuffle(BUFFER_SIZE)
        self.dataset = self.dataset.batch(BATCH_SIZE)
        self.dataset = self.dataset.prefetch(1)
        
    def map_func(self, img_name, cap):
        img_tensor = np.load(img_name.decode('utf-8')+'.npy')
        return img_tensor, cap