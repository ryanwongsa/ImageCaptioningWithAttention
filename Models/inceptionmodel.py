import tensorflow as tf
from Models.Helpers.helper import load_image
from tqdm import tqdm
import numpy as np
# tf.enable_eager_execution()

class InceptionModel(object):
    def __init__(self):
        tf.reset_default_graph()
        image_model = tf.keras.applications.InceptionV3(include_top=False, 
                                                weights='imagenet')
        new_input = image_model.input
        hidden_layer = image_model.layers[-1].output # last convolutional layer (8x8x2048)
        self.image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
        
    def cache_to_numpy_files_forward_pass(self, image_dataset):
        for img, path in tqdm(image_dataset):
            batch_features = self.image_features_extract_model(img) # (batch_size, 8, 8, 2048)         
            batch_features = tf.reshape(batch_features, 
                                      (batch_features.shape[0], -1, batch_features.shape[3])) # (batch_size, 64, 2048)

            for bf, p in zip(batch_features, path):
                path_of_feature = p.numpy().decode("utf-8")
                np.save(path_of_feature, bf.numpy())