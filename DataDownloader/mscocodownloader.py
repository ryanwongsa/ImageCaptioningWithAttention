import tensorflow as tf
import json
import os
from sklearn.utils import shuffle

tf.enable_eager_execution()

class MSCocoDownloader(object):
    def __init__(self, sample_count = None):
        annotation_zip = tf.keras.utils.get_file('captions.zip', 
                                          cache_subdir=os.path.abspath('.'),
                                          origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                          extract = True)
        
        self.annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'

        name_of_zip = 'train2014.zip'
        if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):
            image_zip = tf.keras.utils.get_file(name_of_zip, 
                                      cache_subdir=os.path.abspath('.'),
                                      origin = 'http://images.cocodataset.org/zips/train2014.zip',
                                      extract = True)
            sefl.PATH = os.path.dirname(image_zip)+'/train2014/'
        else:
            self.PATH = os.path.abspath('.')+'/train2014/'
            
        self.annotations = self.load_data()
            
        self.train_captions, self.image_name_vector = self.create_dataset()
        
        if sample_count!= None:
            self.train_captions = self.train_captions[:sample_count]
            self.image_name_vector = self.image_name_vector[:sample_count]
    
    def load_data(self):
        with open(self.annotation_file, 'r') as f:
            return json.load(f)
        return None
    
    def create_dataset(self):
        all_captions = []
        all_img_name_vector = []

        for annot in self.annotations['annotations']:
            caption = '<start> ' + annot['caption'] + ' <end>'
            image_id = annot['image_id']
            full_coco_image_path = self.PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

            all_img_name_vector.append(full_coco_image_path)
            all_captions.append(caption)

        train_captions, img_name_vector = shuffle(all_captions,
                                                  all_img_name_vector,
                                                  random_state=1)
        
        return train_captions, img_name_vector