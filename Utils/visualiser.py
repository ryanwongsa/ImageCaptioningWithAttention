import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class Visualiser(object):
    def __init__(self):
        pass
    
    def show_image(self, image_path, caption):
        plt.figure()
        img = np.array(Image.open(image_path))
        plt.title(caption)
        fig = plt.imshow(img)
        
    def plot_attention(self, image, result, attention_plot):
        temp_image = np.array(Image.open(image))

        fig = plt.figure(figsize=(10, 10))

        len_result = len(result)
        for l in range(len_result):
            temp_att = np.resize(attention_plot[l], (8, 8))
            ax = fig.add_subplot(len_result//2, len_result//2, l+1)
            ax.set_title(result[l])
            img = ax.imshow(temp_image)
            ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

        plt.tight_layout()
        plt.show()