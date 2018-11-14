import tensorflow as tf

# tf.enable_eager_execution()

import time
import numpy as np
import os

from Models.rnndecoder import RNN_Decoder
from Models.cnnencoder import CNN_Encoder
from Models.Helpers.helper import load_image
from tqdm import tqdm

class AttentionModel(object):
    def __init__(self, tokenisation, embedding_dim, units, vocab_size, batch_size, attention_features_shape, image_features_extract_model):
        tf.reset_default_graph()
        
        self.batch_size = batch_size
        self.encoder = CNN_Encoder(embedding_dim)
        self.decoder = RNN_Decoder(embedding_dim, units, vocab_size)
        
        self.tokenizer = tokenisation.tokenizer
        self.tokenisation = tokenisation
        
        self.optimizer = tf.train.AdamOptimizer()
        self.attention_features_shape = attention_features_shape
        self.image_features_extract_model = image_features_extract_model
        self.attention_features_shape = attention_features_shape
        self.loss_plot = []
        
        self.save_checkpoint()
        
    def loss_function(self, real, pred):
        mask = 1 - np.equal(real, 0)
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
        return tf.reduce_mean(loss_)
    
    def train(self, dataset, epochs):
        for epoch in range(epochs):
            start = time.time()
            total_loss = 0
            pbar = tqdm(enumerate(dataset))
            for (batch, (img_tensor, target)) in pbar:
                loss = 0

                # initializing the hidden state for each batch
                # because the captions are not related from image to image
                hidden = self.decoder.reset_state(batch_size=target.shape[0])

                dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']] * self.batch_size, 1)
                
                with tf.GradientTape() as tape:
                    features = self.encoder(img_tensor)

                    for i in range(1, target.shape[1]):
                        # passing the features through the decoder
                        predictions, hidden, _ = self.decoder(dec_input, features, hidden)

                        loss += self.loss_function(target[:, i], predictions)

                        # using teacher forcing
                        dec_input = tf.expand_dims(target[:, i], 1)

                total_loss += (loss / int(target.shape[1]))
                pbar.set_description("Processing "+ str(batch) + ": "+ str(loss))


                variables = self.encoder.variables + self.decoder.variables

                gradients = tape.gradient(loss, variables) 

                self.optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())

                if batch % 20 == 0:
                    print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, 
                                                                  batch, 
                                                                  loss.numpy() / 
                                                                  int(target.shape[1])))
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)
            # storing the epoch end loss value to plot later
            self.loss_plot.append(total_loss/(batch*self.batch_size))

            print ('Epoch {} Loss {:.6f}'.format(epoch + 1, 
                                                 total_loss/(batch*self.batch_size)))
            print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    
            
    def evaluate(self, image):
        with tf.contrib.eager.restore_variables_on_create(tf.train.latest_checkpoint('training_checkpoints2')):
            attention_plot = np.zeros((self.tokenisation.max_length, self.attention_features_shape))

            hidden = self.decoder.reset_state(batch_size=1)

            temp_input = tf.expand_dims(load_image(image)[0], 0)
            img_tensor_val = self.image_features_extract_model(temp_input)
            img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

            features = self.encoder(img_tensor_val)

            dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
            result = []

            for i in range(self.tokenisation.max_length):
                predictions, hidden, attention_weights = self.decoder(dec_input, features, hidden)

                attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

                predicted_id = tf.multinomial(predictions, num_samples=1)[0][0].numpy()
                result.append(self.tokenisation.index_word[predicted_id])

                if self.tokenisation.index_word[predicted_id] == '<end>':
                    return result, attention_plot

                dec_input = tf.expand_dims([predicted_id], 0)

            attention_plot = attention_plot[:len(result), :]
            return result, attention_plot

    def evaluate2(self, image):
        with tf.contrib.eager.restore_variables_on_create(tf.train.latest_checkpoint('training_checkpoints2')):
            attention_plot = np.zeros((self.tokenisation.max_length, self.attention_features_shape))

            hidden = self.decoder.reset_state(batch_size=1)
#             print(load_image(image)[0])
#             temp_input = tf.expand_dims(load_image(image)[0], 0)
            img = tf.keras.applications.inception_v3.preprocess_input(image.astype(np.float32))
            temp_input = tf.expand_dims(img, 0)

            img_tensor_val = self.image_features_extract_model(temp_input)
            img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

            features = self.encoder(img_tensor_val)

            dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
            result = []

            for i in range(self.tokenisation.max_length):
                predictions, hidden, attention_weights = self.decoder(dec_input, features, hidden)

                attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

                predicted_id = tf.multinomial(predictions, num_samples=1)[0][0].numpy()
                result.append(self.tokenisation.index_word[predicted_id])

                if self.tokenisation.index_word[predicted_id] == '<end>':
                    return result, attention_plot

                dec_input = tf.expand_dims([predicted_id], 0)

            attention_plot = attention_plot[:len(result), :]
            return result, attention_plot
        
    def save_checkpoint(self, checkpoint_dir = './training_checkpoints'):
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                         encoder=self.encoder,
                                         decoder=self.decoder)
        
    def save_checkpoint2(self, checkpoint_dir = './training_checkpoints2'):
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        model_var = self.encoder.variables + self.decoder.variables + self.optimizer.variables()
        tf.contrib.eager.Saver(model_var).save(self.checkpoint_prefix)
        
    def load_checkpoint(self, checkpoint_dir = './training_checkpoints'):
        return self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
