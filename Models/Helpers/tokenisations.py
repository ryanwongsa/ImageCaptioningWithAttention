import tensorflow as tf
import numpy as np
import pickle

# tf.enable_eager_execution()

class Tokenisations(object):
    def __init__(self, load_tokens=False):
        self.token_file = "tokenizer.pkl"
        self.token_dict = "token_dict.npy"
        self.tokenizer = None
        if load_tokens==True:
            with open(self.token_file, 'rb') as input:
                self.tokenizer = pickle.load(input)
            token_dict = np.load(self.token_dict).item()
            
            self.max_length = token_dict["max_length"]
            self.index_word = token_dict["index_word"]
            
    
    def prepare_training_tokens(self, captions, top_words_keep_count=5000):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=top_words_keep_count, 
            oov_token="<unk>", 
            filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        self.tokenizer.fit_on_texts(captions)
        self.train_seqs = self.tokenizer.texts_to_sequences(captions)
        
        self.tokenizer.word_index = {
            key:value for key, value in self.tokenizer.word_index.items() if value <= top_words_keep_count}
        
        self.tokenizer.word_index[self.tokenizer.oov_token] = top_words_keep_count + 1
        self.tokenizer.word_index['<pad>'] = 0
        
        self.train_seqs = self.tokenizer.texts_to_sequences(captions)
        self.cap_vector = tf.keras.preprocessing.sequence.pad_sequences(self.train_seqs, padding='post')
        self.max_length = self.calc_max_length(self.train_seqs)

        self.index_word = {value:key for key, value in self.tokenizer.word_index.items()}
   
    def calc_max_length(self, tensor):
        return max(len(t) for t in tensor)
    
    def save_tokeniser(self):
        with open(self.token_file, 'wb') as output:
            pickle.dump(self.tokenizer, output, pickle.HIGHEST_PROTOCOL)
        
        token_dict = {'max_length':self.max_length, 'index_word':self.index_word}
        np.save("token_dict.npy", token_dict)