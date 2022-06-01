#Custom Transformer that extracts columns passed as argument to its constructor
import pandas as pd
import numpy as np
import random
from string import punctuation
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import pickle
import torch
from transformers import AutoTokenizer, BertTokenizer
from text_normalization import TextNormalization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class TextTransformer( BaseEstimator, TransformerMixin ):

    #Class Constructor
    def __init__( self, truncate=None):
        self.truncate = truncate

    #Return self
    def fit( self, X, y = None ):
        return self
    
    @staticmethod
    def truncate(s):
        s = str(s)
        return s[:200]
    
    @staticmethod
    def remove_punctuation(s):
        for p in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~':
            s = s.replace(p, ' ')
        if len(s) == 0:
            return 'na'
        return s

    #Preparing processed texts to be used in model
    def prepare_for_model(self, X):
        return X

    #Method that describes transformer actions
    def transform( self, X, y = None ):
        #lowercase, normalize and preprocessing
        X = pd.Series(X)
        preprocessed = X.apply(self.remove_punctuation).apply(self.remove_single_char).str.lower().apply(self.remove_stopwords).apply(self.truncate)
        preprocessed =  preprocessed.apply(self.normalizer.normalize)
        return self.prepare_for_model(preprocessed)    
    
class NNTransformer(TextTransformer):
    def __init__(self, normalize=False, seq_length=None):
        self.normalize = normalize
        super(NNTransformer, self).__init__(normalize)
        self.seq_length = seq_length
        
    def fit( self, X, y = None ):
        self.tokenizer = Tokenizer()
        X = pd.Series(X)
        X = X.apply(self.remove_punctuation).apply(self.remove_single_char).str.lower().apply(self.remove_stopwords).apply(self.truncate)
        self.tokenizer.fit_on_texts(X)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        if self.seq_length == None:
            self.max_length = X.str.split().apply(len).agg(max)
        else:
            self.max_length = self.seq_length+0
        return self
    
    def prepare_for_model(self, X):
        i = 20000
        texts_numeric = []
        texts_pad = []
        while i<len(X):
            texts_numeric_partial = self.tokenizer.texts_to_sequences(X[i-20000:i])
            texts_numeric.extend(texts_numeric_partial)
            texts_pad.extend(pad_sequences(texts_numeric_partial, self.max_length))
            i += 20000
        texts_numeric_partial = self.tokenizer.texts_to_sequences(X[i-20000:len(X)])
        texts_numeric.extend(texts_numeric_partial)
        texts_pad.extend(pad_sequences(texts_numeric_partial, self.max_length))
        return np.array(texts_pad)
    
        
class BertTransformer(TextTransformer):
    def __init__(self, tokenizer_path, max_length=50, normalize=False):
        self.normalize = normalize
        super(BertTransformer, self).__init__(normalize)
        self.tokenizer_path = tokenizer_path
        with open(self.tokenizer_path, 'rb') as file:
            self.tokenizer = pickle.load(file)
        self.max_length = max_length
        
    def fit( self, X, y = None ):
        return self
        
    def prepare_for_model(self, X):
        input_ids = []
        attention_masks = []
        for text in X:
            encoded_dict = self.tokenizer.encode_plus(
                                text,                     
                                add_special_tokens = True,
                                max_length = self.max_length,      
                                pad_to_max_length = True,
                                return_attention_mask = True, 
                                return_tensors = 'pt',
                           )

            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return input_ids, attention_masks, X
    
class T5Transformer(TextTransformer):
    def __init__(self, tokenizer_path, max_length=50, normalize=False):
        self.normalize = normalize
        super(T5Transformer, self).__init__(normalize)
        self.tokenizer_path = tokenizer_path
        with open(self.tokenizer_path, 'rb') as file:
            self.tokenizer = pickle.load(file)
        self.max_length = max_length
        
    def fit( self, X, y = None ):
        return self
        
    def prepare_for_model(self, X):
        input_ids = []
        attention_masks = []
        input_ids_labels = []
        attention_masks_labels = []

        for text in X:
            encoded_dict = self.tokenizer.encode_plus(
                                text,                     
                                add_special_tokens = True,
                                max_length = self.max_length,      
                                pad_to_max_length = True,
                                return_attention_mask = True, 
                                return_tensors = 'pt',
                       )

            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)


        return input_ids, attention_masks, X
        