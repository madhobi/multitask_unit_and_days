import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sn

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Bidirectional, Dropout, LayerNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from keras.models import Model
from keras.layers import Input
from keras.layers import Concatenate
from keras.utils import to_categorical
from keras.utils import plot_model

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings('ignore')

class MultiTaskModel:
    
    config_create_model = {}
    config_train_model = {}
    model_mt = None
    
    def __init__(self, config_create_model, config_train_model):
        
        self.set_config_create_model(config_create_model)
        self.set_config_train_model(config_train_model)        
        
        
    def create_model_multitask(self, pdemo_shape, max_unit_len, total_units):

        config = self.config_create_model
        
        input_pdemo = Input(shape=pdemo_shape,)
        normalization_layer = LayerNormalization()(input_pdemo)
        hidden_layer_pdemo = Dense(config['pdemo_hidden_dim'], activation='relu')(normalization_layer)        

        input_seq_unit = Input(shape=(max_unit_len))
        embedding_layer_unit = Embedding(total_units+1, config['embed_dim'], input_length=max_unit_len)(input_seq_unit)
        # lstm_layer_unit = LSTM(100)(embedding_layer_unit)
        gru_layer_unit = GRU(config['gru_unit'])(embedding_layer_unit)
        # gru_layer_unit = GRU(100)(input_seq_unit)
        hidden_layer_unit = Dense(config['unit_hidden_dim'], activation='relu')(gru_layer_unit)
        

        concate_layer = Concatenate()([hidden_layer_pdemo, hidden_layer_unit])
        # concate_layer = Concatenate()([dropout_layer_pdemo, dropout_layer_unit])
        dropout_layer = Dropout(config['dropout'])(concate_layer)        

        hidden_layer_unit2 = Dense(config['reg_hidden_dim'], activation='relu')(dropout_layer)        
        hidden_layer_days4 = Dense(config['classifiy_hidden_dim'], activation='relu')(dropout_layer)
        
        output_layer_days_remaining = Dense(1, name='output_days_remaining')(hidden_layer_days4)
        output_layer_next_unit = Dense(total_units, activation='softmax', name='output_next_unit')(hidden_layer_unit2)

        model_mt = Model(inputs=[input_pdemo, input_seq_unit], outputs=[output_layer_days_remaining, output_layer_next_unit] )        
        
        
        model_mt.compile(optimizer=config['optimizer'],
                        loss={'output_days_remaining': config['loss_reg'], 'output_next_unit':config['loss_classify']},
                         loss_weights={'output_days_remaining':config['loss_weight_reg'], 'output_next_unit':config['loss_weight_classify']},
                        metrics={'output_days_remaining':config['metrics_reg'],
                                'output_next_unit':config['metrics_classify']})

        print('Defined multi-task model')
        
        # set_model(model_mt)

        return model_mt


    def train_model_multitask(self, model_mt, pdemos, input_seq_unit, y_days_remaining, y_nextunit_cat):

        config = self.config_train_model
        history = model_mt.fit(x=(pdemos, input_seq_unit), y=(y_days_remaining, y_nextunit_cat), batch_size=config['batch_size'],
                                epochs=config['epochs'], verbose=config['verbose'], validation_split=config['val_split'])
        print('Completed fitting the multitask model!')

        epochs = history.epoch
        hist = pd.DataFrame(history.history)  

        return epochs, hist
    
    def set_config_create_model(self, config_create_model):
        
        self.config_create_model=config_create_model
    
    def set_config_train_model(self, config_train_model):
        
        self.config_train_model=config_train_model
    
    def set_model(self, model_mt):
        
        self.model_mt = model_mt
        
    def get_model(self):
        
        return self.model_mt
    
    def save_model(self, savepath, save_filename):
        
        self.model_mt.save(savepath+save_filename)