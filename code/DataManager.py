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


class DataManager:
    
    # we need four dataframes
    # 1. bedflow with daily medication
    # 2. pdemo encoded
    # 3. diagnostic 
    # 4. training and testing patientids
    
    filepaths = {}
    filters = {}
    bf_med = None
    diag = None
    pdemo_enc = None
    pdemo_pid = None
    units = None
    tok_unit = None
    total_units=None
    key_cols = ['patientid', 'admissionid']
    # test_size = 0.3
    max_seq_len = 30
    target_cols=['days_remaining', 'next_unit']
    seq_col='nhsnunitid_cum'
    cols_to_del=['nhsnunitid']
    train=True

    
    def __init__(self, filepaths, filters, train=True):
        
        self.filepaths = filepaths
        self.filters = filters
        self.train = train
        
    def read_files(self):
        
        filepaths = self.filepaths
        filters = self.filters
        
        bf_filepath = (filepaths['parent_data_folder']
           +filepaths['bf_folder']
           +filepaths['bf_prefix']
           +str(filters['adm_year'])
           +filepaths['bf_filetype'])
        # bf = pd.read_pickle(filepaths['bedflow_filepath'])
        bf = pd.read_pickle(bf_filepath)
        self.bf_med = bf.copy()
        print('bf shape: ', bf.shape)
        
        diag_filepath = (filepaths['parent_data_folder']
                +filepaths['diag_folder']
                +filepaths['diag_filename']
                +filepaths['diag_filetype'])
        # diag = pd.read_pickle(filepaths['diag_filepath'])
        diag = pd.read_pickle(diag_filepath)
        self.diag = diag.copy()
        print('diag shape: ', diag.shape)

        pdemo_filepath = (filepaths['parent_data_folder']
                +filepaths['pdemo_folder']
                +filepaths['pdemo_file_prefix']
                +str(filters['age_cat']) + '_year_'                
                +str(filters['adm_year'])
                +filepaths['pdemo_filetype'])
        # pdemo_enc = pd.read_pickle(filepaths['pdemo_enc_filepath'])
        pdemo_enc = pd.read_pickle(pdemo_filepath)
        self.pdemo_enc = pdemo_enc.copy()
        print('pdemo enc shape: ', pdemo_enc.shape)

        pid_filepath = (filepaths['parent_data_folder']                
                +filepaths['pid_file_prefix']
                +str(filters['age_cat']) + '_year_'
                +str(filters['adm_year'])
                +filepaths['pid_filetype'])
        # pdemo_pid = pd.read_pickle(filepaths['pdemo_pid_filepath'])
        pdemo_pid = pd.read_pickle(pid_filepath)
        self.pdemo_pid = pdemo_pid.copy()
        print('pdemo pid shape: ', pdemo_pid.shape)        

        unit_filepath = (filepaths['parent_data_folder']
                +filepaths['unit_filename']
                +filepaths['unit_filetype'])
        # units = pd.read_csv('./data/nhsnunitid_with_tokindex_and_unitcat.csv')
        # units = pd.read_csv(unit_filepath)
        # self.units = units.copy()
        # print(units.shape)

        tok_filepath = (filepaths['parent_data_folder']
                +filepaths['tok_filename']
                +filepaths['tok_filetype'])
        # with open('./data/tokenizer_nhsnunitlist.pickle', 'rb') as handle:
        with open(tok_filepath, 'rb') as handle:
            tok_unit = pickle.load(handle)
        self.tok_unit = tok_unit
        self.total_units = len(tok_unit.word_index)+1
        # total_units
        
        # udict = dict(zip(units.nhsnunitid, units.nhsnunitname))
        
    def get_pid(self, filters=None, test_size=None):
    
        pid = self.pdemo_pid.copy()
        if(filters==None):
            filters=self.filters
        print('pid shape original: ', pid.shape)
        # for key in filters:
        #     print('filtering for: ', key, '->', filters[key])
        #     pid = pid[(pid[key] == filters[key])].copy()
        #     print('shape after filtering: ', pid.shape)

        pid = pid[self.key_cols].copy()
        
        if(test_size != None):
            train_pid, test_pid = train_test_split(pid, test_size=test_size, random_state=42)
            print('train pid size: ', train_pid.shape)
            print('test pid size: ', test_pid.shape)
        
            return train_pid, test_pid
        else:
            return pid
    
    def join_datasets(self, bf_med, pdemo, diag, pid, key_cols):
    
        key_cols = self.key_cols
        print('bedflow shape: ', bf_med.shape)
        bf_working = pd.merge(pid, bf_med, on=key_cols)
        print('bedflow shape for pids: ', bf_working.shape)

        print('diagnostic shape: ', diag.shape)
        diag_working = pd.merge(pid, diag, on=key_cols)
        print('diagnostic shape for pids: ', diag_working.shape)

        print('pdemo shape: ', pdemo.shape)
        pdemo_working = pd.merge(pid, pdemo, on=key_cols)
        print('pdemo shape for pids: ', pdemo_working.shape)

        # join pdemo with diag    
        pdemo_diag = pd.merge(pdemo_working, diag_working, on=key_cols)
        print('shape after joining pdemo and diagnostic: ', pdemo_diag.shape)

        # join pdemo_diag with bbedflow
        df_feature = pd.merge(pdemo_diag, bf_working, on=key_cols)
        print('joining demo and diag data with bedflow shape: ', df_feature.shape)

        return df_feature
    
    def prepare_padded_seq(self, df_feature, seq_col, tok, max_seq_len=None):
        
        # tokenize and pad the unit sequences
        tokenized_seq = tok.texts_to_sequences(map(str, df_feature[seq_col]))
        if(max_seq_len == None):
            max_seq_len = max([len(x) for x in df_feature[seq_col]])
        print('max_seq_len: ', str(max_seq_len))
        padded_seq = np.array(pad_sequences(tokenized_seq, maxlen=max_seq_len, padding='pre'))

        return padded_seq, max_seq_len

    def prepare_categorical_target(self, df_feature, target_col, tok):

        y_label = df_feature[target_col]
        # print(y_label)
        y_label_tokenized = tok.texts_to_sequences(map(str, y_label))
        # print(y_label_tokenized[0:5])
        num_classes = len(tok.word_index)+1
        print('total labels: ', str(num_classes))
        y_label_cat = to_categorical(y_label_tokenized, num_classes=num_classes)
        # print(y_label_cat[0:5])

        return y_label_cat

    def create_dataset_multitask(self, pid, max_seq_len=None):

        # bf_med = self.bf.copy()
        # pdemo = self.pdemo_enc.copy()
        # diag = self.diag.copy()
        # config = self.config
        key_cols = self.key_cols
        seq_col = self.seq_col
        target_cols = self.target_cols
        cols_to_del = self.cols_to_del
        
        df_feature = self.join_datasets(self.bf_med, self.pdemo_enc, self.diag, pid, key_cols)

        Xfeats = df_feature.copy()
        Xseq = df_feature[seq_col]

        y_days_remaining = df_feature[target_cols[0]]
        tok_unit = self.tok_unit
        y_nextunit_cat = self.prepare_categorical_target(df_feature, target_cols[1], tok_unit)   

        # now preparing the sequences        
        # if(self.train==True):
        if(max_seq_len is None):
            padded_seq_unit, max_seq_len = self.prepare_padded_seq(df_feature, seq_col, tok_unit)
            self.max_seq_len = max_seq_len
        else: # else this is test set
            print('creating dataset for model testing..')
            padded_seq_unit, max_seq_len = self.prepare_padded_seq(df_feature, seq_col, tok_unit, max_seq_len)

        # cols_to_del = config['cols_to_del']+config['key_cols']+config['target_cols']
        cols_to_del = cols_to_del+key_cols+target_cols
        cols_to_del.append(seq_col)
        print('cols to del: ', cols_to_del)
        Xfeats.drop(cols_to_del, axis=1, inplace=True)
        print('Xfeats shape after dropping columns: ', Xfeats.shape)
        
        X = (Xfeats, padded_seq_unit)
        y = (y_days_remaining, y_nextunit_cat)
        
        return df_feature, X, y
    
    def create_dataset_singletask_days_remaining(self, bf_med, pdemo, diag, pid):

        df_feature = join_datasets(bf_med, pdemo, diag, pid, key_cols)

        Xfeats = df_feature.copy()
        Xseq = df_feature[seq_col]

        y_days_remaining = df_feature[target_cols[0]]
        y_nextunit_cat = prepare_categorical_target(df_feature, target_cols[1], tok_unit)   

        # now preparing the sequences        
        if(self.train==True):
            padded_seq_unit, max_seq_len = prepare_padded_seq(df_feature, seq_col, tok_unit)
            self.max_seq_len = max_seq_len
        else: # else this is test set
            padded_seq_unit, max_seq_len = prepare_padded_seq(df_feature, seq_col, tok_unit, self.max_seq_len)

        cols_to_del = cols_to_del+key_cols+target_cols
        cols_to_del.append(seq_col)
        print('cols to del: ', cols_to_del)
        Xfeats.drop(cols_to_del, axis=1, inplace=True)
        print('Xfeats shape after dropping columns: ', Xfeats.shape)
        
        X = (Xfeats, padded_seq_unit)
        y = (y_days_remaining, y_nextunit_cat)
        
        return df_feature, X, y
    
    def create_dataset_singletask_next_unit(self, bf_med, pdemo, diag, pid):

        df_feature = join_datasets(bf_med, pdemo, diag, pid, key_cols)

        Xfeats = df_feature.copy()
        Xseq = df_feature[seq_col]

        y_days_remaining = df_feature[target_cols[0]]
        y_nextunit_cat = prepare_categorical_target(df_feature, target_cols[1], tok_unit)   

        # now preparing the sequences        
        if(self.train==True):
            padded_seq_unit, max_seq_len = prepare_padded_seq(df_feature, seq_col, tok_unit)
            self.max_seq_len = max_seq_len
        else: # else this is test set
            padded_seq_unit, max_seq_len = prepare_padded_seq(df_feature, seq_col, tok_unit, self.max_seq_len)

        cols_to_del = cols_to_del+key_cols+target_cols
        cols_to_del.append(seq_col)
        print('cols to del: ', cols_to_del)
        Xfeats.drop(cols_to_del, axis=1, inplace=True)
        print('Xfeats shape after dropping columns: ', Xfeats.shape)
        
        X = (Xfeats, padded_seq_unit)
        y = (y_days_remaining, y_nextunit_cat)
        
        return df_feature, X, y
    
    
    
    





