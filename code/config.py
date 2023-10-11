import tensorflow as tf

year = 2018
age_cat = 'adult'
filters = {
    'adm_year':year, 
    'age_cat':age_cat
}

filters_pid = {
    'adm_year':year, 
    'age_cat':age_cat,
    'entry_unit':1
}

filepaths = {
    'parent_data_folder': '../data/',
    'bf_folder':'bedflow/',
    'bf_prefix':'bedflow_with_daily_med_encoded_year_',
    'bf_filetype':'.pickle',    
    'pdemo_folder':'pdemos/',
    'pdemo_file_prefix':'pdemos_',
    'pdemo_filetype':'.pickle',    
    'pid_file_prefix':'pid_',    
    'pid_filetype':'.pickle',
    'diag_folder':'diag/',
    'diag_filename':'encoded_diagnosis',
    'diag_filetype':'.pickle',
    'unit_filename':'nhsnunitid_with_tokindex_and_unitcat',
    'unit_filetype':'.csv',
    'tok_filename':'tokenizer_nhsnunitlist',
    'tok_filetype':'.pickle'
}

config_create_model = {
    'optimizer':'adam', 
    'pdemo_hidden_dim':100, 
    'unit_hidden_dim':100,
    'dropout':0.4,
    'embed_dim':100,
    'gru_unit':50,
    'reg_hidden_dim':30, 
    'classifiy_hidden_dim':30,
    'loss_reg':'mse',
    'loss_classify':'categorical_crossentropy', 
    'loss_weight_reg':0.2,
    'loss_weight_classify':0.8,
    'metrics_reg':tf.keras.metrics.MeanAbsoluteError(),
    'metrics_classify':tf.keras.metrics.TopKCategoricalAccuracy(k=2)

}

config_train_model = {
    'batch_size':32, 
    'epochs':10,
    'verbose':2,
    'val_split':0.1
}
