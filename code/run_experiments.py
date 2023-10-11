import tensorflow as tf
import MultiTaskModel as mt
import DataManager as dm
import config as cf
from plot_functions_from_model_predictions import *
from keras.utils import plot_model
from sklearn import metrics
from datetime import datetime



def get_datahandler(filters=None):
    print('Preparing data handler..')
    if(filters is None):
        print('using the default filters from config file..')
        filters = cf.filters
    
    # call data handler to get the training data
    datahandler = dm.DataManager(cf.filepaths, filters)
    datahandler.read_files()    
    
    return datahandler

def test_model(datahandler, test_pid, train_year, model=None, model_path=None):
    print('Evaluating the trained model on test data..')    
    
    if(model != None):
        print('testing on the given model..')
        trained_model = model
    elif(model_path != None):
        print('loading saved model from the path')
        model_path = model_path
        loaded_model = tf.keras.models.load_model(model_path)
        trained_model = loaded_model
    else:
        print('No model or model path were given. Exiting test function.')
        return
    
    max_seq_len = (trained_model.layers[0].input_shape[0])[1]
    test_all_feat, test_X, test_y = datahandler.create_dataset_multitask(test_pid, max_seq_len)
    pdemos = test_X[0]
    input_seq_unit = test_X[1]
    y_days_remaining = test_y[0]
    y_nextunit_cat = test_y[1]
    pdemo_shape = pdemos.shape[1]
    max_seq_len = datahandler.max_seq_len#30
    # total_units = datahandler.total_units
    # units = datahandler.units
    # unitname = units[['nhsnunitid', 'nhsnunitname']].copy()    
    
    y_dr_pred, y_nu_pred = trained_model.predict(x=(pdemos, input_seq_unit), batch_size=1)

    mae = metrics.mean_absolute_error(y_days_remaining, y_dr_pred)
    mse = metrics.mean_squared_error(y_days_remaining, y_dr_pred)
    rmse = metrics.mean_squared_error(y_days_remaining, y_dr_pred, squared=False)
    # Print results
    print("mae: ", mae)
    print("mse: ", mse)
    print("rmse: ", rmse)

    y_true = np.argmax(y_nextunit_cat, axis=1)
    y_pred = y_nu_pred
    topkacc = metrics.top_k_accuracy_score(y_true, y_pred, k=2, labels=range(35))
    y_pred1d = np.argmax(y_pred, axis=1)
    acc = metrics.accuracy_score(y_true, y_pred1d)
    print('acc: ', str(acc))
    print('topkacc(k=2): ', str(topkacc))
    print(metrics.classification_report(y_true, y_pred1d))
    
    # print('------------------------------------------------------------')
    # print('--------calling plot function---------')
    
    # plot_model_pred(test_all_feat, y_dr_pred, y_nu_pred, y_nextunit_cat, train_year, datahandler.filters, unitname)
    

def train_model(filters=None, test=False, test_size=None): 
       
    datahandler = get_datahandler(filters)    
    if(test_size is None):
        train_pid = datahandler.get_pid() # call without test size will put the whole data on one set
    else:
        train_pid, test_pid = datahandler.get_pid(test_size=test_size)  # call with test_size will split the data into train and test set
    # print(train_pid.shape)
    train_all_feat, train_X, train_y = datahandler.create_dataset_multitask(train_pid)
    pdemos = train_X[0]
    input_seq_unit = train_X[1]
    y_days_remaining = train_y[0]
    y_nextunit_cat = train_y[1]
    pdemo_shape = pdemos.shape[1]
    max_seq_len = datahandler.max_seq_len#30
    total_units = datahandler.total_units
    
    mymodel = mt.MultiTaskModel(cf.config_create_model, cf.config_train_model)
    # pdemo_shape, max_unit_len, total_units = 357, 30, 35 # random values for testing
    model_mt = mymodel.create_model_multitask(pdemo_shape, max_seq_len, total_units)
    print(model_mt.summary())
    # plot_model(model_mt, to_file='../images/model_mt_train_year_'+str(filters['adm_year'])+'.png', show_shapes=True, show_layer_names=True)    
    epoch, hist = mymodel.train_model_multitask(model_mt, pdemos, input_seq_unit, y_days_remaining, y_nextunit_cat)
    
    mymodel.set_model(model_mt)
    # now = datetime.now() # current date and time
    # timestamp = now.strftime("%m_%d_%Y_%H_%M_%S")
    savepath = '../saved_model/'
    print('..............................................')
    save_filename = 'model_mt_'+filters['age_cat']+'_year_'+str(filters['adm_year'])#+timestamp
    print('saving the trained model with filename', save_filename)
    print('..............................................')
    mymodel.save_model(savepath, save_filename)
    print('saved model in location ', savepath+save_filename )
    
    hist_savepath = '../hist_log/'
    hist_filename = 'hist_mt_'+str(filters['age_cat'])\
                    +'_'+str(filters['adm_year'])+'.csv'
    hist.to_csv(hist_savepath+hist_filename)
    print('saved history as ', hist_savepath+hist_filename)
    if(test==True):
        print('Calling test model function to evaluate model result in test data..')
        # test_model(datahandler,test_pid, model_mt)
        train_year = filters['adm_year']
        test_model(datahandler, test_pid, train_year, model=model_mt)
