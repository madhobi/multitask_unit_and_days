import pandas as pd
import DataManager as dm
import tensorflow as tf
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

def get_next_unit_pred_cat_to_val(y_nu_pred_raw, y_nextunit_cat_test, k=2):

    # k=2
    y_nu_pred = y_nu_pred_raw
    y_cat = np.argmax(y_nextunit_cat_test, axis=1)
    first_match = 0
    second_match = 0
    no_match = 0
    y_nu_match = []
    for ind in range(0, len(y_nu_pred)):
        # print(item)
        y_pred = np.argmax(y_nu_pred[ind])
        y_pred_top2 = np.argpartition(y_nu_pred[ind], -k)[-k:]

        if(y_cat[ind] == y_pred):
            first_match += 1
            y_nu_match.append(1)
        elif((y_cat[ind] == y_pred_top2[0]) | (y_cat[ind] == y_pred_top2[1])):
            second_match += 1
            y_nu_match.append(1)
        else:
            no_match += 1
            y_nu_match.append(0)

        # break

    print('first match: ', str(first_match))
    print('second match: ', str(second_match))
    print('no match: ', str(no_match))
    match_acc = (first_match+second_match)*1.0/len(y_nu_pred)
    print('match acc: ', str(match_acc))
    
    # return first_match, second_match, no_match
    return y_nu_match

def prepare_test_for_plot(test_all_feature, test_cols, key_cols, y_dr_pred, y_nu_pred_raw, y_nextunit_cat_test):
    
    k=2
    df_test = test_all_feature[key_cols+test_cols].copy()
    df_test['days_remaining_pred'] = y_dr_pred
    y_nu_match = get_next_unit_pred_cat_to_val(y_nu_pred_raw, y_nextunit_cat_test, k)
    df_test['pred_next_unit_match'] = y_nu_match
    df_test['err'] = abs(df_test['days_remaining'] - df_test['days_remaining_pred'])
    
    return df_test

def plotfunc(df, xcol, ycol, xlabel, ylabel, title, savename, orient='h', color='blue'):
    plt.figure(figsize=(16,12))
    b = sn.barplot(data=df, y=ycol, x=xcol, orient=orient, color=color)
    plt.xlabel(xlabel, fontsize=22, fontweight='bold')
    plt.ylabel(ylabel, fontsize=22, fontweight='bold')
    plt.title(title, fontsize=24, fontweight='bold')
    b.set_yticklabels(b.get_ymajorticklabels(), size = 20, weight='bold')
    b.set_xticklabels(b.get_xmajorticklabels(), size = 20, weight='bold')
    plt.grid()
    plt.savefig('../images/'+savename+'.png', bbox_inches='tight')
    # plt.show()    
    plt.close()
    
def plot_acc(df_test, unitname, age_cat, year, color='blue'):
    
    # if(acc):
    df_acc = df_test.groupby(['nhsnunitid'])['pred_next_unit_match'].agg(['sum','count']).reset_index()
    df_acc['acc'] = df_acc['sum'] / df_acc['count']
    print(df_acc.head())
    df_acc = pd.merge(unitname, df_acc, on=['nhsnunitid'])
    print(df_acc.shape)
    df_acc = df_acc[df_acc['count']>=100].copy()
    print(df_acc.shape)
    xcol = 'acc'
    ycol = 'nhsnunitname'
    xlabel = 'Next Unit Prediction Accuracy'
    ylabel = 'Current Unit'
    title = 'Next Unit Prediction Accuracy From '+age_cat.capitalize()+' Units'
    savename = 'next_unit_pred_accuracy_from_'+age_cat+'_hospital_unit_year'+str(year)
    plotfunc(df_acc, xcol, ycol, xlabel, ylabel, title, savename, color=color)
        # df_acc3.head(3)
    # if(err):   
def plot_err(df_test, unitname, age_cat, year, color='orange'):
    
    df_err = df_test.groupby(['nhsnunitid'])['err'].agg(['sum','count']).reset_index()
    df_err['mean_err'] = df_err['sum'] / df_err['count']
    print(df_err.head())
    df_err = pd.merge(unitname, df_err, on=['nhsnunitid'])
    print(df_err.shape)
    df_err = df_err[df_err['count']>=100].copy()
    print(df_err.shape)
    xcol = 'mean_err'
    ycol = 'nhsnunitname'
    xlabel = 'Mean Absolute Error'
    ylabel = 'Current Unit'
    title = 'Mean Absolute Error of Predicting Remaining Days in '+age_cat.capitalize()+' Units'
    savename = 'mae_from_'+age_cat+'_hospital_unit_year'+str(year)
    plotfunc(df_err, xcol, ycol, xlabel, ylabel, title, savename, color=color)

        
def plot_model_pred(test_all_feat, y_dr_pred, y_nu_pred, y_nextunit_cat, train_year, test_filters, unitname):
    
    test_year = test_filters['adm_year']
    age_cat = test_filters['age_cat']
    test_cols = ['nhsnunitid','day_since_admission','days_remaining','next_unit']
    key_cols = ['patientid', 'admissionid']
    df_test = prepare_test_for_plot(test_all_feat, test_cols, key_cols, y_dr_pred, y_nu_pred, y_nextunit_cat)
    df_test.to_pickle('../save_test_result/test_result_adult_train_'+str(train_year)+'_test_'+str(test_year)+'.pickle')
    print('saving df_test shape: ', df_test.shape)
    print(df_test.head())

    print('calling plot test function for acc..')
    plot_acc(df_test, unitname, age_cat, test_year)
    print('calling plot test function for err..')
    plot_err(df_test, unitname, age_cat, test_year)
