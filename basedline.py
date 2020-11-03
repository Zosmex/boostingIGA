
import pandas as pd
import numpy as np
import time
from ipykernel import kernelapp as app
from scipy import stats
import json

import sklearn.metrics as metrics
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, ShuffleSplit
from sklearn.utils import resample
from sklearn import linear_model
from sklearn.feature_selection import RFE
import sklearn_relief as relief

from tensorflow import keras
from keras import Sequential, optimizers
from keras.models import Model, model_from_json, Sequential
from keras.layers import Dense, Input, BatchNormalization, Dropout
from keras.callbacks import ReduceLROnPlateau

#########################################################################

def create_model(n):
    inputs = Input(shape=(n,))
    h_1 = Dense(50, kernel_initializer='he_normal', activation='linear',bias_initializer='zeros')(inputs)
    outputs = Dense(1, kernel_initializer='he_normal', activation='linear',bias_initializer='zeros')(h_1)
    model = Model(inputs=inputs, outputs=outputs)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)
    return model


############################### Evaluation Function ##########################################

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) *100

def mean_arctan_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    AAPE = np.arctan(np.abs((y_true - y_pred)) / np.abs(y_true))

    return np.mean(AAPE) * 100

################################ Training Function #########################################

def train_Shuffle(X, y, epochs, batch_size, n_split = 5):
    ss_maape = []
    ss_rmse = []

    SS = ShuffleSplit(n_splits= n_split, train_size = .6, random_state=101)
    
    for train, test in SS.split(X):
        model = create_model(X.shape[1])
        pred = model.predict(X[test])
        
        maape = mean_arctan_absolute_percentage_error(y.loc[test], pred)
        rmse = np.sqrt(metrics.mean_squared_error(y.loc[test], pred))
            
        ss_maape.append(maape)
        ss_rmse.append(rmse)
        
    return model, ss_maape, ss_rmse

#############################################################
#                   Based Line Model                        #
#############################################################


################# Recursive Feature Elimination ##########################

def RFE_model(X, y, n_epochs, n_batch_size):
    Best_score = {'RMSE':0, 
                'MAAPE':0, 
                'Iteration':0,
                'Features':[]
                }
    MAAPE_RFE = []
    RMSE_RFE = []

    model = linear_model.LinearRegression()

    for k in range(0, X.shape[1]):
        print("Ite :", k+1)
        rfe = RFE(model,k+1)
        X_rfe = rfe.fit_transform(X, y)
        temp = pd.Series(rfe.support_, index = X.columns)
        selected_features_rfe = temp[temp==True].index
        print(selected_features_rfe)

        _, RFE_maape, RFE_rmse = train_Shuffle(X_rfe, y,
                                                   epochs = n_epochs , 
                                                   batch_size = n_batch_size
                                                   )

        print("Avg MAAPE: %.2f%% (+/- %.2f%%)" % (np.mean(RFE_maape), np.std(RFE_maape)))
        print("Avg RMSE: %.2f%% (+/- %.2f%%)" % (np.mean(RFE_rmse), np.std(RFE_rmse)))
    
        MAAPE_RFE.append(np.mean(RFE_maape))
        RMSE_RFE.append(np.mean(RFE_rmse))
    
        if(np.mean(RFE_maape) < Best_score['MAAPE']):
            print('New Record!')
            Best_score['RMSE'] = np.mean(RFE_rmse)
            Best_score['MAAPE'] = np.mean(RFE_maape)
            Best_score['Iteration'] = k
            Best_score['Features'] = selected_features_rfe
        
    return MAAPE_RFE, RMSE_RFE, Best_score

#################### RReliefF ##########################

def RLF_model(X, y, n_epochs, n_batch_size):
    y_list = []
    for i in y.values:
        y_list.append(i[0])
    y_list = np.asarray(y_list)

    Best_score = {'RMSE':0, 
                'MAAPE':0, 
                'Iteration':0,
                'Features':[]
                }
    MAAPE_RLF = []
    RMSE_RLF = []
    sel_RLF=[]

    for k in range(0,X.shape[1]):
        print('Iteration: ', k+1)
        r = relief.RReliefF(n_jobs=1, n_features = k+1)
        X_RLF = r.fit_transform(X.values, y_list)
        sel_RLF.append(X_RLF)

        _, RLF_maape, RLF_rmse= train_Shuffle(X_RLF, y, 100,16)
    
        print("Avg MAAPE: %.2f%% (+/- %.2f%%)" % (np.mean(RLF_maape), np.std(RLF_maape)))
        print("Avg RMSE: %.2f%% (+/- %.2f%%)" % (np.mean(RLF_rmse), np.std(RLF_rmse)))
    
        RMSE_RLF.append(np.mean(RLF_rmse))
        MAAPE_RLF.append(np.mean(RLF_maape))

        if(np.mean(RLF_maape) < Best_score['MAAPE']):
            print('New Record!')
            Best_score['RMSE'] = np.mean(RLF_rmse)
            Best_score['MAAPE'] = np.mean(RLF_maape)
            Best_score['Iteration'] = k
            Best_score['Features'] = X_RLF

    return MAAPE_RLF, RMSE_RLF, Best_score