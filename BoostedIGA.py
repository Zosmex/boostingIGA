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


#########################################################################

def update_weight(y_pred, y_true, D_t):
    newD_t = []
    e_t = [np.abs(float(fx)-float(y)) for fx,y in zip(y_pred, y_true)]
    norm_fac = np.max(e_t)
    L_t = [(np.power((e/norm_fac),2)) for e in e_t]
    L_d = [L*w for L,w in zip(L_t,D_t)]
    B_t = np.sum(L_d)/(1 - np.sum(L_d))
    for i in range(0, len(D_t)):
        newD_t.append(D_t[i] * (np.power(B_t,(1 - L_t[i]))))
    Z = np.sum(newD_t)
    newD_t[:] = [d/Z for d in newD_t]
    return newD_t

#########################################################################

def IGA(w,v,x,D_t, columns):
    S = []
    # connection weight through the different hidden node
    cw = w * np.transpose(v.reshape(-1)) # Wij * Vjk
    cw_h = np.dot(x,w) # sum Xi * Wij
    for p in range(0,x.shape[0]): # input value (X)
        sx = []
        for i in range(0,w.shape[0]): # input-hidden weight (W), number of feature
            s = 0
            for j in range(0,v.shape[0]): # hidden-output weight (V)
                s += (cw[i][j] * x[p][i]) / cw_h[p][j]
            sx.append(s) # shape = number of features
        sx = ((1-D_t[p]) * (sx / np.sum(np.abs(sx))))
        S.append(sx)# examples
    Si = np.sum(S, axis=0)
    u = Si / len(S)
    v = np.sqrt(np.sum(np.power(S - u,2), axis = 0)) / len(S) #variance
    d = {'Feature': columns, 's-coef': u}
    factor_imp = pd.DataFrame(data=d)
    factor_imp.sort_values(by= 's-coef', ascending= False, inplace = True)

    return(factor_imp)

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
        pred = model.predict(X.loc[test])
        
        maape = mean_arctan_absolute_percentage_error(y.loc[test], pred)
        rmse = np.sqrt(metrics.mean_squared_error(y.loc[test], pred))
            
        ss_maape.append(maape)
        ss_rmse.append(rmse)
        
    return model, ss_maape, ss_rmse

#########################################################################

def BoostedIGA(X,y,n_epochs, n_batch_size):
    Best_score = {'RMSE':0, 
                'MAAPE':0, 
                'Iteration':0,
                'Features':[]
                }
    D_t = []
    MAAPE_score = []
    RMSE_score = []

    #init weight
    D_t.append([1/X.shape[0]] * X.shape[0])

    for t in range(0,X.shape[1]):
        print('Feature Selection x Boosting >>> Iteration: %d' % (t+1))
        if (t == 0):
            Xs = pd.DataFrame(data = X, columns = X.columns)
        
        nw1 = create_model(Xs.shape[1])

        #feature selection
        nw1.fit(
                Xs, y, 
                epochs = n_epochs, 
                batch_size= n_batch_size,
                verbose = 0
            )
        W = nw1.layers[1].get_weights()[0]
        V = nw1.layers[2].get_weights()[0]
        print("Feature Selection Process: %d" % (t+1))
        ranked_features = IGA(W, V, Xs.values, D_t[t], Xs.columns) 
        ranked_features.reset_index(inplace = True)
        Xs.drop([ranked_features['Feature'][0]], axis = 1, inplace = True)
        X_sel = X.drop(Xs.columns, axis = 1)
        print("Selected Feature: ",(X_sel.columns))
    
        #Boosting
        nw2, ss_maape, ss_rmse  = train_Shuffle(X_sel,y, 
                                                batch_size=n_epochs, 
                                                epochs=n_batch_size,
                                                n_split = 5
                                            )
    
        print("Avg MAAPE: %.5f%% (+/- %.5f%%)" % (np.mean(ss_maape), np.std(ss_maape)))
        print('Avg RMSE: %.5f%% (+/- %.5f%%)' % (np.mean(ss_rmse), np.std(ss_rmse)))
    
        MAAPE_score.append(np.mean(ss_maape))
        RMSE_score.append(np.mean(ss_rmse))

        D_t.append(update_weight(nw2.predict(X_sel), y.values, D_t[t]))
    
        if(np.mean(ss_maape) < Best_score['MAAPE']):
            print('New record!')

            Best_score['MAAPE'] = np.mean(ss_maape)
            Best_score['RMSE'] = np.mean(ss_rmse)
            Best_score['Iteration'] = t
            Best_score['Features'] = X_sel.columns

    return MAAPE_score, RMSE_score, Best_score
