# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:17:09 2019

@author: marinara.marcato
"""




from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA



import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

from eval import evaluate_pipeline
from data import get_timestamps
from data import get_actigraph

from sklearn.base import BaseEstimator, TransformerMixin

# ------------------------------------------------------------------------- #
#                          Initializing parameters                          #    
# ------------------------------------------------------------------------- #
dir_base = 'E:\\Study\\Subjects'
subjects = os.listdir(dir_base)[1:]
dcs = ['DC1', 'DC2']
bps = ['Back', 'Chest', 'Neck']
df_ep, df_pos, df_info, df_imu = {}, {}, {}, {}

# ------------------------------------------------------------------------- #
#                               Importing data                              #    
# ------------------------------------------------------------------------- #
get_timestamps(subjects, dcs, dir_base, df_info, df_pos, df_ep)
get_actigraph(subjects, dcs, bps, dir_base, df_imu)
# ------------------------------------------------------------------------- #
#               Combining data to create features dataframe                 #    
# ------------------------------------------------------------------------- #           

#                       Populating 'Position' column                        #
        
# Set window size, window offset and transition time 
w_size = 100
w_offset = timedelta(seconds = .25)
t_time = timedelta(seconds = .25)
feat = ['mean','std', 'median', 'min','max']


df_l2 = []  
for subj in subjects:
    for dc in dcs:      
        # Checking if df_ep and therefore df_pos exist, as well as df_imu exists 
        if df_ep[subj][dc] is not None and df_imu[subj][dc] is not None:
            print(subj, dc, 'Position')
            for (s_time, f_time) in zip( df_pos[subj][dc].index.to_series() + t_time , \
                               df_pos[subj][dc].index.to_series().shift(-1) - t_time):
                
                df_l1 = []        
                
                df_l1.append((df_imu[subj][dc][s_time:f_time].rolling(w_size, center = True).mean()).resample(w_offset).first())
                df_l1.append((df_imu[subj][dc][s_time:f_time].rolling(w_size, center = True).std()).resample(w_offset).first())
                df_l1.append((df_imu[subj][dc][s_time:f_time].rolling(w_size, center = True).median()).resample(w_offset).first())
                df_l1.append((df_imu[subj][dc][s_time:f_time].rolling(w_size, center = True).min()).resample(w_offset).first())
                df_l1.append((df_imu[subj][dc][s_time:f_time].rolling(w_size, center = True).max()).resample(w_offset).first())
                #df_l1.append((df_imu[subj][dc][s_time:f_time].rolling(w_size, center = True).sem()).resample(w_offset).first())
                

                df_l2.append( pd.concat(df_l1, axis = 1, keys = feat, \
                              names = ['Statistics','BodyParts', 'Sensor Axis'])\
                               .assign(Dog = subj, DC = dc, Position = df_pos[subj][dc]['Position'][(s_time-t_time)]))  


df_feat = pd.DataFrame()              
df_feat = pd.concat(df_l2)

#                       Populating 'Type' column                         #
pos_type = {'Walking': 'Dynamic', 
            'W-Sniffing floor': 'Dynamic', 
            'Standing':'Static', 
            'Sitting':'Static', 
            'Lying down': 'Static',
            'Jumping up': 'Dynamic', 
            'Jumping down':'Dynamic', 
            'Body shake':'Dynamic', 
            'S-Sniffing floor': 'Static', 
            'Pull on leash': 'Dynamic', 
            'Moving': 'Dynamic'}

df_feat['Type'] = df_feat['Position'].map(pos_type)

# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
#                Machine Learning - Label 'Positions'                       #    
# ------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------- #


# ------------------------------------------------------------------------- #
#                            Defining Classes                             #    
# ------------------------------------------------------------------------- #
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names, dtype=None):
        self.attribute_names = attribute_names
        self.dtype = dtype
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_selected = X[self.attribute_names]
        if self.dtype:
            return X_selected.astype(self.dtype).values
        return X_selected.values


        
# Deleting rows with nan
df_feat.dropna(axis = 0, inplace = True)
df_feat = df_feat[df_feat['Position'] != 'Moving']
# Shuffling data
df_feat = df_feat.take(np.random.permutation(len(df_feat))) 


print(df_feat['Position'].value_counts())
df_feat = df_feat[df_feat['Position'] != 'S-Sniffing floor']
df_feat = df_feat[df_feat['Position'] != 'W-Sniffing floor']

df_all_bal = balance_df(df_feat, 'Position')
df_all_bal['Position'].value_counts()

x_all_bal = df_all_bal.loc[:, feat]
y_all_bal = df_all_bal.loc[:, 'Position'].values

kf = StratifiedKFold(n_splits = 10)
pipe_performance = [] 
LR = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('estimator', LogisticRegression() )       
        ])
RF = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('estimator', RandomForestClassifier() )       
        ]) 
SV = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('estimator', LinearSVC() )       
        ])
print('Classifying Positions' )
for pipe in [LR,RF, SV]:
    print(np.mean(cross_val_score(pipe, x_all_bal, y_all_bal, scoring="accuracy", cv=kf)))
    score = cross_validate(pipe, x_all_bal, y_all_bal, cv=kf, return_train_score=True)
    
    pipe_performance.append([100*np.mean(score['test_score']), 100*np.mean(score['train_score']), 
                                 np.mean(score['fit_time']), np.mean(score['score_time'])])
print(pipe_performance)    

# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
#                   Machine Learning - Label 'Type'                         #    
# ------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------- #

#                          Evaluating Dataset                               #
df_feat.loc[:, 'Type'].value_counts()

df_bal_type = balance_df(df_feat, 'Type')
df_bal_type['Type'].value_counts()

x_bal_type = df_bal_type.loc[:, feat]
y_bal_type = df_bal_type.loc[:, 'Type']



pipe_performance = []
print ('LR, RF, SV classifier for Type Positions')
for pipe in [LR, RF, SV]:
    print(np.mean(cross_val_score(pipe, x_bal_type , y_bal_type , scoring="accuracy", cv=kf)))
    score = cross_validate(pipe, x_bal_type, y_bal_type, cv=kf, return_train_score=True)
    
    pipe_performance.append([100*np.mean(score['test_score']), 100*np.mean(score['train_score']), 
                                 np.mean(score['fit_time']), np.mean(score['score_time'])])
print(pipe_performance)    

#                          Logistic Regression                              #
# We can see that the model is underfitting, so no point in using regularization or PCA
LR = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('estimator', LogisticRegression() )       
        ])
Ridge = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('estimator', RidgeClassifier() )       
        ]) 
    
    
# Evaluating pipelines
pipes = [LR, Ridge]
X_b = [x_bal_type] * 2
y_b = [y_bal_type] * 2
title = 'Difference between LogReg, Ridge'
label = ['LogReg', 'Ridge']
LR = evaluate_pipeline(pipes, X_b, y_b, kf, title, label)
  
#                     Linear Support Vector Classifier                      #
# Creating pipeline

SV = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('estimator', LinearSVC() )       
        ])
LS2 = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components = 120)),
        ('estimator', LinearSVC() )       
        ]) 
LS3 = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components = 100)),
        ('estimator', LinearSVC() )       
        ])
LS4 = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components = 50)),
        ('estimator', LinearSVC() )       
        ]) 
    
pipes = [SV]
title = 'Difference between no PCA, PCA (120), PCA(100), PCA (50) using Linear SCV'
label = ['no PCA', 'PCA - 120 comp',  'PCA - 100 comp', 'PCA - 50 comp']
LS = evaluate_pipeline(pipes, X_b, y_b, kf, title, label)         
    
#                         Random Forest Classifier                          #
RF = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('estimator', RandomForestClassifier() )       
        ]) 
RF2 = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components = 120)),
        ('estimator', RandomForestClassifier() )       
        ])    
RF3 = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components = 100)),
        ('estimator', RandomForestClassifier() )       
        ]) 
RF4 = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components = 50)),
        ('estimator', RandomForestClassifier() )       
        ]) 

pipes = [RF]
title = 'Random Forest - No PCA, PCA (120), PCA(100), PCA (50) '
label = ['No PCA', 'PCA - 120 comp',  'PCA - 100 comp', 'PCA - 50 comp']
RF_Results = evaluate_pipeline(pipes, X_b, y_b, kf, title, label)
      

   
# ------------------------------------------------------------------------- #
# Machine Learning - Tutorial https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60                          #    
# ------------------------------------------------------------------------- # 

#       Splitting DF into Test and Train before starting                   #

df_feat['Type'].value_counts()

X = df_feat.loc[:, feat].values
Y = df_feat.loc[:, 'Type'].values

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size= 0.2, stratify = Y, random_state=0)

# Build new data frame with Scaling and PCA on X_train
scaler = StandardScaler()

scaler.fit(X_train)      
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

pca_50 = PCA(n_components = 50)
pca_50.fit(X_train)                

X_train = pca_50.transform(X_train)     
X_test = pca_50.transform(X_test)  

LogReg = LogisticRegression()

LogReg.fit(X_train, Y_train)
LogReg.score(X_test, Y_test)

explained_variance = pca.explained_variance_ratio_

RF = RandomForestClassifier(max_depth = 5, random_state = 0)
RF.fit(X_train, Y_train)               
RF.score(X_test, Y_test)

# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
#                   Machine Learning - Label 'Position'                     #    
# ------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------- #

# ------------------------------------------------------------------------- #
#                     Classifier for 'Static' Positions                     #    
# ------------------------------------------------------------------------- # 
df_static = df_feat[ df_feat['Type'] == 'Static' ]
df_static['Position'].value_counts()
# Balancing dataframe
df_static_bal = balance_df(df_static, 'Position')
df_static_bal['Position'].value_counts()

X_static_bal = df_static_bal.loc[:, feat]
Y_static_bal = df_static_bal.loc[:, 'Position']

pipe_performance = []
print ('LR, RF, SV classifier for Static Positions')
for pipe in [LR, RF, SV]:
    print(np.mean(cross_val_score(pipe, X_static_bal, Y_static_bal, scoring="accuracy", cv=kf)))
    score = cross_validate(pipe, X_static_bal, Y_static_bal, cv=kf, return_train_score=True)
                                                
    pipe_performance.append([100*np.mean(score['test_score']), 100*np.mean(score['train_score']), 
                                 np.mean(score['fit_time']), np.mean(score['score_time'])])
 
    #print(confusion_matrix( Y_static_bal, (cross_val_predict(pipe, X_static_bal, Y_static_bal, cv= kf))))

print('test_score', 'train_score','fit_time', 'score_time')
print(pipe_performance)


    
# ------------------------------------------------------------------------- #
#                    Classifier for 'Dynamic' Positions                     #    
# ------------------------------------------------------------------------- # 
df_dynamic = df_feat[ df_feat['Type'] == 'Dynamic' ]
df_dynamic['Position'].value_counts()

df_dynamic_bal = balance_df(df_dynamic, 'Position')
df_dynamic_bal['Position'].value_counts()

X_dynamic_bal = df_dynamic_bal.loc[:, feat]
Y_dynamic_bal = df_dynamic_bal.loc[:, 'Position']


pipe_performance = []
print ('LR, RF, SV classifier for Dynamic Positions')
for pipe in [LR, RF, SV]:
    print(np.mean(cross_val_score(pipe, X_dynamic_bal, Y_dynamic_bal, scoring="accuracy", cv=kf)))
    score = cross_validate(pipe, X_dynamic_bal, Y_dynamic_bal, cv=kf, return_train_score=True)
    
    pipe_performance.append([100*np.mean(score['test_score']), 100*np.mean(score['train_score']), 
                                 np.mean(score['fit_time']), np.mean(score['score_time'])])
print(pipe_performance)    
    
