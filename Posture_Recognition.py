# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:17:09 2019

@author: marinara.marcato
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import sklearn.preprocessing 

# Directory containing the data in structure (Dog name -> DC_Device -> Files )
dir_base = "Z:\\Tyndall\\IGDB\\Observational Study\\Data Collection\\Study\\Subjects"
#dir_base = 'N:\\Subjects'
subjects = os.listdir(dir_base)[1:]
dcs = ['DC1', 'DC2']
bps = ['Back', 'Chest', 'Neck']

# ------------------------------------------------------------------------- #
#                               Importing data                              #    
# ------------------------------------------------------------------------- #

#                Importing  file containing the behaviours marked           #
f_name = 'Z:\\Tyndall\\IGDB\\Observational Study\\Data Collection\\Study\\Subjects\\_Timestamps.csv'
df_ts = pd.read_csv(f_name, skiprows = 2, usecols =['Episodes', 'Behaviours'])

#           Importing position and episode info for each dog                #    
df_ep, df_pos = {}, {}
for subj in subjects:
    df_ep[subj], df_pos[subj] = {},{}
    print(subj)
    for dc in dcs:
        df_ep[subj][dc], df_pos[subj][dc] = None, None
        f_name = '%s\\%s\\%s_Timestamps.csv' % (dir_base, subj, dc[-1])  
        if os.path.exists(f_name):
            # Read the information about the behaviour test 
            df_info = pd.read_csv(f_name, index_col = 0, nrows = 4, usecols = [0,1])
            date = df_info[subj]['Date']
            time = df_info[subj]['Start time']
            dt = pd.to_datetime(date + time, format = '%d/%m/%Y%H:%M:%S' )            
            print(dc, dt)
            # Read the episode Virtual Time (VT) 
            df_ep[subj][dc] = pd.read_csv(f_name, skiprows = 6, usecols = ['Episode', 'Ep-VT']).dropna()
            # Create new column for the episode Real Time (RT)
            df_ep[subj][dc].index = dt + pd.to_timedelta(df_ep[subj][dc]['Ep-VT'])         
            # Create new column for the episode Duration
            df_ep[subj][dc]['Duration'] = df_ep[subj][dc].index.to_series().diff().shift(-1)
            
            # Read the position Virtual Time (VT) 
            df_pos[subj][dc] = pd.read_csv(f_name, skiprows = 6, usecols = ['Position', 'Pos-VT']).dropna()
            # Create new column for the position Real Time (RT)
            df_pos[subj][dc].index = dt + pd.to_timedelta(df_pos[subj][dc]['Pos-VT'])         
            # Create new column for the position Duration
            df_pos[subj][dc]['Duration'] = df_pos[subj][dc].index.to_series().diff().shift(-1)        
    

#                           Importing Actigraph IMU                         #    
df_imu = {}
# Iterating through subjects
for subj in subjects:
    df_imu[subj] = {}       
    # Iterating through data collections
    for dc in dcs:
        df_list= []
        df_imu[subj][dc] = None
        # If this the path to data exists
        if os.path.isdir('%s\\%s\\%s_Actigraph' % (dir_base, subj, dc[-1])):
            print(subj, dc)
            # Looping through all bps
            for bp in bps:   
                # Find file path for each bp
                f_name =  glob.glob('%s\\%s\\%s_Actigraph\\*_%s.csv' % (dir_base, subj, dc[-1], bp))
                
                df_list.append(pd.read_csv(f_name[0], index_col = ['Timestamp'], parse_dates = [0], \
                      date_parser = lambda x: pd.to_datetime(x, format = '%Y-%m-%d %H:%M:%S.%f'))\
                        .drop(['Temperature'], axis = 1))
            # Concatenating dataframes for different body parts in one single dataframe
            # Results in one dataframe per dog per data collection
            df_imu[subj][dc] = pd.concat(df_list, axis = 1, keys = bps, \
              names = ['Body Parts', 'Sensor Axis'])
 

# ------------------------------------------------------------------------- #
#               Combining data to create features dataframe                 #    
# ------------------------------------------------------------------------- #           

#                       Populating 'Position' column                         #
        
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
#                   Machine Learning - Label 'Type'                         #    
# ------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------- #


# ------------------------------------------------------------------------- #
#                                 AI1 UCC                                   #    
# ------------------------------------------------------------------------- # 
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA

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
# Shuffling data
df_feat = df_feat.take(np.random.permutation(len(df_feat))) 

y = df_feat.loc[:, 'Type'].values

# Stratified holdout   
ss = StratifiedShuffleSplit(n_splits = 1, train_size = 0.8)
# Stratified k-fold cross-validation
kf = StratifiedKFold(n_splits = 10)


print('PIPELINE 1')
# Creating pipeline
PL1 = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('estimator', LogisticRegression() )       
        ])
    
print('ss', np.mean(cross_val_score(PL1, df_feat, y, scoring="accuracy", cv=ss)))
print('kv', np.mean(cross_val_score(PL1, df_feat, y, scoring="accuracy", cv=kf)))
print('10 folds', np.mean(cross_val_score(PL1, df_feat, y, scoring="accuracy", cv=10)))

print('PIPELINE 2')
PL2 = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components = 100)),
        ('estimator', LogisticRegression() )       
        ])
print('ss',np.mean(cross_val_score(PL2, df_feat, y, scoring="accuracy", cv=ss)))
print('kv', np.mean(cross_val_score(PL2, df_feat, y, scoring="accuracy", cv=kf)))
print('10 folds', np.mean(cross_val_score(PL2, df_feat, y, scoring="accuracy", cv=10)))
   
print('PIPELINE 3')
PL3 = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components = 100)),
        ('estimator', RandomForestClassifier() )       
        ]) 
    
print('ss',np.mean(cross_val_score(PL3, df_feat, y, scoring="accuracy", cv=ss)))
print('kv', np.mean(cross_val_score(PL3, df_feat, y, scoring="accuracy", cv=kf)))
print('10 folds',np.mean(cross_val_score(PL3, df_feat, y, scoring="accuracy", cv=10)))

print('PIPELINE 4')
PL4 = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components = 50)),
        ('estimator', RandomForestClassifier() )       
        ]) 
print('ss',np.mean(cross_val_score(PL4, df_feat, y, scoring="accuracy", cv=ss)))
print('kv', np.mean(cross_val_score(PL4, df_feat, y, scoring="accuracy", cv=kf)))
print('10 folds',np.mean(cross_val_score(PL4, df_feat, y, scoring="accuracy", cv=10)))
   
# ------------------------------------------------------------------------- #
# Machine Learning - Tutorial https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60                          #    
# ------------------------------------------------------------------------- # 

#       Splitting DF into Test and Train before starting                   #
from sklearn.model_selection import train_test_split
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

df_feat['Position'].value_counts()
df_pos = df_feat[df_feat['Position'] != 'Moving']
df_pos['Position'].value_counts()


