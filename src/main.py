# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:17:09 2019

@author: marinara.marcato
"""
import logging
import os
import glob
import numpy as np
import pandas as pd
<<<<<<< HEAD
#import math
#import time
#import matplotlib.pyplot as plt
#from datetime import timedelta
=======
>>>>>>> 724cdbf... test movels saved
from importlib import reload

# Caching Modules
import joblib
from shutil import rmtree
location = 'cachedir'
memory = joblib.Memory(location=location, verbose=10)

# ------------------------------------------------------------------------- #
#                            Importing ML modules                           #    
# ------------------------------------------------------------------------- #

<<<<<<< HEAD
=======
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GridSearchCV
>>>>>>> 724cdbf... test movels saved
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# ------------------------------------------------------------------------- #
#                          Importing local modules                          #    
# ------------------------------------------------------------------------- #
<<<<<<< HEAD
%reload_ext autoreload
%autoreload 2
=======
>>>>>>> 724cdbf... test movels saved
from setup import *
import imports
import process
import learn
import evaluate

logger = log(__name__)
logger.info('Modules imported')
# ------------------------------------------------------------------------- #
#                        Creating/Importing raw df                          #    
# ------------------------------------------------------------------------- #

# creating info, positions, episode dataframes
df_info, df_pos, df_ep = imports.timestamps(subjects, dcs, base_dir)
# creating imu dataset considering the timestamps created
df_imu = imports.actigraph(df_info, base_dir)
# creating raw dataset and saving it 
df_raw = imports.label(df_info, df_pos, df_imu, df_dir)
# importing created raw dataset - shortcut for all the processes above
df_raw = imports.posture(df_dir, 'df_raw')
<<<<<<< HEAD

=======
>>>>>>> 724cdbf... test movels saved
# ------------------------------------------------------------------------- #
#                           Feature Engineering                             #    
# ------------------------------------------------------------------------- # 
# creating dataset with features with user defined settings 
print(df_name, w_size, w_offset, t_time)
df_feat = process.features(df_raw, df_dir, df_name, w_size, w_offset, t_time)
<<<<<<< HEAD
# importing previously created datasets
df_feat = imports.posture(df_dir, 'df_32')  
=======
'''
# importing previously created datasets
df_feat = imports.posture(df_dir, df_fname)  
>>>>>>> 724cdbf... test movels saved
# visualising feature distribution  
df_dist = process.distribution(df_feat, 'Original Dataset')


# ------------------------------------------------------------------------- #
<<<<<<< HEAD
#                            Data Visualisation                             #    
# ------------------------------------------------------------------------- # 

# creating dev and test sets
df_dev, df_val = process.split(df_feat)

# visualising feature distribution for dev and test sets
process.distribution(df_dev, 'Development Dataset')
process.distribution(df_val, 'Validation Dataset')
=======
#                            Data Visualisations                             #    
# ------------------------------------------------------------------------- # 

# creating dev and test sets
df_dev, df_test = process.split(df_feat, 0.2)
df_train, df_val = process.split(df_dev, 0.25)

# visualising feature distribution for dev and test sets
process.distribution(df_dev, 'Development Dataset')
process.distribution(df_test, 'Test Dataset')

# visualising feature distribution for dev and test sets
process.distribution(df_train, 'Train Dataset')
process.distribution(df_val, 'Validation Dataset')
process.distribution(df_test, 'Test Dataset')
>>>>>>> 724cdbf... test movels saved


# ------------------------------------------------------------------------- #
#                Machine Learning - Label 'Positions'                       #    
# ------------------------------------------------------------------------- # 
# select feature names
feat = df_dev.columns[:-4]
<<<<<<< HEAD
=======
# Removing all Magnetometer features 
features = [x for x in feat if "Mag" not in x]

>>>>>>> 724cdbf... test movels saved
# select features
X = df_dev.loc[:, feat]
# setting label
label = 'Position'
# select label
y = df_dev.loc[:, label].values
# setting a cv strategy that accounts for dogs
<<<<<<< HEAD
cv = GroupKFold(n_splits = 10).split(X, y, groups = df_dev.loc[:,'Dog'])
=======
cv0 = GroupKFold(n_splits = 10).split(X, y, groups = df_dev.loc[:,'Dog'])
cv1 = LeaveOneGroupOut(n_splits = 10).split(X, y, groups = df_dev.loc[:,'Dog'])
>>>>>>> 724cdbf... test movels saved


%reload_ext autoreload
%autoreload 2
<<<<<<< HEAD
import learn
import evaluate

pipes = learn.pipes(feat)
pp_dev = evaluate.pipe_perf(df_dev, feat, 'Position', pipes)


=======
import setup
import learn
import evaluate

#################### RF
gs_pipe = Pipeline([
    ('selector', learn.DataFrameSelector(features,'float64')),
    #('scaler', StandardScaler()),
    #('reduce_dim', PCA()), 
    ('estimator', RandomForestClassifier() )       
], memory = memory ) 

gs_params = {
    'estimator__max_depth' : [3, 5, 10], 
    'estimator__max_features' : [80, 100, 120], 
    'estimator__n_estimators' : [25, 35, 50], 
    #'reduce_dim__n_components' : [80, 100, 120], 
}
#################### GB
gs_pipe = Pipeline([
    ('selector', learn.DataFrameSelector(features,'float64')),
    ('estimator', GradientBoostingClassifier())
], memory = memory)

gs_params = {
    'estimator__max_depth' : [10],
    'estimator__max_features' : [20],
    'estimator__n_estimators': [3, 5, 10]
}
################## KNN
gs_pipe = Pipeline([
    ('selector', learn.DataFrameSelector(features,'float64')),
    ('scaler', StandardScaler()),
    ('estimator', KNeighborsClassifier(n_jobs=-1))
], memory = memory)

gs_params = {
    'estimator__n_neighbors' : [2,5,10,20,40],
    'estimator__weights': ['uniform', 'distance']
}
gs_rf = GridSearchCV(gs_pipe, \
    cv = cv0, \
    scoring = 'f1_weighted', \
    param_grid = gs_params, \
    return_train_score = True, n_jobs = -1)
    
gs_rf.fit(X,y, groups = df_dev.loc[:,'Dog'])
evaluate.print_cv(gs_rf)

# Saving Grid Search Results to pickle file 
run = 'GS-KNN-df_32'
joblib.dump(evaluate.gs_results(gs_rf), '../models/{}.pkl'.format(run), compress = 1 )
memory.clear(warn=False)
rmtree(location)


# Loads Grid Search Results from Pickle file
run = 'GS-RF-PCA-df_11-X'

# Comparing Explained Variance Ratios 
gs = joblib.load('../models/{}.pkl'.format('GS-RF-PCA-df_11'))
evaluate.print_cv(gs)
f = sns.scatterplot(data = gs.best_estimator_['reduce_dim'].explained_variance_)
f.axhline(1, color = 'r')
plt.show()
evaluate.print_cv(gs)


## developing main code for functions to work from separate files
pp_dev = evaluate.gs_perf(df_dev, feat, cv, 'Position', pipes)
>>>>>>> 724cdbf... test movels saved


# ---------------       using naive balanced dataset      --------------------- #
df_dist = process.distribution(df_dev)
df_pos_bal = process.naive_balance(df_dev, 'Position')
df_dist = process.distribution(df_pos_bal)

# calculating the number of examples per category
df_sum = df_pos_bal['Position'].value_counts().reset_index(name= 'count')
# calculating the percentage of examples per category
df_sum['percentage'] = df_sum['count']*100 /df_sum['count'].sum()
print(df_sum)

# learn and evaluate 
pp_pos_bal = evaluate.pipe_perf(df_pos_bal, feat, 'Position', pipes)



# old
feat = df_feat.columns[:-4]
pipes = learn.simple_pipes(feat)
cv = StratifiedKFold(n_splits = 10, random_state = 42, shuffle = True)

df_pos_bal = process.balance(df_feat, 'Position')
print( df_pos_bal['Position'].value_counts(), '\n')
pp = evaluate.pipe_perf(df_pos_bal, feat, 'Position', pipes, cv)

'''
# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
#                        Machine Learning - Label 'Type'                    #    
# ------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------- #

df_type_bal = process.balance(df_feat, 'Type')
print( df_type_bal[''Type'].value_counts())
cm = evaluate.pipe_perf(df_type_bal, feat, 'Type', pipes, cv)


# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
#           Machine Learning - Label 'Position' |'Dynamic'                  #    
# ------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------- #
df_dyn_bal = process.balance_df(df_feat[df_feat['Type'] == 'dynamic'],'Position')
print(df_dyn_bal['Position'].value_counts())
dp = evaluate.pipe_perf(df_dyn_bal, feat, 'Position', pipes, cv)


# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
#           Machine Learning - Label 'Position' |'static'                   #    
# ------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------- #
df_stat_bal = process.balance_df(df_feat[df_feat['Type'] == 'static'],'Position')
print(df_dyn_bal['Position'].value_counts())
sp = evaluate.pipe_perf(df_stat_bal, feat, 'Position', pipes, cv)


# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
#       Machine Learning - Organising stuff for improved datasets           #    
# ------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------- #
group = df.loc[:, 'Dog']
#gkf = list(GroupKFold(n_splits = 5).split(x,y, group)
kf = StratifiedKFold(n_splits = 10)
simple(df, label, pipes, cv)

'''