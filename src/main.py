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
from importlib import reload

# Caching Modules
import joblib
from shutil import rmtree
location = 'cachedir'
memory = joblib.Memory(location=location, verbose=10)

# ------------------------------------------------------------------------- #
#                            Importing ML modules                           #    
# ------------------------------------------------------------------------- #
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# ------------------------------------------------------------------------- #
#                          Importing local modules                          #    
# ------------------------------------------------------------------------- #
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

# ------------------------------------------------------------------------- #
#                           Feature Engineering                             #    
# ------------------------------------------------------------------------- # 
# creating dataset with features with user defined settings 
print(df_name, w_size, w_offset, t_time)
df_feat = process.features(df_raw, df_dir, df_name, w_size, w_offset, t_time)

# importing previously created dataset
df_feat = imports.posture(df_dir, df_fname)  
# creating dev and test sets
df_dev, df_test = process.split(df_feat, 0.2)
df_train, df_val = process.split(df_dev, 0.25)

# ------------------------------------------------------------------------- #
#                            Data Visualisations                             #    
# ------------------------------------------------------------------------- # 
# visualising feature distribution  
df_dist = process.distribution(df_feat, 'Original Dataset')

# visualising feature distribution for dev and test sets
process.distribution(df_dev, 'Development Dataset')
process.distribution(df_test, 'Test Dataset')

# visualising feature distribution for dev and test sets
process.distribution(df_train, 'Train Dataset')
process.distribution(df_val, 'Validation Dataset')
process.distribution(df_test, 'Test Dataset')


# ------------------------------------------------------------------------- #
#                Machine Learning - Label 'Positions'                       #    
# ------------------------------------------------------------------------- # 
df = df_dev
# select feature names
feat_all = df.columns[:-4]
# Removing all Magnetometer features 
feat_mag = [x for x in feat_all if "Mag" not in x]

# select features
X = df.loc[:, feat_mag]
# setting label
label = 'Position'
# select label
y = df.loc[:, label].values
# setting a cv strategy that accounts for dogs
cv0 = GroupKFold(n_splits = 10).split(X, y, groups = df.loc[:,'Dog'])
cv1 = LeaveOneGroupOut().split(X, y, groups = df.loc[:,'Dog'])

#################### RANDOM FOREST
gs_pipe = Pipeline([
    ('selector', learn.DataFrameSelector(features,'float64')),
    ('scaler', StandardScaler()),
    ('reduce_dim', PCA()), 
    ('estimator', RandomForestClassifier() )       
], memory = memory ) 

gs_params = {
    'estimator__max_depth' : [3, 5, 10], 
    'estimator__max_features' : [80, 100, 120], 
    'estimator__n_estimators' : [25, 35, 50], 
    #'reduce_dim__n_components' : [80, 100, 120], 
}


#################### GRADIENT BOOSTED TREEES
gs_pipe = Pipeline([
    ('selector', learn.DataFrameSelector(feat_mag,'float64')),
    ('estimator', GradientBoostingClassifier())
], memory = memory)

gs_params = {
    'estimator__max_depth' : [3, 10, 15, 20], 
    'estimator__max_features' : [10, 20, 50, 70],
    'estimator__n_estimators': [5, 10, 15]
}


#                         GRID SEARCH                         #
gs = GridSearchCV(gs_pipe, \
    cv = cv0, \
    scoring = 'f1_weighted', \
    param_grid = gs_params, \
    return_train_score = True, n_jobs = -1)
    
gs.fit(X,y, groups = df.loc[:,'Dog'])
evaluate.gs_output(gs)

# Saving Grid Search Results to pickle file 
run = 'GS-GB-df_32-2'
joblib.dump(evaluate.gs_results(gs), '../models/{}.pkl'.format(run), compress = 1 )
memory.clear(warn=False)
rmtree(location)


# Loading Grid Search Results from Pickle file
run = 'GS-RF-df_32-3'
gs = joblib.load('../models/{}.pkl'.format(run))
evaluate.gs_output(gs)


# Comparing Explained Variance Ratios (PCA)
f = sns.scatterplot(data = gs.best_estimator_['reduce_dim'].explained_variance_)
f.axhline(1, color = 'r')
plt.show()
evaluate.gs_output(gs)


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