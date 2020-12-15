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
#import math
#import time
#import matplotlib.pyplot as plt
#from datetime import timedelta
from importlib import reload

# ------------------------------------------------------------------------- #
#                            Importing ML modules                           #    
# ------------------------------------------------------------------------- #

from sklearn.model_selection import GroupKFold

# ------------------------------------------------------------------------- #
#                          Importing local modules                          #    
# ------------------------------------------------------------------------- #
%reload_ext autoreload
%autoreload 2
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
# importing previously created datasets
df_feat = imports.posture(df_dir, 'df_32')  
# visualising feature distribution  
df_dist = process.distribution(df_feat, 'Original Dataset')


# ------------------------------------------------------------------------- #
#                            Data Visualisation                             #    
# ------------------------------------------------------------------------- # 

# creating dev and test sets
df_dev, df_val = process.split(df_feat)

# visualising feature distribution for dev and test sets
process.distribution(df_dev, 'Development Dataset')
process.distribution(df_val, 'Validation Dataset')


# ------------------------------------------------------------------------- #
#                Machine Learning - Label 'Positions'                       #    
# ------------------------------------------------------------------------- # 
# select feature names
feat = df_dev.columns[:-4]
# select features
X = df_dev.loc[:, feat]
# setting label
label = 'Position'
# select label
y = df_dev.loc[:, label].values
# setting a cv strategy that accounts for dogs
cv = GroupKFold(n_splits = 10).split(X, y, groups = df_dev.loc[:,'Dog'])


%reload_ext autoreload
%autoreload 2
import learn
import evaluate

pipes = learn.pipes(feat)
pp_dev = evaluate.pipe_perf(df_dev, feat, 'Position', pipes)




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