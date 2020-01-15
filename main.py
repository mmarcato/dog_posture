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
from importlib import reload
import logging  # debug, info,warning, error, critical

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter( '%(asctime)s:%(levelname)s:%(message)s')

file_handler = logging.FileHandler('main.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
# ------------------------------------------------------------------------- #
#                          Importing local modules                          #    
# ------------------------------------------------------------------------- #
import data_import
import data_prep
import learn
import evaluate
# ------------------------------------------------------------------------- #
#                          Initializing parameters                          #    
# ------------------------------------------------------------------------- #
dir_base = 'D:\\Study\\Subjects'
subjects = os.listdir(dir_base)[1:]
dcs = ['DC1', 'DC2']
# ------------------------------------------------------------------------- #
#                               Importing data                              #    
# ------------------------------------------------------------------------- #
df_info, df_pos, df_ep = data_import.timestamps(subjects, dcs, dir_base)
df_imu = data_import.actigraph(subjects, dcs, dir_base)
# ------------------------------------------------------------------------- #
#               Combining data to create Feature dataframe                 #    
# ------------------------------------------------------------------------- #           
# Setting parameters: window size, window offset and transition time 
w_size = 100 # equivalent to 1second as data is recorded at 100Hz
w_offset = timedelta(seconds = .25)
t_time = timedelta(seconds = .25)

feat = ['mean','std', 'median', 'min', 'max']
df_feat = data_prep.simple_features(subjects, dcs, df_pos, df_imu, feat, w_size, w_offset, t_time)

# Checking no of examples per category 
print('\n\tNumber of Examples in raw dataframe\n')
print(df_feat['Position'].value_counts(), '\n')
print( df_feat['Type'].value_counts())

# Deleting rows with nan 
df_feat.dropna(axis = 0, inplace = True)
# Deleting rows with Moving 
df_feat = df_feat[df_feat['Position'] != 'moving']
# Shuffling data
df_feat = df_feat.take(np.random.permutation(len(df_feat)))

# Checking no of examples per category after cleaning
print('\n\n\tNumber of Examples in clean dataframe\n')
print( df_feat['Position'].value_counts(), '\n')
print( df_feat['Type'].value_counts())

# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
#                Machine Learning - Label 'Positions'                       #    
# ------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------- #
pipes = learn.simple_pipes(feat)
cv = StratifiedKFold(n_splits = 10)

df_pos_bal = data_prep.balance_df(df_feat, 'Position')
print( df_pos_bal['Position'].value_counts(), '\n')
d, cm = evaluate.pipe_perf(df_pos_bal, feat, 'Position', pipes, cv)

# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
#                        Machine Learning - Label 'Type'                    #    
# ------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------- #

df_type_bal = data_prep.balance_df(df_feat, 'Type')
print( df_type_bal['Position'].value_counts(), '\n')
print( df_type_bal['Type'].value_counts())
cm = evaluate.pipe_perf(df_type_bal, feat, 'Type', pipes, cv)


# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
#           Machine Learning - Label 'Position' |'Dynamic'                  #    
# ------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------- #
df_dyn_bal = data_prep.balance_df(df_feat[df_feat['Type'] == 'dynamic'],'Position')
print(df_dyn_bal['Position'].value_counts())
evaluate.pipe_perf(df_dyn_bal, 'Position')

# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
#           Machine Learning - Label 'Position' |'static'                   #    
# ------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------- #
df_dyn_bal = data_prep.balance_df(df_feat[df_feat['Type'] == 'static'],'Position')
print(df_dyn_bal['Position'].value_counts())
learn.simple_pipes(df_dyn_bal, 'Position')


# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
#                    Machine Learning - Organising stuff                    #    
# ------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------- #


group = df.loc[:, 'Dog']
#gkf = list(GroupKFold(n_splits = 5).split(x,y, group)
kf = StratifiedKFold(n_splits = 10)



simple(df, label, pipes, cv)

