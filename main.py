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

%reload_ext autoreload
%autoreload 2

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
dir_dfs = 'C:\\Users\\marinara.marcato\\Scripts\\dog_posture\\dfs'
dir_base = 'C:\\Users\\marinara.marcato\\Data\\Subjects'
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
w_offset = timedelta(seconds = .25) #offset from start time for the value to be taken
t_time = timedelta(seconds = .25) #transition time between positions

feat = ['mean','std', 'median', 'min', 'max']
df_feat = data_prep.simple_features(subjects, dcs, df_pos, df_imu, feat, w_size, w_offset, t_time)
df_feat.to_csv('%s//df1.csv' % dir_dfs)

# Shuffling data
df_feat = df_feat.take(np.random.RandomState(seed=42).permutation(len(df_feat)))

# Checking no of examples per category 
logger.info('\n\tNumber of Examples in raw dataframe\n')
print(df_feat['Position'].value_counts(), '\n')
print(df_feat['Type'].value_counts())



# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
#                Machine Learning - Label 'Positions'                       #    
# ------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------- #
pipes = learn.simple_pipes(feat)
cv = StratifiedKFold(n_splits = 10, random_state = 42, shuffle = True)

df_pos_bal = data_prep.balance_df(df_feat, 'Position')
print( df_pos_bal['Position'].value_counts(), '\n')
pp = evaluate.pipe_perf(df_pos_bal, feat, 'Position', pipes, cv)


# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
#                        Machine Learning - Label 'Type'                    #    
# ------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------- #

df_type_bal = data_prep.balance_df(df_feat, 'Type')
print( df_type_bal['Type'].value_counts())
cm = evaluate.pipe_perf(df_type_bal, feat, 'Type', pipes, cv)


# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
#           Machine Learning - Label 'Position' |'Dynamic'                  #    
# ------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------- #
df_dyn_bal = data_prep.balance_df(df_feat[df_feat['Type'] == 'dynamic'],'Position')
print(df_dyn_bal['Position'].value_counts())
dp = evaluate.pipe_perf(df_dyn_bal, feat, 'Position', pipes, cv)


# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
#           Machine Learning - Label 'Position' |'static'                   #    
# ------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------- #
df_stat_bal = data_prep.balance_df(df_feat[df_feat['Type'] == 'static'],'Position')
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

