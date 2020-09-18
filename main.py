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
import matplotlib.pyplot as plt
from datetime import timedelta
from importlib import reload

# ------------------------------------------------------------------------- #
#                            Importing ML modules                           #    
# ------------------------------------------------------------------------- #

from sklearn.model_selection import StratifiedKFold
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
import math
import time

logger = log(__name__)
logger.info('Module imports successful')
# ------------------------------------------------------------------------- #
#                            Importing raw data                             #    
# ------------------------------------------------------------------------- #
df_info, df_pos, df_ep = imports.timestamps(subjects, dcs, base_dir)
df_imu = imports.actigraph(df_info, base_dir)
df_raw = process.label(df_info, df_pos, df_imu, df_dir)

# ------------------------------------------------------------------------- #
#                           Feature Engineering                             #    
# ------------------------------------------------------------------------- #    
df_raw = imports.posture(df_dir)       
df_feat = process.simple_features(df_raw, df_dir, df_name, w_size, w_offset, t_time)


df_feat = process.simple_features1(subjects, dcs, df_pos, df_imu, df_dir, df_name, w_size, w_offset, t_time)
df_dev, df_test = process.split(df_feat)


%reload_ext autoreload
%autoreload 2
import process
w, x, y = process.error_check(df3)



# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
#                Machine Learning - Label 'Positions'                       #    
# ------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------- #
pipes = learn.simple_pipes(feat)
cv = StratifiedKFold(n_splits = 10, random_state = 42, shuffle = True)

df_pos_bal = process.balance_df(df_feat, 'Position')
print( df_pos_bal['Position'].value_counts(), '\n')
pp = evaluate.pipe_perf(df_pos_bal, feat, 'Position', pipes, cv)


# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
#                        Machine Learning - Label 'Type'                    #    
# ------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------- #

df_type_bal = process.balance_df(df_feat, 'Type')
print( df_type_bal['Type'].value_counts())
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

