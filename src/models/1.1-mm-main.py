''' Copyright (C) 2021 by Marinara Marcato
         <marinara.marcato@tyndall.ie>, Tyndall National Institute
        University College Cork, Cork, Ireland.
'''

# ------------------------------------------------------------------------- #
#                                  Imports                                  #    
# ------------------------------------------------------------------------- # 
import os
import pandas as pd
import numpy as np

# Caching Libraries
import joblib
from shutil import rmtree
location = 'cachedir'
memory = joblib.Memory(location=location , verbose=10)

# directories and setting path variable
dir_current = os.path.dirname(os.path.realpath(__file__))
dir_parent = os.path.dirname(dir_current)
dir_base = os.path.dirname(dir_parent)
dir_modules = os.path.join(dir_parent, '__modules__')
sys.path.append(dir_modules)

# import setup
%load_ext autoreload
%autoreload 2
import imports
import process 
import analyse
import learn
import evaluate 


# ------------------------------------------------------------------------- #
#                           Define directories                              #    
# ------------------------------------------------------------------------- #

# directory where the dataset is located
dir_df = (dir_base + '\\data\\simple')
# directory to save the model
dir_model = (dir_base + '\\models')


# ------------------------------------------------------------------------- #
#                           Importing Datasets                              #
# ------------------------------------------------------------------------- #

# importing previously created dataset
df_feat, df_dev, df_test, feat_all, feat_mag = imports.posture(dir_df, 'df5_11')  

# select the dataframe and feature set for grid search
feat = feat_mag
df = df_dev


# ------------------------------------------------------------------------- #
#                Machine Learning - Label 'Positions'                       #    
# ------------------------------------------------------------------------- # 

# prepare dataframe for evaluation: select features, label,
#   cv strategy (group = dogs, stractified folds labels proportion)
X, y, groups, cv = learn.df_prep(df, feat, label = 'Position')

# build pipeline and parameters
pipe, params = learn.pipe(feat, 'SFM', 'RF')

# evaluate grid search performance and save to pickle file
gs = evaluate.gs_perf(pipe, params, X, y, groups, cv)

# saving the output of the grid search 
run = 'GS-SKB-RF'
joblib.dump(gs, '{}/Paper/{}.pkl'.format(dir_model, run), compress = 1 )
memory.clear(warn=False)
rmtree(location)
  

evaluate.gs_output(gs)
