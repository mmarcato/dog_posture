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
df_feat = imports.posture(dir_df, 'df5_11')  
np.random.seed(42)
df_feat.insert(216, 'Random', np.random.rand(df_feat.shape[0]))


# ------------------------------------------------------------------------- #
#                           Feature Selection                               #
# ------------------------------------------------------------------------- #

X, y, groups, cv = learn.df_prep(df, feat, label = 'Position')

pipe = learn.RF(feat)
