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
# %load_ext autoreload
# %autoreload 2
import imports, process, analyse, learn, evaluate 

# ------------------------------------------------------------------------- #
#                              Definitions                                  #    
# ------------------------------------------------------------------------- #
dataset = 'simple'
selector = 'SFM'
classifier = 'RF'

# directory where the dataset is located
dir_df = os.path.join(dir_base, 'data', dataset)
# directory to save the model
dir_model = os.path.join(dir_base, 'models', dataset)


# ------------------------------------------------------------------------- #
#                           Importing Datasets                              #
# ------------------------------------------------------------------------- #

# importing previously created dataset
df_feat, df_dev, df_test, feat_all, feat_mag = imports.posture(dir_df, 'df5_11')  

# select the dataframe and feature set for grid search
feat = feat_mag
df = df_dev

# prepare dataframe for evaluation: select features, label,
#   cv strategy (group = dogs, stractified folds labels proportion)
X, y, groups, cv = learn.df_prep(df, feat, label = 'Position')

# ------------------------------------------------------------------------- #
#                      Machine Learning - RF, KNN, GBT                      #
# ------------------------------------------------------------------------- # 

# build pipeline and parameters
pipe, params = learn.pipe(feat, 'SVC', 'RF')
print('pipeline:\n\n', pipe)
print('params:\n\n', params)
print('dir_model:\n\n', dir_model)

# evaluate grid search performance and save to pickle file
gs = evaluate.gs_perf(pipe, params, X, y, groups, cv)

# saving the output of the grid search 
run = 'GS-{}-{}-{}.pkl'.format(dataset, selector, classifier)
gs_path = os.path.join(dir_model, run)
joblib.dump(gs, gs_path, compress = 1 )
memory.clear(warn=False)
rmtree(location)  

print(evaluate.gs_output(gs))