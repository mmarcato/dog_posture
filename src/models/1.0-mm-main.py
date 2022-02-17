''' 
    Copyright (C) 2021 by Marinara Marcato
         <marinara.marcato@tyndall.ie>, Tyndall National Institute
        University College Cork, Cork, Ireland.
'''

# ------------------------------------------------------------------------- #
#                                  Imports                                  #    
# ------------------------------------------------------------------------- # 
# general libraries
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

# Caching Libraries
import joblib
from shutil import rmtree


# ------------------------------------------------------------------------- #
#                              Definitions                                  #    
# ------------------------------------------------------------------------- #
dataset = 'simple'
selector = 'SKB'
classifier = 'RF'

location = 'cachedir'
memory = joblib.Memory(location=location , verbose=10)

# ------------------------------------------------------------------------- #
#                             Local Imports                                 #    
# ------------------------------------------------------------------------- # 
# Define local directories
dir_current = os.path.dirname(os.path.realpath(__file__))
dir_parent = os.path.dirname(dir_current)
dir_base = os.path.dirname(dir_parent)
dir_modules = os.path.join(dir_base, 'src', '__modules__')
# Set path variable
sys.path.append(dir_modules)

# Local Modules
%load_ext autoreload
%autoreload 2
import imports, analyse, learn, evaluate 

# directory where the dataset is located
dir_df = os.path.join(dir_base, 'data', dataset)
# directory to save the model
dir_model = os.path.join(dir_base, 'models', dataset)


# ------------------------------------------------------------------------- #
#                  Importing Datasets - Label 'Positions'                   #
# ------------------------------------------------------------------------- #
# importing previously created dataset
df = imports.posture(dir_df, 'df5_11-dev.csv')  

# define all features 
feat_all = df.columns[:-5]
# select all features - magnetometer     
feat = [x for x in feat_all if "Mag" not in x]

# prepare dataframe for evaluation: select features, label,
#   cv strategy (group = dogs, stractified folds labels proportion)
X, y, groups, cv = learn.df_prep(df, feat, label = 'Position')

# ------------------------------------------------------------------------- #
#                      Machine Learning - RF, KNN, GBT                      #
# ------------------------------------------------------------------------- # 

# build pipeline and parameters
pipe, params = learn.pipe(feat, selector, classifier)
print('pipeline:\n\n', pipe)
print('params:\n\n', params)
print('dir_model:\n\n', dir_model)

# evaluate grid search performance and save to pickle file
gs = evaluate.gs_perf(pipe, params, X, y, groups, cv)

# saving the output of the grid search 
run = '{}-{}-{}.pkl'.format(dataset, selector, classifier)
gs_path = os.path.join(dir_model, run)
joblib.dump(gs, gs_path, compress = 1)

memory.clear(warn=False)
rmtree(location)  
