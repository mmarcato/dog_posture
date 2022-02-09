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
location = 'cachedir'
memory = joblib.Memory(location=location , verbose=10)

# Define local directories
dir_current = os.path.dirname(os.path.realpath(__file__))
dir_parent = os.path.dirname(dir_current)
dir_base = os.path.dirname(dir_parent)
dir_modules = os.path.join(dir_parent, '__modules__')
# Set path variable
sys.path.append(dir_modules)

# Local Modules
%load_ext autoreload
%autoreload 2
import imports, analyse, learn, evaluate 

# ------------------------------------------------------------------------- #
#                           Define directories                              #    
# ------------------------------------------------------------------------- #

# directory where the dataset is located
dir_df = (dir_base + '\\data\\simple')
# directory to save the model
dir_model = (dir_base + '\\models')

# ------------------------------------------------------------------------- #
#                  Importing Datasets - Label 'Positions'                   #    
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
#                      Exploratory Data Analysis                            #    
# ------------------------------------------------------------------------- # 

# breeds
analyse.breed(df_feat)
analyse.breed(df_test)
analyse.breed(df_dev)

# ------------------------------------------------------------------------- #
#                     Machine Learning - Random Forests                     #
# ------------------------------------------------------------------------- # 

# build pipeline and parameters
pipe, params = learn.pipe(feat, 'SKB', 'RF')

# evaluate grid search performance and save to pickle file
gs_rf = evaluate.gs_perf(pipe, params, X, y, groups, cv)

# saving the output of the grid search 
run = 'GS-SKB-RF'
joblib.dump(gs_rf, '{}/Paper/{}.pkl'.format(dir_model, run), compress = 1 )
memory.clear(warn=False)
rmtree(location)  

evaluate.gs_output(gs_rf)


# ------------------------------------------------------------------------- #
#                           Machine Learning - KNN                          #
# ------------------------------------------------------------------------- # 

# build pipeline and parameters
pipe, params = learn.pipe(feat, 'SKB', 'KNN')

# evaluate grid search performance and save to pickle file
gs_knn = evaluate.gs_perf(pipe, params, X, y, groups, cv)

# saving the output of the grid search 
run = 'GS-SKB-KNN'
joblib.dump(gs_knn, '{}/Paper/{}.pkl'.format(dir_model, run), compress = 1 )
memory.clear(warn=False)
rmtree(location)  

evaluate.gs_output(gs_knn)


# ------------------------------------------------------------------------- #
#                      Machine Learning - SFM, RF                           #
# ------------------------------------------------------------------------- # 

# build pipeline and parameters
pipe, params = learn.pipe(feat, 'SVC', 'RF')

# evaluate grid search performance and save to pickle file
gs = evaluate.gs_perf(pipe, params, X, y, groups, cv)

# saving the output of the grid search 
run = 'GS-SVC-RF'
joblib.dump(gs, '{}/Paper/{}.pkl'.format(dir_model, run), compress = 1 )
memory.clear(warn=False)
rmtree(location)  

print(evaluate.gs_output(gs_rf))

