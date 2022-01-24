''' Copyright (C) 2021 by Marinara Marcato
         <marinara.marcato@tyndall.ie>, Tyndall National Institute
        University College Cork, Cork, Ireland.
'''

# ------------------------------------------------------------------------- #
#                                  Imports                                  #    
# ------------------------------------------------------------------------- # 
import os
import pandas as pd
import time
import tsfel

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
modulesdir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '__modules__'))
sys.path.append(modulesdir)

# import setup
%load_ext autoreload
%autoreload 2
import imports
import process 
import learn 
import evaluate 

#############################
'''
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier

    from sklearn.base import BaseEstimator, TransformerMixin
    class DataFrameSelector(BaseEstimator, TransformerMixin):
        def __init__(self, attribute_names, dtype=None):
            self.attribute_names = attribute_names
            self.dtype = dtype
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            X_selected = X[self.attribute_names]
            if self.dtype:
                return X_selected.astype(self.dtype).values
            return X_selected.values
'''
# Caching Libraries
import joblib
from shutil import rmtree
location = 'cachedir'
memory = joblib.Memory(location=location, verbose=10)

import numpy as np
np.random.seed(42)

# ------------------------------------------------------------------------- #
#                           Importing Datasets                              #    
# ------------------------------------------------------------------------- #
# directory where the dataset is located
df_dir = ('..//..//data//processed')

# importing previously created dataset
df_feat = imports.posture(df_dir, 'df4_11')  

# creating dev and test sets
df_dev, df_test = process.split(df_feat, 0.2)
df_train, df_val = process.split(df_dev, 0.25)
df = df_train

# define all features and - magnetometer 
feat_all = df.columns[:-4]
feat_mag = [x for x in feat_all if "Mag" not in x]
feat = feat_mag

# ------------------------------------------------------------------------- #
#                Machine Learning - Label 'Positions'                       #    
# ------------------------------------------------------------------------- # 

# set up the pipelines for learning
pipe = learn.RF(feat)

# define grid search parameters
params = {#
    'estimator__max_depth' : [10, 15, 20],
    'estimator__max_features' : [14, 37, 60],
    'estimator__n_estimators' : [ 40, 100, 160],
    #'reduce_dim__n_components' : [80, 100, 120], 
}

# prepare for evaluation: select features, label, dogs, cv strategy
X, y, groups, cv = learn.df_prep(df, feat, label = 'Position')

start_time = time.time()
# evaluate grid seach performance
gs_results = evaluate.gs_perf(pipe, params, X, y, groups, cv)
print("--- %s seconds ---" % (time.time() - start_time))

# Saving Grid Search Results to pickle file 
run = 'RF-Test2'
gs_results_dir = os.path.abspath(os.path.join(os.path.dirname(parentdir), 'models'))
joblib.dump(gs_results, 
            '{}/{}.pkl'.format(gs_results_dir, run)
            , compress = 1 )
memory.clear(warn=False)
rmtree(location)



# ------------------------------------------------------------------------- #
#                   Load and Evaluate Grid Search results                   #    
# ------------------------------------------------------------------------- #

# Loading Grid Search Results from Pickle file
run = 'RF-Test2'
gs = joblib.load('{}/{}.pkl'.format(gs_results_dir, run))
evaluate.gs_output(gs)

# Calculate mean test score value while maintaining one parameter constant at a time
df_cv = pd.DataFrame(gs.cv_results_)
print(df_cv.groupby(['param_estimator__max_depth'])['mean_test_score'].mean())
print(df_cv.groupby(['param_estimator__n_estimators'])['mean_test_score'].mean())
print(df_cv.groupby(['param_estimator__max_features'])['mean_test_score'].mean())

for depth in df_cv['param_estimator__max_depth'].unique():
    print(depth)
    df = df_cv.loc[df_cv['param_estimator__max_depth']== depth]
    sns.catplot(data=df, kind="bar",
        x="param_estimator__n_estimators", y="mean_test_score", hue="param_estimator__max_features",
        ci="sd", palette="dark", alpha=.6, height=6 )
