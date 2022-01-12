# ------------------------------------------------------------------------- #
#                                  Imports                                  #    
# ------------------------------------------------------------------------- # 
## from src.__modules__ import setup
%load_ext autoreload
%autoreload 2
from src.__modules__ import imports
from src.__modules__ import process 
from src.__modules__ import learn 
from src.__modules__ import evaluate 

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
#                           Importing Datasets                            #    
# ------------------------------------------------------------------------- #
# directory where the dataset is located
df_dir = ('..//..//data//processed')

# importing previously created dataset
df_feat = imports.posture(df_dir, 'df_11')  

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
params = {
    'estimator__max_depth' : [3], #, 5, 10],
    'estimator__max_features' : [80],#, 100, 120],
    'estimator__n_estimators' : [25],# , 35, 50],
    #'reduce_dim__n_components' : [80, 100, 120], 
}

# prepare for evaluation: select features, label, dogs, cv strategy
X, y, groups, cv = learn.df_prep(df, feat, label = 'Position')

# evaluate grid seach performance
gs_results = evaluate.gs_perf(pipe, params, X, y, groups, cv)

# Saving Grid Search Results to pickle file 
run = 'RF-Test'
joblib.dump(gs_results, 
            '../models/{}.pkl'.format(run)
            , compress = 1 )
memory.clear(warn=False)
rmtree(location)


# Loading Grid Search Results from Pickle file
run = 'RF-Test'
gs = joblib.load('../models/{}.pkl'.format(run))
evaluate.gs_output(gs)


