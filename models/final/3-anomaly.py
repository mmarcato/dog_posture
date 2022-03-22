''' 
    Copyright (C) 2022 by Marinara Marcato
         <marinara.marcato@tyndall.ie>, Tyndall National Institute
        University College Cork, Cork, Ireland.
'''

# ------------------------------------------------------------------------- #
#                                  Imports                                  #    
# ------------------------------------------------------------------------- # 
import os
from time import time
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GroupKFold, GridSearchCV


# Caching Libraries
import joblib

import multiprocessing as mp
print("Number of Logical processors: ", mp.cpu_count())

# ------------------------------------------------------------------------- #
#                             Local Imports                                 #    
# ------------------------------------------------------------------------- # 

# Define local directories
dir_current = os.path.dirname(os.path.realpath(__file__))
dir_parent = os.path.dirname(dir_current)
dir_base = os.path.dirname(dir_parent)

# ------------------------------------------------------------------------- #
#                           Model Parameters                                #    
# ------------------------------------------------------------------------- # 


# select dataset  
ft = 'final'
# select model parameters
exp = 3
mdl = 'anomaly'
slt = 'SKB'
clf = 'IF'
run = '{}-{}-{}-{}.pkl'.format(exp, mdl, slt, clf)


# directory where the dataset is located
dir_df = os.path.join(dir_base, 'data', ft)
# directory to save the model
dir_model = os.path.join(dir_base, 'models', ft)

# ------------------------------------------------------------------------- #
#                                   Classes                                 #    
# ------------------------------------------------------------------------- # 

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

class gs_results:
    # Storing Grid Search results
    def __init__(self, gs):
        self.cv_results_ = gs.cv_results_
        self.best_estimator_ = gs.best_estimator_
        self.best_params_ = gs.best_params_
        self.best_score_ = gs.best_score_


# ------------------------------------------------------------------------- #
#                              Importing Dataset                            #
# ------------------------------------------------------------------------- #
# importing previously created dataset
df = pd.read_csv(os.path.join(dir_df, 'df-all-dev.csv'), 
                index_col = ['Timestamp'], 
                parse_dates = ['Timestamp'],
                dayfirst = True,
                date_parser = lambda x: pd.to_datetime(x, format = '%Y-%m-%d %H:%M:%S.%f')    )

# define all features 
feat = df.columns[:-6]

# prepare dataframe for evaluation: select features, label,
#   cv strategy (group = dogs, stractified folds labels proportion)
X = df.loc[:, feat]
y = df.loc[:, 'Shake'].values
groups = df.loc[:,'Dog']

# ------------------------------------------------------------------------- #
#                Machine Learning - Isolation Forest                        #
# ------------------------------------------------------------------------- # 

from imblearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest

pipe = Pipeline([ 
    ('ft', DataFrameSelector(feat,'float64')),
    ('slt', SelectKBest(score_func = f_classif)),
    ('clf', IsolationForest(n_jobs = 1, random_state = 0,
                max_features = 1.0 # selects all features
                ))]
    )

params = { 
            'slt__k':  [10, 20, 35, 55, 80],
            'clf__n_estimators': [25, 50, 100, 250, 500],
            'clf__contamination': [0.005, 0.01, 0.05, 0.1]
        }


cv = GroupKFold(n_splits = 10).split(X, y, groups = groups)


start_time = time()
gs = GridSearchCV(pipe, param_grid = params, 
        scoring = 'f1_weighted', \
        n_jobs = 40, cv = cv, return_train_score = True)
gs.fit(X,y, groups = groups)
end_time = time()
duration = end_time - start_time
print("--- %s seconds ---" % (duration))


# save gs results to pickle file
gs_path = os.path.join(dir_current, run)
print(gs_path)
joblib.dump(gs_results(gs), gs_path, compress = 1 )