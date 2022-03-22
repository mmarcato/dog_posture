''' 
    Copyright (C) 2021 by Marinara Marcato
         <marinara.marcato@tyndall.ie>, Tyndall National Institute
        University College Cork, Cork, Ireland.
'''

# ------------------------------------------------------------------------- #
#                                  Imports                                  #    
# ------------------------------------------------------------------------- # 
import os, joblib
from time import time
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest

#multiprocessing 
import multiprocessing as mp
print("Number of Logical processors: ", mp.cpu_count())

# parallel
import joblib
from dask.distributed import Client

client = Client(processes = False)
# ------------------------------------------------------------------------- #
#                             Local Imports                                 #    
# ------------------------------------------------------------------------- # 

ft = 'simple'
mdl = 'direct'
slt = 'SKB'
clf = 'RF'
run = '{}-{}-{}.pkl'.format(mdl, slt, clf)

# Define local directories
dir_current = os.path.dirname(os.path.realpath(__file__))
dir_parent = os.path.dirname(dir_current)
dir_base = os.path.dirname(dir_parent)

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

class CorrelationThreshold(BaseEstimator, TransformerMixin):
    
    """Feature selector that removes all correlated features.

    This feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning.
    
    Parameters
    ----------
    threshold : float, default=0.95
        Features with a training-set correlation higher than this threshold will
        be removed. The default is to keep all features with non-zero variance,
        i.e. remove the features that have the same value in all samples.

    Returns
    ----------
    selected_features_ : list, shape (n_features)
        Returns a list with the selected feature names.

    """

    def __init__(self, threshold = 0.95):
        self.threshold = threshold
        self.to_drop = None
        self.to_keep = None

    def fit (self, X, y = None ): 
        '''
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Sample vectors from which to compute variances.
        y : any, default=None
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.
        Returns
        -------
        self
        '''
        corr_matrix = np.abs(X.corr())
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        self.to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        self.to_keep = list(set(X.columns) - set(self.to_drop))
        return self
        
    def transform(self, X, y = None):
        X_selected = X[self.to_keep]
        return X_selected
    
    def get_support(self):
        return self.to_keep

class gs_results:
    # Storing Grid Search results
    def __init__(self, gs):
        self.cv_results_ = gs.cv_results_
        self.best_estimator_ = gs.best_estimator_
        self.best_params_ = gs.best_params_
        self.best_score_ = gs.best_score_


# ------------------------------------------------------------------------- #
#                  Importing Datasets - Label 'Positions'                   #
# ------------------------------------------------------------------------- #
# importing previously created dataset
df = pd.read_csv( os.path.join(dir_df, 'df5_11-dev.csv'), 
                index_col = ['Timestamp'], 
                parse_dates = ['Timestamp'],
                dayfirst = True,
                date_parser = lambda x: pd.to_datetime(x, format = '%Y-%m-%d %H:%M:%S.%f')    )

# define all features 
feat_all = df.columns[:-5]
# select all features - magnetometer     
feat = [x for x in feat_all if "Mag" not in x]

# prepare dataframe for evaluation: select features, label,
#   cv strategy (group = dogs, stractified folds labels proportion)
X = df.loc[:, feat]
y = df.loc[:, 'Position'].values
groups = df.loc[:,'Dog']

# ------------------------------------------------------------------------- #
#                      Machine Learning - RF                                #
# ------------------------------------------------------------------------- # 

pipe = Pipeline([ 

    ('ft', DataFrameSelector(feat,'float64')),
    #('scl', StandardScaler()), 
    #('cth', CorrelationThreshold()),
    ('slt', SelectKBest(score_func= f_classif)),
    ('clf', RandomForestClassifier(n_jobs = 1))]
    #,memory = memory
    )

params = { 
            'slt__k': [10],#, 15, 20, 30, 50, 80], 

            'clf__max_depth': [3, 5, 7, 10, 13],
            'clf__n_estimators': [25, 50, 100, 250, 500],
            'clf__class_weight': ['balanced'],
            'clf__max_features': [None],
            'clf__random_state': [0]}

cv = GroupKFold(n_splits = 10).split(X, y, groups = groups)

start_time = time()

gs = GridSearchCV(pipe, param_grid = params, 
        scoring = 'f1_weighted', \
        n_jobs = -1, cv = cv, return_train_score = True)

with joblib.parallel_backend('dask'):
    gs.fit(X,y, groups = groups)

end_time = time()
duration = end_time - start_time
print("\n\n--- %s seconds ---\n\n" % (duration))


# save gs results to pickle file
gs_path = os.path.join(dir_current, run)
print(gs_path)
joblib.dump(gs_results(gs), gs_path, compress = 1)


# clear temporary cache 
# memory.clear(warn=False)
# rmtree(location)
