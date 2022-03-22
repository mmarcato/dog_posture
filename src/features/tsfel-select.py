# ------------------------------------------------------------------------- #
#                                  Imports                                  #    
# ------------------------------------------------------------------------- # 
## General imports
import sys, os, joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from time import time

## sklearn imports
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, VarianceThreshold
from sklearn.model_selection import GroupKFold, cross_validate, GridSearchCV

# Define local directories
dir_current = os.path.dirname(os.path.realpath(__file__))
dir_parent = os.path.dirname(dir_current)
dir_base = os.path.dirname(dir_parent)

# ------------------------------------------------------------------------- #
#                                   Classes                                 #    
# ------------------------------------------------------------------------- # 

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
        X = pd.DataFrame(X)
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
        self.to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        self.to_keep = list(set(X.columns) - set(self.to_drop))
        return self
        
    def transform(self, X, y = None):
        X = pd.DataFrame(X)
        X_selected = X[self.to_keep]
        return X_selected
    
    def get_support(self):
        return self.to_keep


# ------------------------------------------------------------------------- #
#                           Importing Datasets                              #    
# ------------------------------------------------------------------------- #
# select dataframe
df_name = 'df-all-dev'
print('Dataset name:', df_name)
# directory where the datasets are located
df_path = os.path.join(dir_base, 'data', 'tsfel', "{}.csv".format(df_name))
# imports all datasets in directory
df = pd.read_csv( df_path, 
        index_col = ['Timestamp'], 
        parse_dates = ['Timestamp'],
        dayfirst = True,
        date_parser = lambda x: pd.to_datetime(x, format = '%Y-%m-%d %H:%M:%S.%f'))
print("Dataset shape:", df.shape)
# ------------------------------------------------------------------------- #
#                        Feature Selection - Main                           #    
# ------------------------------------------------------------------------- #

# insert a random variable for feature importance comparison
df.insert(0,                # position
            'random',       # column name
            np.random.RandomState(1234).uniform(low=0, high=1, size = (df.shape[0]),)) 

# define all features 
feat = df.columns[:-5]
print(len(feat))

pipe = Pipeline([
        ('ft', DataFrameSelector(feat,'float64')),
        ('var', VarianceThreshold()),
        ('cor', CorrelationThreshold()),
        ('clf', RandomForestClassifier(random_state= 42))
    ])


# prepare dataframe for evaluation: select features, label,
#   cv strategy (group = dogs, stractified folds labels proportion)
X = df.loc[:, feat]
y = df.loc[:, 'Position'].values
groups = df.loc[:,'Dog']
params = dict()
cv = GroupKFold(n_splits = 10).split(X, y, groups = groups)

gs = GridSearchCV(pipe, param_grid = params, 
        scoring = 'f1_weighted', \
        n_jobs = 40, cv = cv, return_train_score = True)


start_time = time()
gs.fit(X,y, groups = groups)
end_time = time()
duration = end_time - start_time
print("\n\n--- %s seconds ---\n\n" % (duration))


class gs_results:
    # Storing Grid Search results
    def __init__(self, gs):
        self.cv_results_ = gs.cv_results_
        self.best_estimator_ = gs.best_estimator_
        self.best_params_ = gs.best_params_
        self.best_score_ = gs.best_score_
print(gs)
run = "TSFEL-SELECT.pkl"
# save gs results to pickle file
gs_path = os.path.join(dir_current, run)
print(gs_path)
joblib.dump(gs_results(gs), gs_path, compress = 1) 


# start_time = time()
# # using df_train to check on the Feature Importances for the RF classifier
# # this will help me pick an optimal  number for the feature selection algorithm
# rf_cv = cross_validate(
#             estimator = rf_pipe, 
#             X = df.loc[:, feat], 
#             y = df.loc[:, 'Position'].values, 
#             groups = df.loc[:,'Dog'],
#             cv= GroupKFold(n_splits = 2), 
#             scoring = 'f1_weighted', 
#             return_train_score= True,
#             return_estimator = True,
#             n_jobs = -1
#         )
# end_time = time()
# duration = end_time - start_time
# print("\n\n--- %s seconds ---\n\n" % (duration))


# class cv_results:
#     # Storing Cross Validate results and the feature names
#     def __init__(self, cv, feat):
#         self.test_score = cv.test_score
#         self.train_score = cv.train_score
#         self.fit_time = cv.fit_time
#         self.score_time = cv.score_time
#         self.estimator = cv.estimator
#         self.feat = feat

# print(rf_cv)
# run = "TSFEL-SELECT.pkl"
# # save gs results to pickle file
# gs_path = os.path.join(dir_current, run)
# print(gs_path)
# joblib.dump(cv_results(rf_cv, feat), gs_path, compress = 1)