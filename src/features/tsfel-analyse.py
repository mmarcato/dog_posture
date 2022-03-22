''' Copyright (C) 2022 by Marinara Marcato
         <marinara.marcato@tyndall.ie>, Tyndall National Institute
        University College Cork, Cork, Ireland.
'''
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

# Define local directories
dir_current = os.getcwd()
dir_parent = os.path.dirname(dir_current)
dir_base = os.path.dirname(dir_parent)
dir_modules = os.path.join(dir_base, 'src', '__modules__')
# Set path variable
sys.path.append(dir_modules)

# Local Modules
%load_ext autoreload
%autoreload 2
import analyse, learn, evaluate 

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

class gs_results:
    # Storing Grid Search results
    def __init__(self, gs):
        self.cv_results_ = gs.cv_results_
        self.best_estimator_ = gs.best_estimator_
        self.best_params_ = gs.best_params_
        self.best_score_ = gs.best_score_

class cv_results:
    # Storing Cross Validate results and the feature names
    def __init__(self, cv, feat):
        self.test_score = cv.test_score
        self.train_score = cv.train_score
        self.fit_time = cv.fit_time
        self.score_time = cv.score_time
        self.estimator = cv.estimator
        self.feat = feat


# ------------------------------------------------------------------------- #
#                    Load Cross Validation results                          #
# ------------------------------------------------------------------------- #

# Loading results from Pickle file

# run name
gs = joblib.load(os.path.join(dir_base, 'src', 'features', 'TSFEL-SELECT.pkl'))
print("Evaluate Grid Search output\n")
# print best estimator results
evaluate.gs_output(gs)

# ------------------------------------------------------------------------- #
#                        Feature Importance Analysis                        #
# ------------------------------------------------------------------------- #

# Evaluate Feature Importance from Select K Features
df_ft = pd.DataFrame({
        'Feature': gs.best_estimator_['cor'].get_support(), 
        'Importance': gs.best_estimator_['clf'].feature_importances_
        })
df_ft.sort_values(by = 'Importance', ascending = False, inplace = True, ignore_index = True)
importance_threshold = df_ft.loc[df_ft['Feature'] == 1,'Importance'].values[0]
type(importance_threshold)
df_ft['keep'] = df_ft['Importance'] > importance_threshold
print("Number of features to keep:", df_ft['keep'].sum())
#### these are the feature number BUT THEY ARE +1.... SO I SHOULD CHANGE IT
print(df_ft.loc[df_ft['keep'], 'Feature'])

gs.best_estimator_.
feat[np.where(gs.best_estimator_['var'].variances_ != 0)]

# Plotting 
slt_ft = gs.best_params_['slt__k']
plt.figure(figsize= (20, 8))
plt.bar(df_ft.loc[:slt_ft,'Feature'],df_ft.loc[:slt_ft, 'Importance'])
plt.xticks(rotation = 45)
plt.title('Best {} Features and their importances using SKB'.format(slt_ft))
plt.xlabel('Features')
plt.ylabel('Importance')
plt.savefig('{}/results/SKB-BestEstimator_BestFeatures'.format(dir_base))
# important features from the best 
rf_ft = list(df_ft.loc[:14, 'Feature'])


### CRAP

# add cv results to a dataframe 
df_cv = pd.DataFrame(cv.test_score)
df_cv['train_test_gap'] = df_cv['mean_train_score'] - df_cv['mean_test_score']
print(df_cv.filter(like = "param").columns.to_list())
print(df_cv.shape)

