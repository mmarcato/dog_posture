import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
#from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

#from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA


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
 

def simple_pipes(feat):
    '''
        Simple Pipelines including feature selection, scaling and estimator (LR, RF or SV)    
    params:
        feat: list with name of columns to be used as features
    '''
    LR = Pipeline([
            ('selector', DataFrameSelector(feat,'float64')),
            ('scaler', StandardScaler()),
            ('estimator', LogisticRegression() )       
            ])
    RF = Pipeline([
            ('selector', DataFrameSelector(feat,'float64')),
            ('scaler', StandardScaler()),
            ('estimator', RandomForestClassifier() )       
            ]) 
    SV = Pipeline([
            ('selector', DataFrameSelector(feat,'float64')),
            ('scaler', StandardScaler()),
            ('estimator', LinearSVC() )       
            ])
    return ({'LR': LR, 'RF': RF, 'SV': SV})

