from sklearn.pipeline import Pipeline
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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Caching Libraries
import joblib
from shutil import rmtree
location = 'cachedir'
memory = joblib.Memory(location=location, verbose=10)

def RF(feat):
    RF = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('estimator', RandomForestClassifier() )       
    ]) 
    return({'RF': RF})

def RF_PCA(feat, memory):
    '''
        WORK IN PROGRESS - IM NOT SURE IF IT IS WORTH SEPARATING THE STEPS INTO DIFFERENT FILES
    '''
    p = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('reduce_dim', PCA()),
        ('estimator', RandomForestClassifier() )       
    ], memory = memory ) 
    return(p)

def GB(feat):
    GB = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('estimator', GradientBoostingClassifier() )       
    ])
    return({'GB': GB})

def pipes(feat):
    '''
        Simple Pipelines including feature selection, scaling and estimator (LR, RF or SV)    
    params:
        feat: list with name of columns to be used as features
    '''
    RF = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', 'passthrough'),
        ('reduce_dim', 'passthrough'),
        ('estimator', RandomForestClassifier() )       
    ]) 
    GB = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', 'passthrough'),
        ('reduce_dim', 'passthrough'),
        ('estimator', GradientBoostingClassifier())
    ])
    return ({'RF': RF, 'GB' : GB})


