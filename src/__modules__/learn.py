from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold

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

# Caching Libraries
import joblib
from shutil import rmtree
location = 'cachedir'
memory = joblib.Memory(location=location, verbose=10)

def df_prep(df, feat, label):
    ''' Extracts X by selecting feat columns in df and y from label columns in df
        Inputs:
            df: dataframe
            feat: list of strings that represent the column names in df to be selected
            label: string column name in df to be used as the target label    
        
        Output: 
            returns X (dataframe), y(array), groups(list) and cv() 
    '''
    X = df.loc[:, feat]
    y = df.loc[:, label].values
    groups = df.loc[:,'Dog']
    cv = GroupKFold(n_splits = 10).split(X, y, groups = df.loc[:,'Dog'])

    return(X, y, groups, cv)

def RF(feat):
    RF = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('estimator', RandomForestClassifier() )       
    ]) 
    return(RF)

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


