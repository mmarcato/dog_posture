from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel

import numpy as np

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

# Caching Libraries
import joblib
from shutil import rmtree
location = 'cachedir'
memory = joblib.Memory(location=location, verbose=10)

def df_prep(df, feat, label):
    ''' Extracts X by selecting feat columns in df and y from label columns in df
        
        Parameters
        ----------
            df: dataframe
            feat: list of strings that represent the column names in df to be selected
            label: string column name in df to be used as the target label    
        
        Returns
        -------
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
        ('estimator', RandomForestClassifier(random_state = 42))       
    ], memory = memory ) 
    return(p)

def GB(feat):
    GB = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('estimator', GradientBoostingClassifier(random_state = 42) )       
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


