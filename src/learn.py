from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
<<<<<<< HEAD

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
#from sklearn.preprocessing import LabelEncoder


#from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
#from sklearn.model_selection import KFold
#from sklearn.model_selection import StratifiedShuffleSplit


from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.linear_model import RidgeClassifier


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate


from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA

 
def LR(feat):
    LR = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('estimator', LogisticRegression() )       
    ])
    return({'LR': LR})

def RF(feat):
    RF = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('estimator', RandomForestClassifier() )       
    ]) 
    return({'RF': RF})

def GB(feat):
    LR = Pipeline([
=======

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
>>>>>>> 724cdbf... test movels saved
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
<<<<<<< HEAD
    LR = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('estimator', LogisticRegression() )       
    ])    
=======
>>>>>>> 724cdbf... test movels saved
    RF = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('reduce_dim', 'passthrough'),
        ('estimator', RandomForestClassifier() )       
    ]) 
    GB = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('reduce_dim', 'passthrough'),
        ('estimator', GradientBoostingClassifier())
    ])
    return ({'LR': LR, 'RF': RF, 'GB' : GB})

<<<<<<< HEAD


=======
>>>>>>> 724cdbf... test movels saved

