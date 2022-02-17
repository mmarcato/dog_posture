# ------------------------------------------------------------------------- #
#                                   Imports                                 #    
# ------------------------------------------------------------------------- # 

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel

import numpy as np
from scipy.spatial import distance  


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

# Caching Libraries
import joblib
from shutil import rmtree
location = 'cachedir'
memory = joblib.Memory(location=location, verbose=10)


# ------------------------------------------------------------------------- #
#                                  Functions                                #    
# ------------------------------------------------------------------------- # 


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


def DTW(a, b):
    '''
    Dynamic Time Warping 
    Implementation code from
    https://stackoverflow.com/questions/57015499/how-to-use-dynamic-time-warping-with-knn-in-python
    TO DO 
    '''   
    an = a.size
    bn = b.size
    pointwise_distance = distance.cdist(a.reshape(-1,1),b.reshape(-1,1))
    cumdist = np.matrix(np.ones((an+1,bn+1)) * np.inf)
    cumdist[0,0] = 0

    for ai in range(an):
        for bi in range(bn):
            minimum_cost = np.min([cumdist[ai, bi+1],
                                   cumdist[ai+1, bi],
                                   cumdist[ai, bi]])
            cumdist[ai+1, bi+1] = pointwise_distance[ai,bi] + minimum_cost

    return cumdist[an, bn]

# ------------------------------------------------------------------------- #
#                Defining Techniques and Hyperparameters                    #
# ------------------------------------------------------------------------- # 

selector = {
        'passthrough' : 'passthrough',
        'SKB' : SelectKBest(score_func= f_classif),
        'SVC': SelectFromModel(LinearSVC()), 
        'RF' : SelectFromModel(RandomForestClassifier(random_state = 42))
}

selector_hyper = {

        'SKB' : {
                    'slt__k': [10, 15]},#, 20, 30, 50, 80]},

        'SVC' : {
                    'slt__estimator__penalty': ['l1', 'l2'],
                    'slt__estimator__C' : [0.01, 1, 100, 1000, 10000]},
}

classifier = {
        # 'LogisticRegression'     : LogisticRegression(),
        'RF' : RandomForestClassifier(n_jobs = 1),
        'KNN' : KNeighborsClassifier(n_jobs = 1),
        'GBT' : GradientBoostingClassifier(),
        'KNN_DTW' : KNeighborsClassifier(n_jobs = 1, metric=DTW)
}

classifier_hyper = {
        # 'LogisticRegression':{
        #                             'penalty'     : ['l2'],
        #                             'C'           : np.logspace(0, 4, 10),
        #                             'solver'      : ['lbfgs', 'liblinear', 'saga'],
        #                             'class_weight': ['balanced'],
        #                             'random_state': [0]},
        
        # 'SVM':{
        #                             'clf__C'           : [0.01, 0.1, 1, 10, 100, 1000],
        #                             'clf__gamma'       : [1, 0.1, 0.01, 0.001, 0.0001],
        #                             'clf__kernel'      : ['rbf', 'linear'],
        #                             'clf__class_weight': ['balanced'],
        #                             'clf__random_state': [0]},
        'GBT':{
                'clf__max_features': [None],
                'clf__random_state': [0],
                'clf__learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
                'clf__max_depth': [3, 5, 7, 10, 13],
                'clf__n_estimators': [25, 50, 75, 100, 175, 250, 500],
        },
        'RF':{
                'clf__max_depth': [3],
                #clf__max_depth': [3, 5, 7, 10, 13],
                'clf__n_estimators': [25],
                #'clf__n_estimators': [25, 50, 75, 100, 175, 250, 500],
                'clf__class_weight': ['balanced'],
                'clf__max_features': [None],
                'clf__random_state': [0]},
       
        'KNN':{
                'clf__n_neighbors': [1,5,10,20,30,40,50,60,70,80]}
                # weights were commented out because they don't affect mean_test_score p=0.943 > 0.05                 
                #'clf__weights': ['uniform', 'distance']}       
                
}


# ------------------------------------------------------------------------- #
#                                Pipeline                                   #
# ------------------------------------------------------------------------- #

def pipe(feat, slt, clf):

    params = { **selector_hyper[slt], **classifier_hyper[clf]}

    pipe = Pipeline([ 
        ('ft', DataFrameSelector(feat,'float64')),
        ('scl', StandardScaler()), 
        ('slt', selector[slt]),
        ('clf', classifier[clf])], memory = memory)

    return(pipe, params)
