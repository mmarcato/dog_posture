''' 
    Copyright (C) 2021 by Marinara Marcato
         <marinara.marcato@tyndall.ie>, Tyndall National Institute
        University College Cork, Cork, Ireland.
'''

# ------------------------------------------------------------------------- #
#                                  Imports                                  #    
# ------------------------------------------------------------------------- # 
import os
from time import time
import pandas as pd

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
from sklearn.feature_selection import SelectFromModel

# Caching Libraries
import joblib
from shutil import rmtree
location = 'cachedir'
memory = joblib.Memory(location=location, verbose=10)


# ------------------------------------------------------------------------- #
#                             Local Imports                                 #    
# ------------------------------------------------------------------------- # 

ft = 'simple'
slt = 'SKB'
clf = 'RF'
run = '{}-{}-{}.pkl'.format(ft, slt, clf)

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
cv = GroupKFold(n_splits = 10).split(X, y, groups = df.loc[:,'Dog'])


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
                    'slt__k': [10, 15, 20, 30, 50, 80]},

        'SVC' : {
                    'slt__estimator__penalty': ['l1', 'l2'],
                    'slt__estimator__C' : [0.01, 1, 100, 1000, 10000]},
}

classifier = {
        # 'LogisticRegression'     : LogisticRegression(),
        'RF' : RandomForestClassifier(n_jobs = -1),
        'KNN' : KNeighborsClassifier(n_jobs = -1),
        'GBT' : GradientBoostingClassifier(),
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
#                      Machine Learning - RF, KNN, GBT                      #
# ------------------------------------------------------------------------- # 
params = { **selector_hyper[slt], **classifier_hyper[clf]}

pipe = Pipeline([ 

            ('ft', DataFrameSelector(feat,'float64')),
            ('scl', StandardScaler()), 
            ('slt', selector[slt]),
            ('clf', classifier[clf])], 
            
            memory = memory)

start_time = time()
gs = GridSearchCV(pipe, param_grid = params, 
        scoring = 'f1_weighted', \
        n_jobs = -1, cv = cv, return_train_score = True)
gs.fit(X,y, groups = groups)
end_time = time()
duration = end_time - start_time
print("--- %s seconds ---" % (duration))


joblib.dump(gs, '{}/{}.pkl'.format(dir_model, run), compress = 1 )
memory.clear(warn=False)
rmtree(location)

print(gs_results(gs))