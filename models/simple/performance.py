''' Copyright (C) 2022 by Marinara Marcato
         <marinara.marcato@tyndall.ie>, Tyndall National Institute
        University College Cork, Cork, Ireland.
'''
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
import joblib

# Define local directories
dir_current = os.path.dirname(os.path.realpath(__file__))
dir_parent = os.path.dirname(dir_current)
dir_base = os.path.dirname(dir_parent)
dir_modules = os.path.join(dir_base, 'src', '__modules__')
# Set path variable
sys.path.append(dir_modules)

# Local Modules
%load_ext autoreload
%autoreload 2
import imports, analyse, learn, evaluate 


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

class gs_results:
    # Storing Grid Search results
    def __init__(self, gs):
        self.cv_results_ = gs.cv_results_
        self.best_estimator_ = gs.best_estimator_
        self.best_params_ = gs.best_params_
        self.best_score_ = gs.best_score_


# ------------------------------------------------------------------------- #
#                   Load and Evaluate Grid Search results                   #
# ------------------------------------------------------------------------- #

# Path to grid search results pickle files
gs_path = os.path.join(dir_base, 'models', 'simple', 'f1_weighted')
print("Evaluate Grid Search output\n")

# KNN
gs_knn = joblib.load(os.path.join(gs_path, 'direct-SKB-KNN.pkl'))
evaluate.gs_output(gs_knn)
# add cv results to a dataframe 
df_cv_knn = pd.DataFrame(gs_knn.cv_results_)
df_cv_knn['train_test_gap'] = df_cv_knn['mean_train_score'] - df_cv_knn['mean_test_score']
opt = df_cv_knn.filter(like = "param_").columns.to_list()
for col in opt: print(col, len(df_cv_knn[col].unique()), df_cv_knn[col].unique())
print('Hyperparameters combinations: ', df_cv_knn.shape[0])


# RANDOM FOREST
gs_rf =  joblib.load(os.path.join(gs_path, 'direct-SKB-RF.pkl'))
evaluate.gs_output(gs_rf)
# add cv results to a dataframe 
df_cv_rf = pd.DataFrame(gs_rf.cv_results_)
df_cv_rf['train_test_gap'] = df_cv_rf['mean_train_score'] - df_cv_rf['mean_test_score']
opt = df_cv_knn.filter(like = "param_").columns.to_list()
for col in opt: print(col, len(df_cv_knn[col].unique()), df_cv_knn[col].unique())
print('Hyperparameters combinations: ', df_cv_rf.shape[0])

# GRADIENT BOOSTED TREES
gs_gbt = joblib.load(os.path.join(gs_path, 'direct-SKB-GBT.pkl'))
evaluate.gs_output(gs_gbt)
# add cv results to a dataframe 
df_cv_gbt = pd.DataFrame(gs_gbt.cv_results_)
df_cv_gbt['train_test_gap'] = df_cv_gbt['mean_train_score'] - df_cv_gbt['mean_test_score']
opt = df_cv_knn.filter(like = "param_").columns.to_list()
for col in opt: print(col, len(df_cv_knn[col].unique()), df_cv_knn[col].unique())
#
print('Hyperparameters combinations: ', df_cv_gbt.shape[0])


# ------------------------------------------------------------------------- #
#                             Import Test data                              #
# ------------------------------------------------------------------------- #
# directory where the datasets are located
df_dir = os.path.join(dir_base, 'data', 'simple')
# imports all datasets in directory
df = imports.posture(df_dir, 'df5_12-test')


# ------------------------------------------------------------------------- #
#                    Calculating f-scores in Test Set                       #
# ------------------------------------------------------------------------- #
X_true = df.iloc[:,:-5]
y_true = df["Position"]
labels = y_true.unique()

# Use estimator to predict on test set
y_knn = gs_knn.best_estimator_.predict(X_true)
y_rf = gs_rf.best_estimator_.predict(X_true)
y_gbt = gs_gbt.best_estimator_.predict(X_true)

## Calculate f1-scores
df_f1_class = pd.DataFrame({
    'label':  labels,
    'knn_score': f1_score(y_true, y_knn, average = None),
    'rf_score': f1_score(y_true, y_rf, average = None),
    'gbt_score': f1_score(y_true, y_gbt, average = None)
})
print(df_f1_class)

df_f1_avg = pd.DataFrame({
    'estimator': ['knn', 'rf', 'gbt'],
    'macro': [f1_score(y_true, y_knn, average = 'macro'),
                f1_score(y_true, y_rf, average = 'macro'), 
                f1_score(y_true, y_gbt, average = 'macro')
                ],
    'micro': [f1_score(y_true, y_knn, average = 'micro'),
                f1_score(y_true, y_rf, average = 'micro'), 
                f1_score(y_true, y_gbt, average = 'micro')
                ],
    'weighted': [f1_score(y_true, y_knn, average = 'weighted'),
                f1_score(y_true, y_rf, average = 'weighted'), 
                f1_score(y_true, y_gbt, average = 'weighted')
                ],
})
print(df_f1_avg)

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_true, y_knn, xticks_rotation = 45)
ConfusionMatrixDisplay.from_predictions(y_true, y_rf, xticks_rotation = 45)
ConfusionMatrixDisplay.from_predictions(y_true, y_gbt, xticks_rotation = 45)

