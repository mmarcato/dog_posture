''' Copyright (C) 2022 by Marinara Marcato
         <marinara.marcato@tyndall.ie>, Tyndall National Institute
        University College Cork, Cork, Ireland.
'''
import os, sys, joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, roc_auc_score


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
#                             Import Test data                              #
# ------------------------------------------------------------------------- #

df_dir = os.path.join(dir_base, 'data', 'final')

print('\n\nImport Test and Golden Set\n')
df_test = pd.read_csv(os.path.join(df_dir, 'df-all-test.csv'), 
                parse_dates = ['Timestamp'],
                dayfirst = True,
                date_parser = lambda x: pd.to_datetime(x, format = '%Y-%m-%d %H:%M:%S.%f'))

df_golden = pd.read_csv(os.path.join(df_dir, 'df-golden.csv'), 
                parse_dates = ['Timestamp'],
                dayfirst = True,
                date_parser = lambda x: pd.to_datetime(x, format = '%Y-%m-%d %H:%M:%S.%f'))

df_tg = pd.concat([df_test, df_golden], ignore_index = True)
# ------------------------------------------------------------------------- #
#                           Load Grid Search results                        #
# ------------------------------------------------------------------------- #

# load grid search results from pickle file
gs_path = os.path.join(dir_base, 'models', 'final')

gs_anomaly =  joblib.load(os.path.join(gs_path, '3-anomaly-SKB-IF.pkl'))
evaluate.gs_output(gs_anomaly)

gs_normal =  joblib.load(os.path.join(gs_path, '3-normal-SKB-RF.pkl')) 
evaluate.gs_output(gs_normal)

# ------------------------------------------------------------------------- #
#                      F1-scores for MODELS on Test Set                     #
# ------------------------------------------------------------------------- #

### ANOMALY MODEL
evaluate.model(X_true = df_test.iloc[:,:-6], 
        y_true = df_test.Shake, 
        model = gs_anomaly.best_estimator_)

#### NORMAL MODEL
df_normal = df_test[df_test.Shake == 1]
evaluate.model(X_true = df_normal.iloc[:,:-6], 
        y_true = df_normal.Position, 
        model = gs_normal.best_estimator_)

# ------------------------------------------------------------------------- #
#                  F1-scores for EXPERIMENT 3 on Test Set                   #
# ------------------------------------------------------------------------- #
y_test = evaluate.exp3(df_test, gs_anomaly, gs_normal)
df_test_metrics = evaluate.metrics(y_test)
y_test.to_csv(os.path.join(dir_base, 'results', 'exp3-best-test-predictions.csv'))

y_golden = evaluate.exp3(df_golden, gs_anomaly, gs_normal)
df_golden_metrics = evaluate.metrics(y_golden)  
y_golden.to_csv(os.path.join(dir_base, 'results', 'exp3-best-golden-predictions.csv'))

y_tg = evaluate.exp3(df_tg, gs_anomaly, gs_normal)
df_tg_metrics = evaluate.metrics(y_tg)  
# y_golden.to_csv(os.path.join(dir_base, 'results', 'exp3-best-golden-predictions.csv'))

# ------------------------------------------------------------------------- #
#                                  Random                                   #    
# ------------------------------------------------------------------------- #


# Checking the body shakes 
# 3 of them (Douglas DC1 x 2 and July DC2) seem to be out of alignment in time (about 3s ahead)

df_test.loc[y_shake.index , ['Timestamp','Dog', 'DC', 'Position']]

df_test.loc[df_test['Position'] == 'body shake', ['Timestamp','Dog', 'DC', 'Position']]

y_pred[df_test['Position'] == 'body shake']
df_f1_class[df_f1_class['True'] == 'body shake']

# # AUC
# y_true_d = pd.get_dummies(y_true)
# y_normal_d = pd.get_dummies(y_normal)

# print('\nAUC scores')
# print('macro: {:0.4f}'.format(roc_auc_score(y_true_d, y_normal_d, 
#         labels = y_true_d.columns, average = 'macro', multi_class = 'ovo')))
# print('micro: {:0.4f}'.format(roc_auc_score(y_true_d, y_normal_d, 
#         labels = y_true_d.columns, average = 'micro', multi_class = 'ovo')))
# print('weighted: {:0.4f}'.format(roc_auc_score(y_true_d, y_normal_d, 
#         labels = y_true_d.columns, average = 'weighted', multi_class = 'ovo')))
