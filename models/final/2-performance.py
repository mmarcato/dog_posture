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


# ------------------------------------------------------------------------- #
#                           Load Grid Search results                        #
# ------------------------------------------------------------------------- #

# load grid search results from pickle file
gs_path = os.path.join(dir_base, 'models', 'final')

# TYPE MODEL
gs_type =  joblib.load(os.path.join(gs_path, '2-type-SKB-RF.pkl'))
gs_type =  joblib.load(os.path.join(gs_path, '2-type-SKB-RF-1.pkl'))
gs_type =  joblib.load(os.path.join(gs_path, '2-type-SKB-RF-uos.pkl'))
evaluate.gs_output(gs_type)

# STATIC MODEL
gs_static =  joblib.load(os.path.join(gs_path, '2-static-SKB-RF.pkl')) 
gs_static =  joblib.load(os.path.join(gs_path, '2-static-SKB-RF-1.pkl')) 
gs_static =  joblib.load(os.path.join(gs_path, '2-static-SKB-RF-uos.pkl')) 
df_static = df_test[df_test['Type'] == 'static']
evaluate.gs_output(gs_static)

# DYNAMIC MODEL
gs_dynamic =  joblib.load(os.path.join(gs_path, '2-dynamic-SKB-RF.pkl'))
gs_dynamic =  joblib.load(os.path.join(gs_path, '2-dynamic-SKB-RF-1.pkl'))
gs_dynamic =  joblib.load(os.path.join(gs_path, '2-dynamic-SKB-IF.pkl'))
df_dynamic = df_test[df_test['Type'] == 'dynamic']
evaluate.gs_output(gs_dynamic)

# ------------------------------------------------------------------------- #
#                      F1-scores for MODELS on Test Set                     #
# ------------------------------------------------------------------------- #

evaluate.model(X_true = df_test.iloc[:,:-6], 
        y_true = df_test.Type, 
        model = gs_type.best_estimator_)

evaluate.model(X_true = df_static.iloc[:,:-6], 
        y_true = df_static.Position, 
        model = gs_static.best_estimator_)

evaluate.model(X_true = df_dynamic.iloc[:,:-6], 
        y_true = df_dynamic.Position, 
        model = gs_dynamic.best_estimator_)

# ------------------------------------------------------------------------- #
#                  F1-scores for EXPERIMENT 2 on Test Set                   #
# ------------------------------------------------------------------------- #

### SIMPLE 
gs_type =  joblib.load(os.path.join(gs_path, '2-type-SKB-RF.pkl'))
gs_static =  joblib.load(os.path.join(gs_path, '2-static-SKB-RF.pkl'))
gs_dynamic =  joblib.load(os.path.join(gs_path, '2-dynamic-SKB-RF.pkl'))

## COMPLEX
gs_type =  joblib.load(os.path.join(gs_path, '2-type-SKB-RF-1.pkl'))
gs_static =  joblib.load(os.path.join(gs_path, '2-static-SKB-RF-1.pkl')) 
gs_dynamic =  joblib.load(os.path.join(gs_path, '2-dynamic-SKB-RF-1.pkl'))

## BEST
gs_type =  joblib.load(os.path.join(gs_path, '2-type-SKB-RF-uos.pkl'))
gs_static =  joblib.load(os.path.join(gs_path, '2-static-SKB-RF.pkl')) 
gs_dynamic =  joblib.load(os.path.join(gs_path, '2-dynamic-SKB-IF.pkl'))

y_test = evaluate.exp2(df_test, gs_type, gs_static, gs_dynamic)
df_test_metrics = evaluate.metrics(y_test)
y_test.to_csv(os.path.join(dir_base, 'results', 'exp2-best-test-predictions.csv'))

y_golden =  evaluate.exp2(df_golden, gs_type, gs_static, gs_dynamic)
df_golden_metrics = evaluate.metrics(y_golden)
y_golden.to_csv(os.path.join(dir_base, 'results', 'exp2-best-golden-predictions.csv'))

# ------------------------------------------------------------------------- #
#                         Analyse GS Hyperparameters                        #
# ------------------------------------------------------------------------- #

# add cv results to a dataframe 
gs = gs_type
df_cv = pd.DataFrame(gs.cv_results_)
df_cv['train_test_gap'] = df_cv['mean_train_score'] - df_cv['mean_test_score']
opt = df_cv.filter(like = "param_").columns.to_list()
for col in opt: print(col, len(df_cv[col].unique()), df_cv[col].unique())
print('Hyperparameters combinations: ', df_cv.shape[0])

## SELECT K BEST ANALYSIS
print("SELECT K BEST ANALYSIS\n\nUnique K values, test and train scores: ")
print(df_cv.groupby(['param_slt__k'])[['mean_test_score', 'mean_train_score', 'train_test_gap']].mean())


## RANDOM FOREST ANALYSIS
# Calculate mean test score value while maintaining one parameter constant at a time
print("RANDOM FOREST ANALYSIS\n\nUnique values, test and train scores: ")
print(df_cv.groupby(['param_clf__max_depth'])[['mean_test_score', 'mean_train_score', 'train_test_gap']].mean())
print(df_cv.groupby(['param_clf__n_estimators'])[['mean_test_score', 'mean_train_score', 'train_test_gap']].mean())
