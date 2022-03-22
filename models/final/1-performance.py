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
#                      F1-scores for MODELS on Test Set                     #
# ------------------------------------------------------------------------- #

# load grid search results from pickle file
gs_path = os.path.join(dir_base, 'models', 'final')
gs_direct =  joblib.load(os.path.join(gs_path, '1-direct-SKB-RF.pkl'))
evaluate.gs_output(gs_direct)

# ------------------------------------------------------------------------- #
#                  F1-scores for EXPERIMENT 1 on Test Set                   #
# ------------------------------------------------------------------------- #

y_test = evaluate.exp1(df_test, gs_direct)
df_test_metrics = evaluate.metrics(y_test)
y_test.to_csv(os.path.join(dir_base, 'results', 'exp1-test-predictions.csv'))

y_golden = evaluate.exp1(df_golden, gs_direct)
df_golden_metrics = evaluate.metrics(y_golden)
y_golden.to_csv(os.path.join(dir_base, 'results', 'exp1-golden-predictions.csv'))

## AUC
y_true_d = pd.get_dummies(y_true)
y_pred_d = pd.get_dummies(y_pred)

print('\nAUC scores')
print('macro: {:0.4f}'.format(roc_auc_score(y_true_d, y_pred_d, 
        labels = y_true_d.columns, average = 'macro', multi_class = 'ovo')))
print('micro: {:0.4f}'.format(roc_auc_score(y_true_d, y_pred_d, 
        labels = y_true_d.columns, average = 'micro', multi_class = 'ovo')))
print('weighted: {:0.4f}'.format(roc_auc_score(y_true_d, y_pred_d, 
        labels = y_true_d.columns, average = 'weighted', multi_class = 'ovo')))

# ------------------------------------------------------------------------- #
#                         Analyse GS Hyperparameters                        #
# ------------------------------------------------------------------------- #

# add cv results to a dataframe 
df_cv = pd.DataFrame(gs_direct.cv_results_)
df_cv['train_test_gap'] = df_cv['mean_train_score'] - df_cv['mean_test_score']
opt = df_cv.filter(like = "param_").columns.to_list()
for col in opt: print(col, len(df_cv[col].unique()), df_cv[col].unique())
print('Hyperparameters combinations: ', df_cv.shape[0])

## SELECT K BEST ANALYSIS
print("SELECT K BEST ANALYSIS\n\nUnique K values, test and train scores: ")
print(df_cv.groupby(['param_slt__k'])[['mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score', 'train_test_gap']].mean())


## RANDOM FOREST ANALYSIS
# Calculate mean test score value while maintaining one parameter constant at a time
print("RANDOM FOREST ANALYSIS\n\nUnique values, test and train scores: ")
print(df_cv.groupby(['param_clf__max_depth'])[['mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score', 'train_test_gap']].mean())
print(df_cv.groupby(['param_clf__n_estimators'])[['mean_test_score', 'mean_train_score', 'train_test_gap']].mean())


# TEST SIGNIFICANT OF RESULTS
# test whether there is a significant difference between
# different selected values for a hyperparameter
var = 'param_slt__k'
print(df_cv[var].unique())
data1 = df_cv.loc[df_cv[var] == 10, 'mean_test_score']
data2 = df_cv.loc[df_cv[var] == 80, 'mean_test_score']
# Example of the Student's t-test
from scipy.stats import ttest_ind
stat, p = ttest_ind(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))

# find the hyperparameter combination that yields the smalles train_test_gap
df_cv.sort_values('train_test_gap', ascending = False)\
        .loc[:,['param_clf__max_depth', 'param_clf__n_estimators', 'param_slt__k', 'mean_train_score', 'mean_test_score',]]


# ------------------------------------------------------------------------- #
#                           Plot GS Results                              #
# ------------------------------------------------------------------------- #

# PLOTTING MEAN TRAIN MINUS TEST SCORE VS. HYPER-PARAMETERS
var ="param_clf__max_depth"
for val in df_cv[var].unique():
    print(var, val)
    df = df_cv.loc[df_cv[var]== val]
    var1 =  'param_slt__k'
    var2 = "param_clf__n_estimators"
    sns.catplot(data=df, kind="bar",
        x=var1, hue=var2, y="train_test_gap", 
        ci="sd", palette="dark", alpha=.6, height=6 )

# PLOTTING MEAN TEST SCORE VS HYPER-PARAMETERS
var ='param_slt__k'
for val in df_cv[var].unique():
    print(var, val)
    df = df_cv.loc[df_cv[var]== val]
    var1 = "param_clf__n_estimators"
    var2 = "param_clf__max_depth"
    sns.catplot(data=df, kind="bar",
        x=var1, hue=var2, y="mean_test_score", 
        ci="sd", palette="dark", alpha=.6, height=6 )

# ------------------------------------------------------------------------- #
#                        Feature Importance Analysis                        #
# ------------------------------------------------------------------------- #

##### ALL FEATURES

# Evaluate Feature Importance from Select K Features SKB
df_ft = pd.DataFrame({'Feature': gs_direct.best_estimator_['ft'].attribute_names, 
                'SKB Score': gs_direct.best_estimator_['slt'].scores_,
                'SKB P-value': gs_direct.best_estimator_['slt'].pvalues_})
df_ft['SKB Score_%'] = df_ft['SKB Score'] / df_ft['SKB Score'].sum()

# feature columns
df_ft['Position'], df_ft['Sensor'], df_ft['Axis'] = df_ft.Feature.str.split('.',2).str
df_ft['Axis'], df_ft['Type'] = df_ft.Axis.str.split('_',1).str
df_ft['Type'] = df_ft.Type.str.split('_', 1).str[0]

# RF Feature importance information (imu, sensor and Type)
print('SKB Score_%')
print(df_ft.groupby(['Position'])['SKB Score_%'].sum().round(2))
print(df_ft.groupby(['Position','Sensor'])['SKB Score_%'].sum().reset_index().round(2))

print(df_ft.groupby(['Position','Sensor', 'Axis'])['SKB Score_%'].sum())

print(df_ft.groupby(['Type'])['SKB Score_%'].sum().sort_values(ascending = False))



### SKB SELECTED FEATURES
## SKB and RF importances
idx = gs_direct.best_estimator_['slt'].get_support(indices=True)
df_slt = pd.DataFrame({'Feature': gs_direct.best_estimator_['ft'].attribute_names[idx],
                'SKB Score': gs_direct.best_estimator_['slt'].scores_[idx],
                'SKB P-value': gs_direct.best_estimator_['slt'].pvalues_[idx],
                'RF Importance': gs_direct.best_estimator_['clf'].feature_importances_
        })
df_slt['SKB Score_%'] = df_slt['SKB Score'] / df_slt['SKB Score'].sum()

df_slt['Position'], df_slt['Sensor'], df_slt['Axis'] = df_slt.Feature.str.split('.',2).str
df_slt['Axis'], df_slt['Type'] = df_slt.Axis.str.split('_',1).str
df_slt['Type'] = df_slt.Type.str.split('_', 1).str[0]

# SKB Feature importance information (imu, sensor and Type)
print('RF Importance_%')
print(df_slt.groupby(['Position'])['RF Importance'].sum().round(2))
print(df_slt.groupby(['Position','Sensor'])['RF Importance'].sum().round(2))
# print(df_slt.groupby(['IMU','Sensor', 'Axis'])['RF Importance'].sum())
df_imp = df_slt.groupby(['Type'])['RF Importance'].sum().sort_values(ascending = False).reset_index()

from tsfel.feature_extraction.features_settings import get_features_by_domain as ft_domain
my_dict = {
        'Type': {
        'temporal':
                list(ft_domain('temporal')['temporal'].keys()),
        'statistical':
                list(ft_domain('statistical')['statistical'].keys()),
        'spectral':
                list(ft_domain('spectral')['spectral'].keys())
}       }
ft_domain = pd.DataFrame(my_dict)
ft_domain = ft_domain.explode('Type').rename_axis('Domain').reset_index()
print('Number of original features per domain:\n', ft_domain['Domain'].value_counts())


df_ft = pd.merge(df_imp, ft_domain)
df1  = df_ft.groupby(['Domain']).sum().reset_index()
df2 = df_ft.groupby(['Domain'])['Type'].unique().reset_index()
df = pd.merge(df1, df2)
df['Number'] = df['Type'].map(lambda x: len(x))
print(df)
print(list(df.loc[0,'Type']))
print(list(df.loc[1,'Type']))
print(list(df.loc[2,'Type']))


# plotting
plt.plot(df_slt['SKB P-value'])
plt.plot(df_slt['RF Importance_%'])
plt.hist(df_slt['SKB Importance_%'], bins = 20)
plt.hist(df_slt['RF Importance_%'], bins = 20)


# Plotting feature importances
plt.figure(figsize= (20, 8))
plt.bar(df_slt.loc[:,'Feature'], df_slt.loc[:, 'Importance_%'])
plt.xticks(rotation = 45)
plt.title('Best {} Features and their importances using SKB'.format(slt_no))
plt.xlabel('Features')
plt.ylabel('Importance')
plt.savefig('{}/results/SKB-BestEstimator_BestFeatures'.format(dir_base))

