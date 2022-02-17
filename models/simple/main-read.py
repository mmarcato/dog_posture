''' Copyright (C) 2022 by Marinara Marcato
         <marinara.marcato@tyndall.ie>, Tyndall National Institute
        University College Cork, Cork, Ireland.
'''
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.base import BaseEstimator, TransformerMixin
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
import analyse, learn, evaluate 


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

# Loading Grid Search Results from Pickle file

# run name
run = 'direct-SKB-RF.pkl'
gs_path = os.path.join(dir_base, 'models', 'simple', run)
gs = joblib.load(gs_path)
print("Evaluate Grid Search output\n")
# print best estimator results
evaluate.gs_output(gs)

# add cv results to a dataframe 
df_cv = pd.DataFrame(gs.cv_results_)
df_cv['train_test_gap'] = df_cv['mean_train_score'] - df_cv['mean_test_score']
print(df_cv.filter(like = "param").columns.to_list())
print(df_cv.shape)


# ------------------------------------------------------------------------- #
#                              Analyse Results                              #
# ------------------------------------------------------------------------- #

## SELECT K BEST ANALYSIS
print("SELECT K BEST ANALYSIS\n\nUnique K values, test and train scores: ")
print(df_cv.groupby(['param_slt__k'])[['mean_test_score', 'mean_train_score']].mean())


## K NEAREST NEIGHBOURS ANALYSIS
print("K NEAREST NEIGHBOURS ANALYSIS\n\nUnique n_neighbor values, test and train scores: ")
print(df_cv.groupby(['param_clf__n_neighbors'])[['mean_test_score', 'mean_train_score']].mean())


## RANDOM FOREST ANALYSIS
# Calculate mean test score value while maintaining one parameter constant at a time
print("RANDOM FOREST ANALYSIS\n\nUnique values, test and train scores: ")
print(df_cv.groupby(['param_clf__max_depth'])[['mean_test_score', 'mean_train_score']].mean())
print(df_cv.groupby(['param_clf__n_estimators'])[['mean_test_score', 'mean_train_score']].mean())

# test whether there is a significant difference between
# different selected values for a hyperparameter
var = 'param_slt__k'
print(df_cv[var].unique())
data1 = df_cv.loc[df_cv[var] == 10, 'mean_test_score']
data2 = df_cv.loc[df_cv[var] == 50, 'mean_test_score']
# Example of the Student's t-test
from scipy.stats import ttest_ind
stat, p = ttest_ind(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))


# ------------------------------------------------------------------------- #
#                              Plot Results                              #
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

# Evaluate Feature Importance from Select K Features
df_ft = pd.DataFrame({'Feature': gs.best_estimator_['ft'].attribute_names, 
        'Importance' : gs.best_estimator_['slt'].scores_})
df_ft.sort_values(by = 'Importance', ascending = False, inplace = True, ignore_index = True)

# Plotting 
slt_ft = gs.best_params_['slt__k']
plt.figure(figsize= (20, 8))
plt.bar(df_ft.loc[:slt_ft,'Feature'],df_ft.loc[:slt_ft, 'Importance'])
plt.xticks(rotation = 45)
plt.title('Best {} Features and their importances using SKB'.format(slt_ft))
plt.xlabel('Features')
plt.ylabel('Importance')
plt.savefig('{}/results/SKB-BestEstimator_BestFeatures'.format(dir_base))
# important features from the best 
rf_ft = list(df_ft.loc[:14, 'Feature'])