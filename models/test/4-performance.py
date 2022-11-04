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

# select dataset  
ft = 'test'
# select model parameters
exp = 4
mdl = 'imbalanced'
slt = 'SKB'
clf = 'RF'
# run = '{}-{}-{}-{}.pkl'.format(exp, mdl, slt, clf)
param = 1 
run = '{}-{}-{}-{}-{}.pkl'.format(exp, mdl, slt, clf, param)

# load grid search results from pickle file
gs_path = os.path.join(dir_base, 'models', 'test')
gs_imb =  joblib.load(os.path.join(gs_path, run))
print('Evaluate Grid Search output\n')
evaluate.gs_output(gs_imb)

# ------------------------------------------------------------------------- #
#                             Import Test data                              #
# ------------------------------------------------------------------------- #
print('\n\nImport Test Set\n')
# directory where the datasets are located
df_dir = os.path.join(dir_base, 'data', 'test')

# select test dataframe and label according to model
# EXPERIMENT 1 - DEFAULT
label = 'Position'
df_test = imports.posture(df_dir, 'df-all-test')

# print dataframe stats
print('Model:', mdl)
print('Dataset:', df_test.shape)
print('Label:', df_test.groupby(label)[label].count())

# select test data for prediction
X_true = df_test.iloc[:,:-6]
y_true = df_test.loc[:, label]
labels = y_true.unique()

# ------------------------------------------------------------------------- #
#              Calculate f1-scores for model in Test Set                    #
# ------------------------------------------------------------------------- #
# Use best estimator to predict on test set
y_rf = gs_imb.best_estimator_.predict(X_true)

## Calculate f1-scores
df_f1_class = pd.DataFrame({
    'label':  labels,
    'f1_score': f1_score(y_true, y_rf, labels = labels, average = None)
})

print(df_f1_class)
print('\nf1 scores')
print('macro: {:0.4f}'.format(f1_score(y_true, y_rf, average = 'macro')))
print('micro: {:0.4f}'.format(f1_score(y_true, y_rf, average = 'micro')))
print('weighted: {:0.4f}'.format(f1_score(y_true, y_rf, average = 'weighted')))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_true, y_rf, xticks_rotation = 45)



# ------------------------------------------------------------------------- #
#                         Analyse GS Hyperparameters                        #
# ------------------------------------------------------------------------- #

# add cv results to a dataframe 
df_cv = pd.DataFrame(gs_imb.cv_results_)
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
print(df_cv.groupby(['param_clf__base_estimator__max_depth'])[['mean_test_score', 'mean_train_score', 'train_test_gap']].mean())
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
df_ft = pd.DataFrame({'Feature': gs_imb.best_estimator_['ft'].attribute_names, 
        'Importance' : gs_imb.best_estimator_['slt'].scores_})
df_ft.sort_values(by = 'Importance', ascending = False, inplace = True, ignore_index = True)

# Plotting 
slt_ft = gs_imb.best_params_['slt__k']
plt.figure(figsize= (20, 8))
plt.bar(df_ft.loc[:slt_ft,'Feature'],df_ft.loc[:slt_ft, 'Importance'])
plt.xticks(rotation = 45)
plt.title('Best {} Features and their importances using SKB'.format(slt_ft))
plt.xlabel('Features')
plt.ylabel('Importance')
plt.savefig('{}/results/SKB-BestEstimator_BestFeatures'.format(dir_base))
# important features from the best 
rf_ft = list(df_ft.loc[:slt_ft, 'Feature'])