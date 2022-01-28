''' 
    Copyright (C) 2021 by Marinara Marcato
         <marinara.marcato@tyndall.ie>, Tyndall National Institute
        University College Cork, Cork, Ireland.
'''

# ------------------------------------------------------------------------- #
#                                  Imports                                  #    
# ------------------------------------------------------------------------- # 
import os
import pandas as pd
import numpy as np
np.random.seed(42)

# Machine Learning Libraries

# Caching Libraries
import joblib
from shutil import rmtree
location = 'cachedir'
memory = joblib.Memory(location=location , verbose=10)

# Define local directories
dir_current = os.path.dirname(os.path.realpath(__file__))
dir_parent = os.path.dirname(dir_current)
dir_base = os.path.dirname(dir_parent)
dir_modules = os.path.join(dir_parent, '__modules__')
# Set path variable
sys.path.append(dir_modules)

# Local Modules
%load_ext autoreload
%autoreload 2
import imports
import process 
import analyse
import learn
import evaluate 



# ------------------------------------------------------------------------- #
#                           Define directories                              #    
# ------------------------------------------------------------------------- #

# directory where the dataset is located
dir_df = (dir_base + '\\data\\simple')
# directory to save the model
dir_model = (dir_base + '\\models')

# ------------------------------------------------------------------------- #
#                           Importing Datasets                              #    
# ------------------------------------------------------------------------- #
# importing previously created dataset
df_feat = imports.posture(dir_df, 'df5_11')  

# define all features and - magnetometer 
feat_all = df_feat.columns[:-5]
feat_mag = [x for x in feat_all if "Mag" not in x]
# select the feature set
feat = feat_mag

# ------------------------------------------------------------------------- #
#                     Split Datasets (Dog and Breed)                        #    
# ------------------------------------------------------------------------- #

# separating golden retriever 'Tosh'
df_gr = df_feat.loc[df_feat['Breed'] == 'GR']

# test set with 20% of observations, 60% LRxGR (Douglas, Elf, Goober) 40% LR (Meg, July)
df_test = df_feat[df_feat.Dog.isin(['Douglas', 'Elf', 'Goober', 'Meg', 'July'])]
# dogs for dev set for 80% observation, 60% LRxGR (Douglas, Elf, Goober) 40% LR (Meg, July) 
df_dev = df_feat[~df_feat.Dog.isin(['Tosh', 'Douglas', 'Elf', 'Goober', 'Meg', 'July'])]

# select the dataframe for grid search
df = df_dev

# ------------------------------------------------------------------------- #
#                      Exploratory Data Analysis                            #    
# ------------------------------------------------------------------------- # 

# breeds
analyse.breed(df_feat)
analyse.breed(df_test)
analyse.breed(df_dev)

# observations

# ------------------------------------------------------------------------- #
#                Machine Learning - Label 'Positions'                       #    
# ------------------------------------------------------------------------- # 

# prepare dataframe for evaluation: select features, label,
#   cv strategy (group = dogs, stractified folds labels proportion)
X, y, groups, cv = learn.df_prep(df, feat, label = 'Position')

selector = {
        'passthrough' : 'passthrough',
        'SKB' : SelectKBest(score_func= f_classif),
        'SVM': SelectFromModel(LinearSVC())
}

selector_hyper = {

        'SKB' : {
                    'slt__k': [10, 15, 20, 30, 50, 80]},

        'SVM' : {
                    'slt__penalty': ['l1', 'l2'],
                    'slt__C' : [0.01, 1, 100, 1000, 10000]},
}

classifier = {
        # 'LogisticRegression'     : LogisticRegression(),
        'RF' : RandomForestClassifier(),
        'KNN': KNeighborsClassifier()
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
        
        'RF':{
                'clf__max_features': [None],
                'clf__max_depth': [3, 5, 7, 10],
                'clf__n_estimators': [25, 50, 100, 250, 500],
                'clf__class_weight': ['balanced'],
                'clf__random_state': [0]},
       
        'KNN':{
                'clf__n_neighbors': [5, 10, 15, 20],
                'clf__weights': ['uniform', 'distance']}                        
}

slt_key = 'SKB'
clt_key = 'RF'

slt = selector[slt_key]
clf = classifier[clt_key]
params = { **selector_hyper[slt_key], **classifier_hyper[clt_key]}


pipe = Pipeline([ 
    ('selector', DataFrameSelector(feat,'float64')),
    ('scl', StandardScaler() ), 
    ('slt', slt),
    ('clf', clf )])



# evaluate grid search performance and save to pickle file
gs_rf = evaluate.gs_perf(pipe, params, X, y, groups, cv)

  


################## RANDOM FORESTS ##################
# set up the pipelines for learning
pipe1 = learn.SKB_RF
pipe1 = learn.SFM_SVC_RF(feat)
pipe2 = learn.SFM_ET_RF()


# define grid search parameters
params1 = {'estimator__max_depth' : [5, 10, 15, 20],
            'estimator__n_estimators' : [ 20, 40, 100, 160],
            'selection__k': [10, 15, 20, 25, 30, 35]}

params2 = {'estimator__max_depth' : [5, 10, 15, 20],
            'estimator__n_estimators' : [ 20, 40, 100, 160],
            'selection__penalty': ['l1', 'l2'],
            'selection__C' : [0.01, 1, 100, 1000, 10000]} 

      

# evaluate grid search performance and save to pickle file
gs_rf = evaluate.gs_perf(pipe, params, X, y, groups, cv)
joblib.dump(gs, '{}/{}.pkl'.format(dir_model, run), compress = 1 )
memory.clear(warn=False)
rmtree(location)


# load results from grid search 
gs_rf = joblib.load('{}/{}.pkl'.format(dir_model, 'RF-Test3'))

# Random Forest feature importance
df_ft = pd.DataFrame({'Feature': gs_rf.best_estimator_['selector'].attribute_names, 
        'Importance' : gs_rf.best_estimator_['estimator'].feature_importances_})
df_ft.sort_values(by = 'Importance', ascending = False, inplace = True, ignore_index = True)

# Plotting feature importance
# plt.plot(df_ft.loc[:14,'Feature'],df_ft.loc[:14, 'Importance'])
# plt.xticks(rotation = 45)

# important features from the best 
rf_ft = list(df_ft.loc[:14, 'Feature'])



################## K NEAREST NEIGHBOURS ##################

# set up the pipelines for learning with random forest feature set
pipe = learn.KNN(rf_ft)

# define grid search parameters
params = {
    'estimator__n_neighbors' : [40,50,60,70,80],
}

# evaluate grid seach performance
gs_knn = evaluate.gs_perf(pipe, params, X, y, groups, cv)
           # dir_model, run = 'KNN-Test3')


location = 'cachedir'
memory = joblib.Memory(location=location , verbose=10)

joblib.dump(gs_knn, '{}/{}.pkl'.format(dir_model, 'KNN-Test2'), compress = 1 )
memory.clear(warn=False)
rmtree(location)


evaluate.gs_output(gs_knn)



# ------------------------------------------------------------------------- #
#                   Load and Evaluate Grid Search results                   #    
# ------------------------------------------------------------------------- #

# Loading Grid Search Results from Pickle file
run = 'RF-Test3'
gs = joblib.load('{}/{}.pkl'.format(dir_model, run))
evaluate.gs_output(gs)

gs.cv_results_

# Calculate mean test score value while maintaining one parameter constant at a time
df_cv = pd.DataFrame(gs.cv_results_)
print(df_cv.groupby(['param_estimator__max_depth'])['mean_test_score'].mean())
print(df_cv.groupby(['param_estimator__n_estimators'])['mean_test_score'].mean())
print(df_cv.groupby(['param_estimator__max_features'])['mean_test_score'].mean())

print(df_cv.groupby(['param_estimator__max_depth'])['mean_train_score'].mean())
print(df_cv.groupby(['param_estimator__n_estimators'])['mean_train_score'].mean())
print(df_cv.groupby(['param_estimator__max_features'])['mean_train_score'].mean())

for depth in df_cv['param_estimator__max_depth'].unique():
    print(depth)
    df = df_cv.loc[df_cv['param_estimator__max_depth']== depth]
    sns.catplot(data=df, kind="bar",
        x="param_estimator__n_estimators", y="mean_test_score", hue="param_estimator__max_features",
        ci="sd", palette="dark", alpha=.6, height=6 )



