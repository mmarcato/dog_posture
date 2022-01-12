# ------------------------------------------------------------------------- #
#                                  Imports                                  #    
# ------------------------------------------------------------------------- # 
## General imports
import sys, os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#pd.set_option('display.max_rows', None)
import numpy as np
#np.random.seed(42)

# Setting up caching to save models
import joblib


# automatic feature selection 
from kneed import KneeLocator

## sklearn imports
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.decomposition import PCA

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

## Local imports
%load_ext autoreload
%autoreload 2
from __modules__ import imports
from __modules__ import process
from __modules__ import learn
from __modules__ import evaluate 


# ------------------------------------------------------------------------- #
#                           Importing Datasets                              #    
# ------------------------------------------------------------------------- #
# directory where the datasets are located
df_dir = 'C:\\Users\\marinara.marcato\\Project\\Scripts\\dog_posture\\data\\tsfel-1'
# imports all dataframes in directory
df_feat = imports.features_tsfel(df_dir)

# creating dev and test sets
df_dev, df_test = process.split(df_feat, 0.2)
df_train, df_val = process.split(df_dev, 0.25)
df = df_train

# define all features and - magnetometer 
feat_all = df.columns[:-4]


# ------------------------------------------------------------------------- #
#                        Exploratory Data Analysis                          #    
# ------------------------------------------------------------------------- #

 
# --------------------------- EDA - Dog Imbalance ------------------------- #   

# Development set Position - calculating the number of examples per category
df_dog = df['Dog'].unique()
 

# -----------------------   EDA - Class Imbalance  ------------------------ # 

# class balance
df['Type'].value_counts()
# class balance
df['Position'].value_counts()

# Development set Position - calculating the number of examples per category
df_pos = df['Position'].value_counts().reset_index(name= 'count')
# Development set Position - calculating the percentage of examples per category
df_pos['percentage'] = df_pos['count']*100 /df_pos['count'].sum()

# Plot percentage of points per category
plt.bar(df_pos['index'], df_pos['count'])
plt.xlabel('Position')
plt.xticks(rotation = 45)
plt.ylabel('Number of examples')

# Development set Position - calculating the number of examples per category
df_type = df['Type'].value_counts().reset_index(name= 'count')
# Development set Position - calculating the percentage of examples per category
df_type['percentage'] = df_pos['count']*100 /df_pos['count'].sum()

# Plot percentage of points per category
plt.bar(df_type['index'], df_type['count'])
plt.xlabel('Type')
plt.xticks(rotation = 45)
plt.ylabel('Number of examples') 
            
# ------------------------------------------------------------------------- #
#                       Feature Selection - Functions                       #    
# ------------------------------------------------------------------------- #

# cross validation function for feature selection
def fs_cv(df, feat, pipe): 
    return(cross_validate(
            estimator = pipe, 
            X = df.loc[:, feat], 
            y = df.loc[:, 'Position'].values, 
            groups = df.loc[:,'Dog'],
            cv= GroupKFold(n_splits = 10), 
            scoring = 'f1_weighted', 
            return_train_score=True,
            return_estimator = True
        ))

def cv_importances(score, feat_all):
    df_feat = pd.DataFrame(index =  feat_all)
    for fold in score['estimator']:
        ##print("\n\n FOLD!")
        feat_corr = fold['correlation'].get_support()
        feat_var = fold['variance'].get_support()
        #print(feat_corr[:5], len(feat_corr))
        #print(feat_var[:5], len(feat_var) )
        features = np.array(feat_corr)[np.array(feat_var)]
        importances = fold['classifier'].feature_importances_
        #print(features[:10], len(features))
        #print(importances[:10], len(importances), importances.sum())
        df_ft = pd.DataFrame( data = importances, index = features, columns = ['values'])
        df_feat = df_feat.merge(df_ft, how = 'outer', left_index = True, right_index = True)
    df_feat['mean'] = df_feat.mean(axis = 1)
    df_feat.sort_values(by = 'mean', ascending = False, inplace = True)
    return (df_feat)

def ft_knee(df_feat):
    """
    Uses Knee locator algorithm to find optimal number of features
    considering the importance given by a classifier
        https://github.com/arvkevi/kneed

    Parameters:
        df_feat (pd.Dataframe): [description]
    Returns:
    """
    y = df_feat['mean'].dropna()
    x = range(len(y))

    kn = KneeLocator(
        x, 
        y, 
        curve = 'convex', 
        direction = 'decreasing', 
        interp_method='polynomial')

    print('Number of features until knee:', kn.knee)
    print('Feature importances sum until the knee:', df_feat['mean'][:kn.knee].sum())
    plt.plot(x,y)
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    return(kn.knee)

def print_scores(score): 
    print('\nfit_time', score['fit_time'].mean(), 'score_time', score['score_time'].std())
    print('train_score', score['train_score'].mean(), score['train_score'].std())
    print('test_score', score['test_score'].mean(), score['test_score'].std())

''' Not working, may not be necessary 
    DEPENDS ON TOO MANY THINGS XD
    def gs_importances(estimator, feat_all):
        df_feat = pd.DataFrame(index =  feat_all)
        feat_corr = estimator['variance'].get_support()
        feat_var = estimator['selection'].get_support()
        print(feat_corr[:5], len(feat_corr))
        print(feat_var[:5], len(feat_var), sum(feat_var) )
        features = np.array(feat_corr)[np.array(feat_var)]
        #features = 
        print(features[:5], len(features), sum(features) )
        importances = estimator['classifier'].feature_importances_
        print(features[:10], len(features))
        print(importances[:10], len(importances), importances.sum())
        df_ft = pd.DataFrame( data = importances, index = features, columns = ['values'])
        df_feat.sort_values(by = 'values', ascending = False, inplace = True)
        return (df_feat)
    gs_feat = gs_importances(pipe_score.best_estimator_, feat_all)'''


# ------------------------------------------------------------------------- #
#                        Feature Selection - Main                           #    
# ------------------------------------------------------------------------- #
rf_pipe1 = Pipeline([
        ('correlation', learn.CorrelationThreshold()),
        ('variance', VarianceThreshold()),
        ('classifier', RandomForestClassifier(random_state= 42))
    ])
rf_pipe2 = Pipeline([
        ('correlation', learn.CorrelationThreshold()),
        ('variance', VarianceThreshold()),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state= 42))
    ])

# using df_train to check on the Feature Importances for the RF classifier
# this will help me pick an optimal  number for the feature selection algorithm
rf_score1 = fs_cv(df, feat_all, rf_pipe1)
rf_score2 = fs_cv(df, feat_all, rf_pipe2)

# pipeline with scaling steps showed better results
print_scores(rf_score1)
print_scores(rf_score2)

# getting the optimal number of important features 
kn_knee1 = ft_knee(ft_importances(rf_score1))
kn_knee2 = ft_knee(ft_importances(rf_score2))

# set up the pipelines for learning
pipe = Pipeline([
        ('correlation', learn.CorrelationThreshold()),
        ('variance', VarianceThreshold()),   
        ('scaler', StandardScaler()),
        ('selection' , 'passthrough'),
        ('reduction',  'passthrough'),
        ('classifier', RandomForestClassifier(random_state= 42))       
    ]) 

# selects kn_knee2 features
params = {
        'selection': [  SelectKBest(f_classif, k=kn_knee2),
                        SelectFromModel(
                                RandomForestClassifier(random_state= 42), 
                                max_features = kn_knee2)], 
        'reduction': [  'passthrough', 
                        PCA()]
        }

pipe_score = evaluate.gs_perf(pipe, 
                params,
                X = df_dev.loc[:, feat_all], 
                y = df_dev.loc[:, 'Position'].values, 
                groups = df_dev.loc[:,'Dog'],
                # should I have done it this way instead ?
                # GroupKFold(n_splits = 10).split(X, y, groups = df.loc[:,'Dog'])
                cv= GroupKFold(n_splits = 10) 
                )

evaluate.gs_output(pipe_score)
# Saving Grid Search Results to pickle file 
model_dir = 'C:\\Users\\marinara.marcato\\Project\\Scripts\\dog_posture\\models\\paper'
model_name = 'FS-Simple'
joblib.dump(pipe_score, 
            '{}\\{}.pkl'.format(model_dir, model_name),
            compress = 1 )

# variable to add caching to pipeline, handy when fitting transformer is costly
#location = 'cachedir'
#memory = joblib.Memory(location=location, verbose=10)
# Delete the temporary cache before exiting
#memory.clear(warn=False)
#rmtree(location)


# Loading Grid Search Results from Pickle file
gs = joblib.load('{}\\{}.pkl'.format(model_dir, model_name))
evaluate.gs_output(gs)



''' Not used
    # inserting a random variable for feature importance comparison
    df.insert(0,                # position
                'random',       # column name
                np.random.RandomState(1234).uniform(low=0, high=1, size = (df.shape[0]),)) 
'''
# selects estimator from cross_validate, then one of the folds
# selects the step in the estimator, then random forest classifier
# then the feature importances
importance_rf = pd.DataFrame({
                    'features': feat, 
                    'importance_rf': rf_score['estimator'][0].steps[0][1].feature_importances_
                })
importance_rf.sort_values(by  = 'importance_rf', 
                        ascending = False,
                        ignore_index = True, 
                        inplace = True)

importance_kb = pd.DataFrame({
                    'features': feat[kb_score['estimator'][1].steps[0][1].get_support()],
                    'importance_kb': kb_score['estimator'][1].steps[1][1].feature_importances_
                })
importance_kb.sort_values(by  = 'importance_kb', 
                            ascending = False,
                            ignore_index = True, 
                            inplace = True)

importance_sm = pd.DataFrame({
                    'features': feat[sm_score['estimator'][1].steps[0][1].get_support()],
                    'importance_sm': sm_score['estimator'][1].steps[1][1].feature_importances_
                })
importance_sm.sort_values(by  = 'importance_sm', 
                            ascending = False,
                            ignore_index = True, 
                            inplace = True)

plt.bar(importance_rf['features'], importance_rf['importance_rf'], label = 'rf')
plt.bar(importance_kb['features'], importance_kb['importance_kb'], label = 'kb')
plt.bar(importance_sm['features'], importance_sm['importance_sm'], label = 'sm')
# horizontal line at the importance threshold chosen with l1 penalty
plt.axhline(sm_score['estimator'][1].steps[0][1].threshold_, color = 'r')
# vertical line at the end of the 
plt.axvline(len(importance_sm), color = 'r')
plt.legend()
plt.ylabel('feature importances')
plt.xlabel('feature names')
plt.show()

importance = pd.merge(importance_rf, importance_kb, on = 'features', how = 'outer')
importance = pd.merge(importance, importance_sm, on = 'features', how = 'outer')