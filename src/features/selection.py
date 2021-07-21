# ------------------------------------------------------------------------- #
#                                  Imports                                  #    
# ------------------------------------------------------------------------- # 
## General imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)
from sklearn.feature_selection import SelectKBest, f_classif

## Local imports
%load_ext autoreload
%autoreload 2
from src.__modules__ import imports
from src.__modules__ import process 
from src.__modules__ import learn 
from src.__modules__ import evaluate 


# ------------------------------------------------------------------------- #
#                           Importing Datasets                              #    
# ------------------------------------------------------------------------- #
# directory where the dataset is located
df_dir = ('..//..//data//processed')

# importing previously created dataset
df_feat = imports.posture(df_dir, 'df_12')  

# creating dev and test sets
df_dev, df_test = process.split(df_feat, 0.2)
df_train, df_val = process.split(df_dev, 0.25)

df = df_train


# ------------------------------------------------------------------------- #
#                        Exploratory Data Analysis                          #    
# ------------------------------------------------------------------------- #

 
# ------------------------------------------------------------------------- #
#                           EDA - Dog Imbalance                             #    
# ------------------------------------------------------------------------- #

# Development set Position - calculating the number of examples per category
df_dog = df['Dog'].unique()
 
# ------------------------------------------------------------------------- #
#                          EDA - Class Imbalance                            #    
# ------------------------------------------------------------------------- #


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
#                        EDA - Variable Correlation                         #    
# ------------------------------------------------------------------------- #
df_corr = df.iloc[:,:-4].corr().stack().reset_index()

# rename the columns
df_corr.columns = ['f1', 'f2', 'correlation']

# create a mask to identify rows with duplicate features
mask_dups = (df_corr[['f1', 'f2']]\
                .apply(frozenset, axis=1).duplicated()) | \
                (df_corr['f1']==df_corr['f2']) 

# apply the mask to clean the correlation dataframe
df_corr = df_corr[~mask_dups]

print(df_corr['correlation'].describe())
df_corr['correlation'].hist()

df_corr.loc[df_corr['correlation']>0.99, :].sort_values('correlation', ascending = False)

idx = (df_corr['correlation']<0.99) & (df_corr['correlation']>0.95)
df_corr.loc[idx, :].sort_values('correlation', ascending = False)



# ------------------------------------------------------------------------- #
#                        EDA - p-values                         #    
# ------------------------------------------------------------------------- #


# ------------------------------------------------------------------------- #
#                            Feature Selection                              #    
# ------------------------------------------------------------------------- #

# generate random variable and add it to the dataframe
r = np.random.RandomState(1234).uniform(low=0, high=1, size = (df.shape[0]),)
df.insert(0,               # position
               'random',        # columns name
               r)               # data
# features used for learning
feat =  df.columns[:-4]

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

rf_pipe = Pipeline([
        ('classifier', RandomForestClassifier() )       
    ]) 

kb_pipe = Pipeline([
        ('kb' , SelectKBest(f_classif, k=50)),
        ('classifier', RandomForestClassifier())       
    ]) 

# set up the pipelines for learning
sm_pipe = Pipeline([
        ('sm', SelectFromModel(RandomForestClassifier())),
        ('classifier', RandomForestClassifier() )       
    ]) 
    
rf_score = fs_cv(df, feat, rf_pipe)
kb_score = fs_cv(df, feat, kb_pipe)
sm_score = fs_cv(df, feat, sm_pipe)

def print_scores(score): 
    print('\nfit_time', score['fit_time'].mean(), 'score_time', score['score_time'].std())
    print('train_score', score['train_score'].mean(), score['train_score'].std())
    print('test_score', score['test_score'].mean(), score['test_score'].std())

print_scores(rf_score)
print_scores(kb_score)
print_scores(sm_score)


# selects estimator from cross_validate, then one of the folder
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