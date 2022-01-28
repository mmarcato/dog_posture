# Loading Grid Search Results from Pickle file
import joblib
import pandas as pd
import seaborn as sns
import evaluate


evaluate.gs_output(gs)

# pickle file and dataframe names
gs = joblib.load('../models/{}.pkl'.format('GS-GB-df_12'))

########################    Start: CV Results  ########################
cv = pd.DataFrame(gs.cv_results_)
cv = cv.melt(id_vars = ['param_estimator__max_depth', 'param_estimator__max_features', 'param_estimator__n_estimators'],\
    value_vars = ['mean_train_score','mean_test_score'])
    #,'std_train_score', 'std_test_score') )
sns.catplot(x = 'param_estimator__max_depth',\
    y = 'value', \
    hue = 	'variable', \
    col  = 'param_estimator__max_features', \
    row = 'param_estimator__n_estimators',\
    data = cv, kind = 'bar', height = 4, aspect =0.7\
    )
cv = cv[['param_estimator__max_depth','param_estimator__max_features', 'param_estimator__n_estimators','mean_train_score','mean_test_score','std_train_score', 'std_test_score']]
cv.sort_values('mean_test_score', ascending = False)
########################    End: CV Results  ########################

########################    Start: Feature importance   ########################

# Chest and Back variables seem to be equally important while neck not so much
# Another thing that I could analyse is feature correlation

ft_imp = gs.best_estimator_['estimator'].feature_importances_
ft_name = gs.best_estimator_['selector'].attribute_names 
df_imp = pd.DataFrame(data = {'importance' : ft_imp , 'feat' : ft_name })
df_imp = df_imp.sort_values(by = 'importance', ascending = False, ignore_index = True)
df_imp[['stat', 'location', 'sensor', 'axis' ]] = pd.DataFrame(df_imp['feat'].str.split('.').to_list())

sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(14,5)})
sns.catplot(x = 'sensor', y = 'importance', hue = 'stat', col  = 'location', 
                    data = df_imp, kind = 'bar', height = 4, aspect =0.7, )
sns.catplot(x = 'sensor', y = 'importance', hue = 'axis', col  = 'location', 
                    data = df_imp, kind = 'bar', height = 4, aspect =0.7)
sns.catplot(x = 'location', y = 'importance', data = df_imp, kind = 'bar')

# creating a list to add the feature importance 
dfs = []
for col in df_imp.columns[2:]: 
    # append dataframes with sum, mean and std of each of variable category as a list
    # sort values, drop importance level in multindex column, 
    # creates a column named category for the stat, location, sensor, axis  
    dfs.append(df_imp.groupby(col)\
        .agg({'importance': ['sum', 'mean','std']})\
        .sort_values(by = [('importance', 'sum')], ascending = False)\
        .droplevel(0, axis = 1)\
        .assign(category = col))
dfs = pd.concat(dfs)

## Stats: std > max > mean > sum ~ median > kurt ~ skew
# Location: Chest > Back >>>>> Neck
# Sensor: Acc >>> Gyr> Mag 
# Axis: z>y>x
sns.barplot(x = dfs.index, y = 'sum', hue = 'category', data = dfs, ci = 'std', \
            linewidth=0.1, errcolor =  'black', errwidth =2, capsize = .1, dodge = False)  

sns.barplot(x = dfs.index, y = 'mean', hue = 'category', data = dfs, ci = 'std', \
            linewidth=0.1, errcolor =  'black', errwidth =2, capsize = .1, dodge = False)  


########################    End: Feature importance   ########################

# Comparing Explained Variance Ratios (PCA)
f = sns.scatterplot(data = gs.best_estimator_['reduce_dim'].explained_variance_)
f.axhline(1, color = 'r')
plt.show()
evaluate.gs_output(gs)


''' Implementing grid search function'''

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from learn import DataFrameSelector 
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV

## Removing PCA and adding restriction to max_feature in RF
gs_name = 'GS-RF-df_32'
gs_params = {
    'estimator__max_depth' : [3, 5, 10], 
    'estimator__n_estimators' : [25, 35, 50], 
    'estimator__max_features' : [80, 100, 120], 
}

gs_pipe = Pipeline([
    ('selector', DataFrameSelector(features,'float64')),
    ('estimator', RandomForestClassifier() )       
], memory = memory) 

gs_rf = GridSearchCV(gs_pipe, n_jobs = -1 , \
    cv = GroupKFold(n_splits = 10).split(X, y, groups = df_dev.loc[:,'Dog']), \
        scoring = 'f1_weighted', param_grid = gs_params, return_train_score = True)

gs_rf.fit(X,y, groups = df_dev.loc[:,'Dog'])

# Saving Grid Search Results to pickle file 
joblib.dump(gs_results(gs_rf), '../models/{}.pkl'.format(gs_name), compress = 1 )









grid_RF6 = GridSearchCV(RF, n_jobs =-1 , \
    cv = cv, \
    scoring = 'f1_weighted' , param_grid = RF_params, return_train_score = True)
grid_RF6.fit(X,y, groups = df_dev.loc[:,'Dog'])
print("Best: %f using %s" % (grid_RF3.best_score_, grid_RF3.best_params_))

grid_RF4 = GridSearchCV(RF, n_jobs =-1 , \
    cv = GroupKFold(n_splits = 10).split(X, y, groups = df_dev.loc[:,'Dog']), \
        scoring = 'f1_micro' , param_grid = RF_params, return_train_score = True)
grid_RF4.fit(X,y, groups = df_dev.loc[:,'Dog'])
print("Best: %f using %s" % (grid_RF4.best_score_, grid_RF4.best_params_))

GB = Pipeline([
    ('selector', DataFrameSelector(feat,'float64')),
    ('reduce_dim', PCA()),  # SelectKBest(f_regression, k=50)],)
    ('estimator', GradientBoostingClassifier())
])

GB_params = {
    'estimator__max_depth' : [3, 10, 15, 20],
    'estimator__n_estimators': [10],
    'reduce_dim__n_components': [0.95, 80, 100], 
}


start = time()
GB_grid= GridSearchCV(GB, n_jobs = -1 , \
    cv = GroupKFold(n_splits = 10).split(X, y, groups = df_dev.loc[:,'Dog']), \
        scoring = 'f1_weighted' , param_grid = GB_params, return_train_score = True)
GB_grid.fit(X,y)
print("Best score {} using params %s" % (GB_grid.best_score_, GB_grid.best_params_))
print("Best estimator: {} " % (GB_grid.best_estimator_))

end = time()
print(end-start)

GB_params = GB_grid.best_params_
GB_score = GB_grid.best_score_
GB_estimator = GB_grid.best_estimator_
mean_scores = np.array(GB_grid.cv_results_['mean_test_score'])

## Gradient Boosting
print("Best features: {} ".format(GB_estimator.feature_importances_))
print("Train score: {} ".format(GB_estimator.train_score_))




''' caching transformers within a pipeline''' 

from joblib import Memory
from shutil import rmtree
from time import time

location = 'cachedir'
memory = Memory(location=location, verbose=10)

GB_cached = Pipeline([
    ('selector', DataFrameSelector(feat,'float64')),
    ('reduce_dim', PCA()),  # SelectKBest(f_regression, k=50)],)
    ('estimator', GradientBoostingClassifier())
], memory = memory)

start = time()

GB_grid_cached= GridSearchCV(GB_cached, n_jobs = -1 , \
    cv = GroupKFold(n_splits = 10).split(X, y, groups = df_dev.loc[:,'Dog']), \
        scoring = 'f1_weighted' , param_grid = GB_params, return_train_score = True)
GB_grid_cached.fit(X,y)

end = time()
print(end-start)


#########################


##### to be added to process

def ts_features(df_raw, df_dir, df_name, w_size, w_offset, t_time):
    '''
        Calculate more advanced features while filtering the most important ones 
    '''
    print('Processing simple features')

    # Finding transitions
    print(df_raw.columns)
    df_raw = process.transitions(df_raw)
    print(df_raw.columns)

    df_l2 = []
    # Iterating over the periods between transitions
    for (s_time, f_time) in zip( df_raw.loc[df_raw['Transition'] == True].index[:-1][0:2] + t_time , \
                                df_raw.loc[df_raw['Transition'].shift(-1) == True].index [0:2]- t_time):
        
        print (s_time,  f_time)
        df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).to_frame()
        # if there is not a transition in time -> transitions in time 
        if(~df_raw.ix[df_raw.index.get_loc(s_time-t_time)+1, 'Trans-Time']):
            print('\tCalculating Features\n ')
            df_l1 = []   
            feat = df_raw.columns[:-7].append(pd.Index(['Position']))
            print(df_raw.ix[s_time:f_time, 'Position'])
            df_rolled = roll_time_series(df_raw.ix[s_time:f_time, feat], column_id = 'Position', max_timeshift = w_size, rolling_direction  = -w_size, column_kind=None )
            df_l1.append(extract_features(df_rolled, column_id = 'Position').resample(w_offset).first())


            #df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).mean()).resample(w_offset).first())
            #df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).std()).resample(w_offset).first())
            #df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).median()).resample(w_offset).first())
            #df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).min()).resample(w_offset).first())
            #df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).max()).resample(w_offset).first())
            #df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).sem()).resample(w_offset).first())
            #print(df_l1)

            print (df_raw.loc[s_time, ['Dog','DC', 'Position']].values)

            df_l2.append( pd.concat(df_l1, axis = 1, # keys = ['mean','std', 'median', 'min', 'max'],\
            names = ['Statistics','BodyPart.SensorAxis'])\
            .assign(Dog = df_raw.loc[s_time,'Dog'], DC = df_raw.loc[s_time,'DC'], Type = df_raw.loc[s_time,'Type'], Position = df_raw.loc[s_time, 'Position']))  
           
        else:
            print('Do not calculate features\n\n' )

    df = pd.concat(df_l2)
    # Renaming the columns to contain stats.bodypart.sensor.axis, e.g. mean.Back.Acc.X, keeping last 4 columns (info and label) the same
    df.columns = df.columns[:-4].map('{0[0]}.{0[1]}'.format).append(df.columns[-4:].droplevel(1))
    print('Shape before before dropping NAs', df.shape)
    df = df.dropna()
    print('Shape before after dropping NAs', df.shape)

    print('Save df to csv')
    df.to_csv('%s\\%s.csv' % (df_dir, df_name))
    df_logger = log(df_name, log_file = '%s\\%s.log' % (df_dir, df_name))
    df_logger.info('\n\t Dataset created with tsfresh parameters: \n\ndf_name: {}, w_size: {}, w_offset: {}us, t_time: {}us'.format(df_name, w_size, w_offset, t_time))
    df_logger.info('\n\t Number of Examples in raw dataframe \n {} \n\n {} '.format(df['Position'].value_counts(), df['Type'].value_counts()))
    df_logger.info('\n\t Including data from  \n {}'.format( df.groupby(['Dog', 'DC']).size() ))
    logger.info('\t {}: Dataset created. See log for parameter details'.format(df_name))

    return (df)    


### to be added to process ###
def balance (df, label):
    ''' 
        WORK IN PROGRESS
        Balances df based on label and considering 

        This approach will need more logic, because some dogs only have very few examples of some positions 
    '''
    print('\nBalancing df for label', label , '\n')
    label = 'Position'
    df_list = []
    
    for pos in df[label].unique():
        print(pos)
        # number of examples in the category with the least amount of examples
        no_label = np.min(df[label].value_counts())
        dogs = dog_counts.size
        print(no_label)
        print(dogs)
        #
        no_label_dog =  int(no_label / dogs)
        print('cat_dog', no_label_dog)
        dog_counts = df.Dog[df[label] == pos].value_counts().sort_values(ascending = True)
        print(dog_counts)
        if (dog_counts[0] < no_label_dog):
            #df_list.append(df['Dog'].isin())
            print(no_label)
            # new number of examples per category for each dog is previous minus the ones already accounted for 
            no_label =- dog_counts.where(dog_counts<no_label_dog).sum()
        
        df_list = df[df[label] == pos].sample(no_label)


        print(pos)
        # number of dogs in each category
        # number of examples to sample from each dog
        df_list.append(df[df[label] == pos].sample(min_sample))
    df_balanced = pd.concat(df_list)
    return df_balanced

