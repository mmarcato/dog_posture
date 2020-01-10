# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:17:09 2019

@author: marinara.marcato
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA

# ------------------------------------------------------------------------- #
#                          Initializing parameters                          #    
# ------------------------------------------------------------------------- #
#dir_base = "Z:\\Tyndall\\IGDB\\Observational Study\\Data Collection\\Study\\Subjects"
dir_base = 'E:\\Subjects'
subjects = os.listdir(dir_base)[1:]
dcs = ['DC1', 'DC2']
bps = ['Back', 'Chest', 'Neck']

# ------------------------------------------------------------------------- #
#                            Defining functions                             #    
# ------------------------------------------------------------------------- #
def balance_df (df, label):
    df_list = []
    small_sample = np.min(df[label].value_counts())
    for pos in df[label].unique():
        df_list.append(df[df[label] == pos].sample(small_sample))
    df_balanced = pd.concat(df_list)
    return df_balanced

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

def Evaluate_Pipeline (pipelines, X, y, cross_val, title, label): 
    pipe_performance = []                                   
    parameters = ('Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1(%)', 'test_score (%)', 'train_score (%)', 'fit_time (s)', 'score_time (s)' )
    print (parameters)
    for i in range (len(pipelines)):
        [[TN, FP], [FN, TP]] = confusion_matrix( y[i], (cross_val_predict(pipelines[i], X[i], y[i], cv= cross_val)))

        # performance measurements: accuracy, precision, recall and f1 are calculated based on the confusion matrix
        # http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        # http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py                                   
        accuracy = (TN+TP)/(TN+TP+FN+FP) 
        precision = (TP)/(TP+FP) 
        recall = (TP)/(TP+FN) 
        f1 = 2*precision*recall/(precision + recall) 
        # I could have used the function below to calculate the same parameters
            #however, it is not that easy to plot the data from it and it takes longer to calculate than the above
                # I compared the results from both methods and they are the same
        # classification_report(y, cross_val_predict(pipelines[i], X[i], y[i], cv= kf_st ), digits = 4)
        
        score = cross_validate(pipelines[i], X[i], y[i], cv= cross_val, return_train_score=True )
                                                
        pipe_performance.append([100*accuracy, 100*precision, 100*recall, 100*f1, 
                                 100*np.mean(score['test_score']), 100*np.mean(score['train_score']), 
                                 np.mean(score['fit_time']), np.mean(score['score_time'])])
  
        print(label[i], '\t', ["%.5f" % elem for elem in pipe_performance[i]])

        
    # Plotting the graph for visual performance comparison 
    width = .9/len(pipelines)
    index = np.arange(9)
    colour = ['b', 'r', 'g', 'y', 'm', 'c', 'k']
    
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    for i in range (len(pipelines)):
        ax.bar(index[0:6] + width*i, pipe_performance[i][0:6], width, color = colour[i], label = label[i])  
        ax2.bar(index[6:8] + width*i, pipe_performance[i][6:8], width, color = colour[i], label = label[i])  

    ax.set_xticks(index + width*(len(pipe_performance)-1) / 2)
    ax.set_xticklabels(parameters, rotation=45)
    ax.legend()
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim([0,110])
    ax2.set_ylabel('Time (s)')
    plt.title(title)
    plt.figure(figsize=(10,20))
    plt.show()

    return (pipe_performance)  

# ------------------------------------------------------------------------- #
#                               Importing data                              #    
# ------------------------------------------------------------------------- #
f_name = 'E:\\Subjects\\_Timestamps.csv'
df_ts = pd.read_csv(f_name, skiprows = 2, usecols =['Episodes', 'Behaviours'])

#                   Importing position and episode info                     #    
df_ep, df_pos = {}, {}
for subj in subjects:
    df_ep[subj], df_pos[subj] = {},{}
    print(subj)
    for dc in dcs:
        df_ep[subj][dc], df_pos[subj][dc] = None, None
        f_name = '%s\\%s\\%s_Timestamps.csv' % (dir_base, subj, dc[-1])  
        if os.path.exists(f_name):
            # Read the information about the behaviour test 
            df_info = pd.read_csv(f_name, index_col = 0, nrows = 4, usecols = [0,1])
            date = df_info[subj]['Date']
            time = df_info[subj]['Start time']
            dt = pd.to_datetime(date + time, format = '%d/%m/%Y%H:%M:%S' )            
            print(dc, dt)
            # Read the episode Virtual Time (VT) 
            df_ep[subj][dc] = pd.read_csv(f_name, skiprows = 6, usecols = ['Episode', 'Ep-VT']).dropna()
            # Create new column for the episode Real Time (RT)
            df_ep[subj][dc].index = dt + pd.to_timedelta(df_ep[subj][dc]['Ep-VT'])         
            # Create new column for the episode Duration
            df_ep[subj][dc]['Duration'] = df_ep[subj][dc].index.to_series().diff().shift(-1)
            
            # Read the position Virtual Time (VT) 
            df_pos[subj][dc] = pd.read_csv(f_name, skiprows = 6, usecols = ['Position', 'Pos-VT']).dropna()
            # Create new column for the position Real Time (RT)
            df_pos[subj][dc].index = dt + pd.to_timedelta(df_pos[subj][dc]['Pos-VT'])         
            # Create new column for the position Duration
            df_pos[subj][dc]['Duration'] = df_pos[subj][dc].index.to_series().diff().shift(-1)        
    

#                           Importing Actigraph IMU                         #    
df_imu = {}
# Iterating through subjects
for subj in subjects:
    df_imu[subj] = {}       
    # Iterating through data collections
    for dc in dcs:
        df_list= []
        df_imu[subj][dc] = None
        # If this the path to data exists
        if os.path.isdir('%s\\%s\\%s_Actigraph' % (dir_base, subj, dc[-1])):
            print(subj, dc)
            # Looping through all bps
            for bp in bps:   
                # Find file path for each bp
                f_name =  glob.glob('%s\\%s\\%s_Actigraph\\*_%s.csv' % (dir_base, subj, dc[-1], bp))
                
                df_list.append(pd.read_csv(f_name[0], index_col = ['Timestamp'], parse_dates = [0], \
                      date_parser = lambda x: pd.to_datetime(x, format = '%Y-%m-%d %H:%M:%S.%f'))\
                        .drop(['Temperature'], axis = 1))
            # Concatenating dataframes for different body parts in one single dataframe
            # Results in one dataframe per dog per data collection
            df_imu[subj][dc] = pd.concat(df_list, axis = 1, keys = bps, \
              names = ['Body Parts', 'Sensor Axis'])
 

# ------------------------------------------------------------------------- #
#               Combining data to create features dataframe                 #    
# ------------------------------------------------------------------------- #           

#                       Populating 'Position' column                         #
        
# Set window size, window offset and transition time 
w_size = 100
w_offset = timedelta(seconds = .25)
t_time = timedelta(seconds = .25)
feat = ['mean','std', 'median', 'min','max']


df_l2 = []  
for subj in subjects:
    for dc in dcs:      
        # Checking if df_ep and therefore df_pos exist, as well as df_imu exists 
        if df_ep[subj][dc] is not None and df_imu[subj][dc] is not None:
            print(subj, dc, 'Position')
            for (s_time, f_time) in zip( df_pos[subj][dc].index.to_series() + t_time , \
                               df_pos[subj][dc].index.to_series().shift(-1) - t_time):
                
                df_l1 = []        
                
                df_l1.append((df_imu[subj][dc][s_time:f_time].rolling(w_size, center = True).mean()).resample(w_offset).first())
                df_l1.append((df_imu[subj][dc][s_time:f_time].rolling(w_size, center = True).std()).resample(w_offset).first())
                df_l1.append((df_imu[subj][dc][s_time:f_time].rolling(w_size, center = True).median()).resample(w_offset).first())
                df_l1.append((df_imu[subj][dc][s_time:f_time].rolling(w_size, center = True).min()).resample(w_offset).first())
                df_l1.append((df_imu[subj][dc][s_time:f_time].rolling(w_size, center = True).max()).resample(w_offset).first())
                #df_l1.append((df_imu[subj][dc][s_time:f_time].rolling(w_size, center = True).sem()).resample(w_offset).first())
                

                df_l2.append( pd.concat(df_l1, axis = 1, keys = feat, \
                              names = ['Statistics','BodyParts', 'Sensor Axis'])\
                               .assign(Dog = subj, DC = dc, Position = df_pos[subj][dc]['Position'][(s_time-t_time)]))  


df_feat = pd.DataFrame()              
df_feat = pd.concat(df_l2)

#                       Populating 'Type' column                         #
pos_type = {'Walking': 'Dynamic', 
            'W-Sniffing floor': 'Dynamic', 
            'Standing':'Static', 
            'Sitting':'Static', 
            'Lying down': 'Static',
            'Jumping up': 'Dynamic', 
            'Jumping down':'Dynamic', 
            'Body shake':'Dynamic', 
            'S-Sniffing floor': 'Static', 
            'Pull on leash': 'Dynamic', 
            'Moving': 'Dynamic'}

df_feat['Type'] = df_feat['Position'].map(pos_type)

# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
#                Machine Learning - Label 'Positions'                       #    
# ------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------- #
# Deleting rows with nan
df_feat.dropna(axis = 0, inplace = True)
df_feat = df_feat[df_feat['Position'] != 'Moving']
# Shuffling data
df_feat = df_feat.take(np.random.permutation(len(df_feat))) 


print(df_feat['Position'].value_counts())
df_feat = df_feat[df_feat['Position'] != 'S-Sniffing floor']
df_feat = df_feat[df_feat['Position'] != 'W-Sniffing floor']

df_all_bal = balance_df(df_feat, 'Position')
df_all_bal['Position'].value_counts()

x_all_bal = df_all_bal.loc[:, feat]
y_all_bal = df_all_bal.loc[:, 'Position'].values

kf = StratifiedKFold(n_splits = 10)
pipe_performance = [] 
LR = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('estimator', LogisticRegression() )       
        ])
RF = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('estimator', RandomForestClassifier() )       
        ]) 
SV = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('estimator', LinearSVC() )       
        ])
print('Classifying Positions' )
for pipe in [LR,RF, SV]:
    print(np.mean(cross_val_score(pipe, x_all_bal, y_all_bal, scoring="accuracy", cv=kf)))
    score = cross_validate(pipe, x_all_bal, y_all_bal, cv=kf, return_train_score=True)
    
    pipe_performance.append([100*np.mean(score['test_score']), 100*np.mean(score['train_score']), 
                                 np.mean(score['fit_time']), np.mean(score['score_time'])])
print(pipe_performance)    

# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
#                   Machine Learning - Label 'Type'                         #    
# ------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------- #

#                          Evaluating Dataset                               #
df_feat.loc[:, 'Type'].value_counts()

df_bal_type = balance_df(df_feat, 'Type')
df_bal_type['Type'].value_counts()

x_bal_type = df_bal_type.loc[:, feat]
y_bal_type = df_bal_type.loc[:, 'Type']



pipe_performance = []
print ('LR, RF, SV classifier for Type Positions')
for pipe in [LR, RF, SV]:
    print(np.mean(cross_val_score(pipe, x_bal_type , y_bal_type , scoring="accuracy", cv=kf)))
    score = cross_validate(pipe, x_bal_type, y_bal_type, cv=kf, return_train_score=True)
    
    pipe_performance.append([100*np.mean(score['test_score']), 100*np.mean(score['train_score']), 
                                 np.mean(score['fit_time']), np.mean(score['score_time'])])
print(pipe_performance)    

#                          Logistic Regression                              #
# We can see that the model is underfitting, so no point in using regularization or PCA
LR = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('estimator', LogisticRegression() )       
        ])
Ridge = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('estimator', RidgeClassifier() )       
        ]) 
    
    
# Evaluating pipelines
pipes = [LR, Ridge]
X_b = [x_bal_type] * 2
y_b = [y_bal_type] * 2
title = 'Difference between LogReg, Ridge'
label = ['LogReg', 'Ridge']
LR = Evaluate_Pipeline(pipes, X_b, y_b, kf, title, label)
  
#                     Linear Support Vector Classifier                      #
# Creating pipeline

SV = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('estimator', LinearSVC() )       
        ])
LS2 = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components = 120)),
        ('estimator', LinearSVC() )       
        ]) 
LS3 = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components = 100)),
        ('estimator', LinearSVC() )       
        ])
LS4 = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components = 50)),
        ('estimator', LinearSVC() )       
        ]) 
    
pipes = [SV]
title = 'Difference between no PCA, PCA (120), PCA(100), PCA (50) using Linear SCV'
label = ['no PCA', 'PCA - 120 comp',  'PCA - 100 comp', 'PCA - 50 comp']
LS = Evaluate_Pipeline(pipes, X_b, y_b, kf, title, label)         
    
#                         Random Forest Classifier                          #
RF = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('estimator', RandomForestClassifier() )       
        ]) 
RF2 = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components = 120)),
        ('estimator', RandomForestClassifier() )       
        ])    
RF3 = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components = 100)),
        ('estimator', RandomForestClassifier() )       
        ]) 
RF4 = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components = 50)),
        ('estimator', RandomForestClassifier() )       
        ]) 

pipes = [RF]
title = 'Random Forest - No PCA, PCA (120), PCA(100), PCA (50) '
label = ['No PCA', 'PCA - 120 comp',  'PCA - 100 comp', 'PCA - 50 comp']
RF_Results = Evaluate_Pipeline(pipes, X_b, y_b, kf, title, label)
      

   
# ------------------------------------------------------------------------- #
# Machine Learning - Tutorial https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60                          #    
# ------------------------------------------------------------------------- # 

#       Splitting DF into Test and Train before starting                   #

df_feat['Type'].value_counts()

X = df_feat.loc[:, feat].values
Y = df_feat.loc[:, 'Type'].values

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size= 0.2, stratify = Y, random_state=0)

# Build new data frame with Scaling and PCA on X_train
scaler = StandardScaler()

scaler.fit(X_train)      
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

pca_50 = PCA(n_components = 50)
pca_50.fit(X_train)                

X_train = pca_50.transform(X_train)     
X_test = pca_50.transform(X_test)  

LogReg = LogisticRegression()

LogReg.fit(X_train, Y_train)
LogReg.score(X_test, Y_test)

explained_variance = pca.explained_variance_ratio_

RF = RandomForestClassifier(max_depth = 5, random_state = 0)
RF.fit(X_train, Y_train)               
RF.score(X_test, Y_test)

# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
#                   Machine Learning - Label 'Position'                     #    
# ------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------- #

# ------------------------------------------------------------------------- #
#                     Classifier for 'Static' Positions                     #    
# ------------------------------------------------------------------------- # 
df_static = df_feat[ df_feat['Type'] == 'Static' ]
df_static['Position'].value_counts()
# Balancing dataframe
df_static_bal = balance_df(df_static, 'Position')
df_static_bal['Position'].value_counts()

X_static_bal = df_static_bal.loc[:, feat]
Y_static_bal = df_static_bal.loc[:, 'Position']

pipe_performance = []
print ('LR, RF, SV classifier for Static Positions')
for pipe in [LR, RF, SV]:
    print(np.mean(cross_val_score(pipe, X_static_bal, Y_static_bal, scoring="accuracy", cv=kf)))
    score = cross_validate(pipe, X_static_bal, Y_static_bal, cv=kf, return_train_score=True)
                                                
    pipe_performance.append([100*np.mean(score['test_score']), 100*np.mean(score['train_score']), 
                                 np.mean(score['fit_time']), np.mean(score['score_time'])])
 
    #print(confusion_matrix( Y_static_bal, (cross_val_predict(pipe, X_static_bal, Y_static_bal, cv= kf))))

print('test_score', 'train_score','fit_time', 'score_time')
print(pipe_performance)


    
# ------------------------------------------------------------------------- #
#                    Classifier for 'Dynamic' Positions                     #    
# ------------------------------------------------------------------------- # 
df_dynamic = df_feat[ df_feat['Type'] == 'Dynamic' ]
df_dynamic['Position'].value_counts()

df_dynamic_bal = balance_df(df_dynamic, 'Position')
df_dynamic_bal['Position'].value_counts()

X_dynamic_bal = df_dynamic_bal.loc[:, feat]
Y_dynamic_bal = df_dynamic_bal.loc[:, 'Position']


pipe_performance = []
print ('LR, RF, SV classifier for Dynamic Positions')
for pipe in [LR, RF, SV]:
    print(np.mean(cross_val_score(pipe, X_dynamic_bal, Y_dynamic_bal, scoring="accuracy", cv=kf)))
    score = cross_validate(pipe, X_dynamic_bal, Y_dynamic_bal, cv=kf, return_train_score=True)
    
    pipe_performance.append([100*np.mean(score['test_score']), 100*np.mean(score['train_score']), 
                                 np.mean(score['fit_time']), np.mean(score['score_time'])])
print(pipe_performance)    
    
