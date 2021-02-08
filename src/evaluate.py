# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:21:08 2019

@author: marinara.marcato
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV

# Caching Modules
import joblib
from shutil import rmtree

class gs_results:
    # Storing Grid Search results
    def __init__(self, gs):
        self.cv_results_ = gs.cv_results_
        self.best_estimator_ = gs.best_estimator_
        self.best_params_ = gs.best_params_
        self.best_score_ = gs.best_score_

def gs_output(gs):
    '''
        Printing key metricts from the best estimator selected by GS algorithm
    '''
    best_idx_ = np.argmax(gs.cv_results_['mean_test_score'])
    print("Best Estimator \nTest mean: {:.6f}\t std: {:.6f}\nTrain mean: {:.6f} \t std:  {:.6f}\nparameters: {}".format( \
        np.max(gs.cv_results_['mean_test_score']), gs.cv_results_['std_test_score'][best_idx_],\
        gs.cv_results_['mean_train_score'][best_idx_],  gs.cv_results_['std_train_score'][best_idx_],\
        gs.best_params_))
        
def gs_dump(gs, gs_name, gs_dir, memory, location):    
# Saving Grid Search Results to pickle file 
    joblib.dump(gs, '{}/{}.pkl'.format(gs_dir, gs_name), compress = 1 )
    memory.clear()
    rmtree(location)

def gs_load(gs_name, gs_dir ):
    gs = joblib.load('{}/{}.pkl'.format(gs_dir, gs_name))
    gs_output(gs)
    return(gs)

def gs_perf (gs_pipe, gs_params, df):
    '''
        WORK IN PROGRESS - IM NOT SURE IF IT IS WORTH SEPARATING THE STEPS INTO DIFFERENT FILES
    '''
    gs_rf = GridSearchCV(gs_pipe, n_jobs = -1 , \
        cv = GroupKFold(n_splits = 10).split(X, y, groups = df.loc[:,'Dog']), \
            scoring = 'f1_weighted', param_grid = gs_params, return_train_score = True)
    gs_rf.fit(X,y, groups = df_dev.loc[:,'Dog'])
    return(gs_output(gs_rf))

def pipe_perf (df, feat, cv, label, pipes):     
    '''
        Evaluate pipes and plot 
        params:
            df: dataframe containing features and label
            label: list of column name to be used as target
            pipes: dictionary of keys(pipeline name) and value(actual pipeline)
            cv: cross validation splitting strategy to be used
    '''        
    df = df_dev
    X = df.loc[:, feat]
    y = df.loc[:, label].values

    print('Classifying', label , '\nbased on features', feat,\
         '\nusing pipelines', list(pipes.keys()), '\nCross Val Strategy', cv)
    
    perf = []       
    reports = {}
    scores = {}      
    for name, pipe in pipes.items():
        #[[TN, FP], [FN, TP]] = confusion_matrix( y, (cross_val_predict(pipe, X, y, cv = cv)))
        '''
        performance measurements: accuracy, precision, recall and f1 are calculated based on the confusion matrix
        http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py                                   
        accuracy = (TN+TP)/(TN+TP+FN+FP) 
        precision = (TP)/(TP+FP) 
        recall = (TP)/(TP+FN) 
        f1 = 2*precision*recall/(precision + recall) 
        Parameters above can be calculated with function, however, it is not that easy to access the data 
        from it and it takes longer to calculate than the above
         '''
        report = classification_report(y, cross_val_predict(pipe, X, y, cv=GroupKFold(n_splits = 10), groups = df_dev.loc[:,'Dog'], n_jobs = -1), output_dict = True)

        reports[name] = report

        print(report)

        score = cross_validate(pipe, X, y, cv= GroupKFold(n_splits = 10), scoring = 'f1_score', groups = df_dev.loc[:,'Dog'], n_jobs = -1, return_train_score=True )

        print(score)
                                                
        perf.append([label, name, df.shape, str(cv)] + \
                        list(np.mean(list(score.values()), axis = 1)) + \
                            list(report['weighted avg'].values()))                       
        cols = ( ['Classifier', 'Pipeline', 'Examples', 'CV'] + list(score.keys()) + list(report['weighted avg'].keys()) )
 
        scores[name] = score

        print(perf)


    return(pd.DataFrame(perf, columns = cols).set_index('Pipeline'), reports, scores)
    
def plot_perf (pipes, perf):
    # Plotting the graph for visual performance comparison 
    width = .9/len(pipes)
    index = np.arange(9)
    colour = ['b', 'r', 'g', 'y', 'm', 'c', 'k']
    
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    for i in range (len(pipes)):
        ax.bar(index[0:6] + width*i, perf[i][0:6], width, color = colour[i], label = label[i])  
        ax2.bar(index[6:8] + width*i, perf[i][6:8], width, color = colour[i], label = label[i])  

    ax.set_xticks(index + width*(len(perf)-1) / 2)
    ax.set_xticklabels(parameters, rotation=45)
    ax.legend()
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim([0,110])
    ax2.set_ylabel('Time (s)')
    plt.title(title)
    plt.figure(figsize=(10,20))
    plt.show()

    return (perf)  
