"""

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from sklearn.model_selection import ShuffleSplit, GroupKFold
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report

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
    print("Best Estimator \nTest mean: {:.4f}\t std: {:.4f}\nTrain mean: {:.4f} \t std:  {:.4f}\nparameters: {}".format(
        np.max(gs.cv_results_['mean_test_score']), 
        gs.cv_results_['std_test_score'][best_idx_],
        gs.cv_results_['mean_train_score'][best_idx_],  
        gs.cv_results_['std_train_score'][best_idx_],
        gs.best_params_)) #
        
def gs_dump(gs, gs_name, gs_dir, memory, location):    
    ''' 
    Saving Grid Search Results to pickle file 
    '''
    joblib.dump(gs, '{}/{}.pkl'.format(gs_dir, gs_name), compress = 1 )
    memory.clear()
    rmtree(location)

def gs_load(gs_name, gs_dir ):
    gs = joblib.load('{}/{}.pkl'.format(gs_dir, gs_name))
    gs_output(gs)
    return(gs)

def gs_perf (gs_pipe, gs_params, X, y, groups, cv):
    '''
        WORK IN PROGRESS - IM NOT SURE IF IT IS WORTH SEPARATING THE STEPS INTO DIFFERENT FUNCTIONS
    '''
    gs = GridSearchCV(gs_pipe, param_grid = gs_params, 
            scoring = 'f1_weighted', \
            n_jobs = -1, cv = cv, return_train_score = True)
    gs.fit(X,y, groups = groups)

    return(gs_results(gs))

def pipe_perf (df, feat, label, pipes):     
    '''
        Evaluate pipes and plots
        params:
            df: dataframe containing features and label
            label: list of column name to be used as target
            pipes: dictionary of keys(pipeline name) and value(actual pipeline)

        return: list with three elements
            dataframe: with performance of the pipeline
            reports: precision, recall, f1-score, support for each of the classes

    '''
    X = df.loc[:, feat]
    y = df.loc[:, label].values

    print('Classifying', label , '\nbased on features', feat,\
         '\nusing pipelines', list(pipes.keys()))
    
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
        # CV (fit + predict) to set get classification report
        report = classification_report(
                        y, 
                        cross_val_predict(pipe, X, y, cv=GroupKFold(n_splits = 10), 
                        groups = df.loc[:,'Dog']),                     
                        output_dict = True
                )
                    
        # add new dictionary entry where key = name and value = report
        reports[name] = report
        print(report)

        # fit and predict to calculate calculate f1_weighted score with CV 10 GroupKFold
        score = cross_validate(
                    pipe, X, y, 
                    scoring = 'f1_weighted', 
                    cv= GroupKFold(n_splits = 10), 
                    groups = df.loc[:,'Dog'],
                    return_train_score=True,
                    return_estimator = True
                )

        # add new dictionary entry where key = name and value = score
        scores[name] = score
        print(score)

        # appending all data to a list                                        
        perf.append(
                        [label, name, df.shape] + 
                        # fit_time, score_time, test_score, train_score
                        list(np.mean(list(score.values()), axis = 1)) +   
                        # precision, recall, f1-score, support average among classes
                        list(report['weighted avg'].values())
                    ) 
        # creating a list with columns names     
        cols = ( 
                    ['Classifier', 'Pipeline', 'Examples'] + 
                    list(score.keys()) +
                    list(report['weighted avg'].keys()) 
                )

        print(perf)
    
    # returns a list with three elements
    return(pd.DataFrame(perf, columns = cols).set_index('Pipeline'), reports, scores)
       
def plot_cv_results (pipes, perf, parameters, title):
    # Plotting the graph for visual performance comparison 
    width = .9
    index = np.arange(9)
    colour = ['b', 'r', 'g', 'y', 'm', 'c', 'k']
    
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    for i in range (len(parameters)):
        ax.bar(index[0:6] + width, perf[i][0:6], width, color = colour[i], label = label[i])  
        ax2.bar(index[6:8] + width, perf[i][6:8], width, color = colour[i], label = label[i])  

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


def plot_perf_old (pipes, perf, parameters, title):
    # Plotting the graph for visual performance comparison 
    width = .9 /(len(pipes) - 1) 
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
