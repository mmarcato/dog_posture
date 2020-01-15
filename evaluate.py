# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:21:08 2019

@author: marinara.marcato
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def pipe_perf (df, feat, label, pipes, cv):     
    '''
        Evaluate pipes and plot 
        params:
            df: dataframe containing features and label
            label: list of column name to be used as target
            pipes: dictionary of keys(pipeline name) and value(actual pipeline)
            cv: cross validation splitting strategy to be used
    '''        
    no = df[label].value_counts()
    X = df.loc[:, feat]
    y = df.loc[:, label].values
    
    print('Classifying', label , '\nbased on features', feat,\
         '\nusing pipelines', list(pipes.keys()), '\nCross Val Strategy', cv)
    
    perf = []             
    for name, pipe in pipes.items():
        #[[TN, FP], [FN, TP]] = confusion_matrix( y, (cross_val_predict(pipe, X, y, cv = cv)))

        # performance measurements: accuracy, precision, recall and f1 are calculated based on the confusion matrix
        # http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        # http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py                                   
        #accuracy = (TN+TP)/(TN+TP+FN+FP) 
        #precision = (TP)/(TP+FP) 
        #recall = (TP)/(TP+FN) 
        #f1 = 2*precision*recall/(precision + recall) 
        # Parameters above can be calculated with function, however, it is not that easy to access the data 
        # from it and it takes longer to calculate than the above
        
        #d = classification_report(y, cross_val_predict(pipe, X, y, cv= cv ), output_dict = True)

        score = cross_validate(pipe, X, y, cv= cv, return_train_score=True )
                                                
        perf.append([label, name, no, cv,
                            100*np.mean(score['test_score']), 100*np.mean(score['train_score']), 
                            np.mean(score['fit_time']), np.mean(score['score_time'])    ])
                              
        cols = ( 'Classifier', 'Pipeline', 'Examples', 'CV', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1(%)', 'test_score (%)',\
                     'train_score (%)', 'fit_time (s)', 'score_time (s)' )
        
    return(pd.DataFrame(perf, columns = cols, index = 'Name'))


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
