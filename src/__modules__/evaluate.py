"""

"""
# ------------------------------------------------------------------------- #
#                                  Imports                                  #    
# ------------------------------------------------------------------------- # 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from sklearn.decomposition import PCA

from sklearn.model_selection import ShuffleSplit, GroupKFold
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay, f1_score

from sklearn.model_selection import GridSearchCV

# Caching Modules
import joblib
from shutil import rmtree

# ------------------------------------------------------------------------- #
#                                   Classes                                 #    
# ------------------------------------------------------------------------- # 

class gs_results:
    # Storing Grid Search results
    def __init__(self, gs):
        self.cv_results_ = gs.cv_results_
        self.best_estimator_ = gs.best_estimator_
        self.best_params_ = gs.best_params_
        self.best_score_ = gs.best_score_


# ------------------------------------------------------------------------- #
#                                  Functions                                #    
# ------------------------------------------------------------------------- # 
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
        gs.best_params_))
        
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

def gs_perf (gs_pipe, gs_params, X, y, groups, cv):# dir_model, run):
    '''
        WORK IN PROGRESS - IM NOT SURE IF IT IS WORTH SEPARATING THE STEPS INTO DIFFERENT FUNCTIONS
    '''
    # location = 'cachedir'
    # memory = joblib.Memory(location=location , verbose=10)


    start_time = time.time()
    gs = GridSearchCV(gs_pipe, param_grid = gs_params, 
            scoring = 'f1_weighted', \
            n_jobs = -1, cv = cv, return_train_score = True)
    gs.fit(X,y, groups = groups)
    end_time = time.time()
    duration = end_time - start_time
    print("--- %s seconds ---" % (duration))
        
    # joblib.dump(gs, '{}/{}.pkl'.format(dir_model, run), compress = 1 )
    # memory.clear(warn=False)
    # rmtree(location)

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

def metrics(y_final):
    
    # defining variables for performance evaluation
    y_true = y_final['Position']
    y_pred = y_final['Predicted']
    labels = np.sort(y_true.unique())
    
    # Calculate metrics from confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred, labels  = labels)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN)

    # Create dataframe with metrics
    df_metrics = pd.DataFrame({
        'label':  labels,
        'TPR': TPR, 'TNR': TNR, 'ACC': ACC, 'PPV': PPV,
        'f1_score': f1_score(y_true, y_pred, labels = labels, average = None)
    })

    print(df_metrics)
    print('\nf1 scores')
    print('macro: {:0.4f}'.format(f1_score(y_true, y_pred, average = 'macro')))
    print('micro: {:0.4f}'.format(f1_score(y_true, y_pred, average = 'micro')))
    print('weighted: {:0.4f}'.format(f1_score(y_true, y_pred, average = 'weighted')))

    # Confusion matrix - normalised by rows (true)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
        normalize = 'true', values_format ='.2f', cmap='Blues', 
        display_labels=labels, xticks_rotation = 45)
    return(df_metrics)

def model(X_true, y_true, model):    
    y_pred = model.predict(X_true)
    y_pred = pd.Series(y_pred, index = X_true.index)


    labels = np.sort(y_true.unique())
    ## Calculate f1-scores
    df_f1_class = pd.DataFrame({
        'label':  labels,
        'f1_score': f1_score(y_true, y_pred, labels = labels, average = None)
    })

    print(df_f1_class)
    print('\nf1 scores')
    print('macro: {:0.4f}'.format(f1_score(y_true, y_pred, average = 'macro')))
    print('micro: {:0.4f}'.format(f1_score(y_true, y_pred, average = 'micro')))
    print('weighted: {:0.4f}'.format(f1_score(y_true, y_pred, average = 'weighted')))


def exp1(df_test, gs_direct):
    
    # print models dev set performance
    print('Development set performance')
    gs_output(gs_direct)
    
    # select test data for prediction
    X_true = df_test.iloc[:,1:-6]
    y_true = df_test.Position

    # DIRECT
    y_pred = gs_direct.best_estimator_.predict(X_true)
    y_pred = pd.Series(y_pred, index = X_true.index)

    # merge True and Predicted ys 
    # ensure rows are aligned (same timestamp)
    y_final = df_test[['Timestamp', 'Dog', 'DC', 'Position']].merge(
            y_pred.rename('Predicted'), 
            left_index = True, right_index = True, how = 'left')

    ## cheking error by date and dog/dc
    y_final['Error'] = np.where(y_final.Position == y_final.Predicted, 0, 1)
    print(y_final.groupby([y_final.Timestamp.dt.date])['Error'].mean())
    print(y_final.groupby([y_final['Dog'], y_final['DC']])['Error'].count())
    
    return(y_final)

def exp2(df_test, gs_type, gs_static, gs_dynamic):

    # print models dev set performance
    print('Development set performance')
    gs_output(gs_type)
    gs_output(gs_static)
    gs_output(gs_dynamic)

    # select test data for prediction
    X_true = df_test.iloc[:,1:-6]

    # TYPE MODEL
    y_type = gs_type.best_estimator_.predict(X_true)
    y_type = pd.Series(y_type, index = X_true.index)

    # STATIC MODEL
    X_static = X_true[y_type == 'static']
    y_static = gs_static.best_estimator_.predict(X_static)
    y_static = pd.Series(y_static, index = X_static.index)

    # DYNAMIC
    X_dynamic = X_true[y_type == 'dynamic']
    y_dynamic = gs_dynamic.best_estimator_.predict(X_dynamic)
    if isinstance(y_dynamic[0], np.int32):
        y_dynamic = np.where(y_dynamic == -1, "body shake", "walking")
    y_dynamic = pd.Series(y_dynamic, index = X_dynamic.index)

    # combine static and dynamic predictions
    y_pred = pd.concat([y_static, y_dynamic])

    # merge True and Predicted ys 
    # ensure rows are aligned (same timestamp)
    y_final = df_test[['Timestamp','Dog', 'DC', 'Position']].merge(
            y_pred.rename('Predicted'), 
            left_index = True, right_index = True, how = 'left')
        
    ## cheking error by date and dog/dc
    y_final['Error'] = np.where(y_final.Position == y_final.Predicted, 0, 1)
    y_final.groupby([y_final.Timestamp.dt.date])['Error'].mean()
    y_final.groupby([y_final['Dog'], y_final['DC']])['Error'].count()
    
    return(y_final)

def exp3(df_test, gs_anomaly, gs_normal):
    # print models dev set performance
    print('Development set performance')
    gs_output(gs_anomaly)
    gs_output(gs_normal)

    # select test data for prediction
    X_true = df_test.iloc[:,1:-6]
    y_true = df_test.Shake

    # Use best estimator to predict on test set
    y_anomaly = gs_anomaly.best_estimator_.predict(X_true)
    y_anomaly = pd.Series(y_anomaly, index = X_true.index)

    # select rows with abnormal behaviour and replace -1 with body shake
    y_shake = y_anomaly[y_anomaly == -1].replace(-1, 'body shake')

    # select rows with normal behaviour
    X_pos = df_test.iloc[y_anomaly[y_anomaly == 1].index, 1:-6]
    y_pos = gs_normal.best_estimator_.predict(X_pos)
    y_pos = pd.Series(y_pos, index = X_pos.index)
    print("Position dataframe shape:", y_pos.shape)
    print("Position dataframe labels:", y_pos.unique())

    y_pred = pd.concat([y_shake, y_pos])
    print("Predicted dataframe shape:" , y_pred.shape)

    # merge True and Predicted ys - ensure rows are aligned (same timestamp)
    y_final = df_test[['Timestamp','Dog', 'DC', 'Position']].merge(
            y_pred.rename('Predicted'), 
            left_index = True, right_index = True, how = 'left')
    
        
    ## cheking error by date and dog/dc
    y_final['Error'] = np.where(y_final.Position == y_final.Predicted, 0, 1)
    y_final.groupby([y_final.Timestamp.dt.date])['Error'].mean()
    y_final.groupby([y_final['Dog'], y_final['DC']])['Error'].count()
    
    return(y_final)