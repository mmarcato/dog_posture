# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:21:08 2019

@author: marinara.marcato
"""
def balance_df (df, label):
    df_list = []
    small_sample = np.min(df[label].value_counts())
    for pos in df[label].unique():
        df_list.append(df[df[label] == pos].sample(small_sample))
    df_balanced = pd.concat(df_list)
    return df_balanced


def evaluate_pipeline (pipelines, X, y, cross_val, title, label): 
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

def Validate_Pipeline (pipelines, X, y, cross_val, title, label): 
    pipe_performance = []
    y_predicted = []                                              
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


def Eval_Pipeline_3 (pipelines, X, y, cross_val, title, label): 
    pipe_performance = []
    y_predicted = []                                              
    parameters = ('Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1(%)', 'test_score (%)', 'train_score (%)', 'fit_time (s)', 'score_time (s)' )
    print (parameters)
    for i in range (len(pipelines)):
        [[TN, FP, FP], [FN, TP]] = confusion_matrix( y[i], (cross_val_predict(pipelines[i], X[i], y[i], cv= cross_val)))

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