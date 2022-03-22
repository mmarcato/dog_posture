import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

# Define local directories
dir_current = os.getcwd()
dir_parent = os.path.dirname(dir_current)
dir_base = os.path.dirname(dir_parent)
print(dir_base)

# import predictions dataset
df_pred =  pd.read_csv(os.path.join(dir_base, 'results', 'golden-predictions.csv'), 
                parse_dates = ['Timestamp'],
                index_col = 0,
                dayfirst = True,
                date_parser = lambda x: pd.to_datetime(x, format = '%d/%m/%Y %H:%M:%S'))
labels = np.sort(df_pred.Position.unique())
print(df_pred.columns)

# classification report 
df_exp1 = pd.DataFrame(classification_report(df_pred.Position, df_pred.Predicted_1, 
                            labels = labels, output_dict = True)).transpose()
df_exp2 = pd.DataFrame(classification_report(df_pred.Position, df_pred.Predicted_2, 
                            labels = labels, output_dict = True)).transpose()
df_exp3 = pd.DataFrame(classification_report(df_pred.Position, df_pred.Predicted_3, 
                            labels = labels, output_dict = True)).transpose()
print('Experiment 1\n', df_exp1)
print('Experiment 2\n', df_exp2)
print('Experiment 3\n', df_exp3)

experiments = {'Experiment 1': df_pred.Predicted_1,
                'Experiment 2': df_pred.Predicted_2,
                'Experiment 3': df_pred.Predicted_3}
print(experiments)


## CONFUSIONMATRIXDISPLAY FROM PREDICTIONS
for i, (key, predicted) in enumerate(experiments.items()):
    print(key)
    ConfusionMatrixDisplay.from_predictions(df_pred.Position, predicted, 
        normalize = 'true', values_format ='.2f', cmap='Blues', 
        display_labels=labels, xticks_rotation = 45)



## CONFUSIONMATRIXDISPLAY FROM PREDICTIONS
f, ax = plt.subplots(1, len(experiments), figsize=(20, 5), sharey='row')

for i, (key, predicted) in enumerate(experiments.items()):
    disp = ConfusionMatrixDisplay.from_predictions(df_pred.Position, predicted, 
    normalize = 'true', cmap='Blues', values_format='.2f',
    display_labels=labels, xticks_rotation = 45, colorbar = False, ax = ax[i])
    
    disp.ax_.set_title(key)
    disp.ax_.set_xlabel('')
    if i!=0:
        disp.ax_.set_ylabel('')

f.text(0.4, -0.1, 'Predicted label', ha='left')
f.colorbar(disp.im_, ax=ax)
plt.show()


## CONFUSIONMATRIXDISPLAY
f, axes = plt.subplots(1, len(experiments), figsize=(20, 5), sharey='row')

for i, (key, predicted) in enumerate(experiments.items()):
    cm = confusion_matrix(df_pred.Position, predicted)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=axes[i], xticks_rotation=45)
    disp.ax_.set_title(key)
    disp.im_.colorbar.remove()
    disp.ax_.set_xlabel('')
    if i!=0:
        disp.ax_.set_ylabel('')

f.text(0.4, -0.1, 'Predicted label', ha='left')
plt.subplots_adjust(wspace=0.20, hspace=0.1)
f.colorbar(disp.im_, ax=axes)
plt.show()


## CALCULATING ERROR PERCENTAGE PER DOG
df_pred.groupby(['Dog']).Error_1.mean()

