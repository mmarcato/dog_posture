''' Copyright (C) 2022 by Marinara Marcato
         <marinara.marcato@tyndall.ie>, Tyndall National Institute
        University College Cork, Cork, Ireland.
'''
# ------------------------------------------------------------------------- #
#                           Importing Global Modules                        #
# ------------------------------------------------------------------------- #
import os, sys
import numpy as np
import pandas as pd
from datetime import timedelta

# ------------------------------------------------------------------------- #
#                           Importing Local Modules                         #
# ------------------------------------------------------------------------- #

dir_current = os.path.dirname(os.path.realpath(__file__))
dir_parent = os.path.dirname(dir_current)
dir_base = os.path.dirname(dir_parent)
dir_modules = os.path.join(dir_base, 'src', '__modules__')
# Set path variable
sys.path.append(dir_modules)

# Local Modules
# %load_ext autoreload
# %autoreload 2
import imports


# ------------------------------------------------------------------------- #
#                           Import Dasets from Folder                       #
# ------------------------------------------------------------------------- #

dir_old = os.path.join(dir_base, 'data', 'tsfel')
dir_new = os.path.join(dir_base, 'data', 'final')

# import all 
df_old = imports.posture(dir_old, 'df-Tosh-1' )

# select rows with the 5 position type for learning
df_new = df_old[df_old.Position.isin(['standing', 'walking', 
                'sitting', 'lying down', 'body shake'])]

# create column for shake (anomaly detection)
df_new['Shake'] = np.where(df_new.Position == 'body shake', -1, +1)
print('Dataframe shape:', df_new.shape)
print('Observations per class:\n', df_new.Position.value_counts(),  df_new.Position.value_counts(normalize = True))
print('Body shakes', df_new[df_new.Shake == -1].shape)

df_new.to_csv(os.path.join(dir_new, 'df-golden.csv'))

# splitting the entire dataset into dev and test sets (and golden retriever - gr)
imports.split(df_new, dir_new, 'df-all')

# # splitting the entire dataset into dev and test sets
# df_dev, df_test = process.split(df_new, 0.2)
# # splitting dev dataset into train and val
# df_train, df_val = process.split(df_dev, 0.25)

# # saving dataframes
# df_test.to_csv('%s\\%s.csv' % (dir_df, 'data_test'))
# df_dev.to_csv('%s\\%s.csv' % (dir_df, 'data_dev'))
# df_train.to_csv('%s\\%s.csv' % (dir_df, 'data_dev'))
# df_val.to_csv('%s\\%s.csv' % (dir_df, 'data_val'))


# ------------------------------------------------------------------------- #
#                           Analyse TSFEL Datasets                          #
# ------------------------------------------------------------------------- #

df_all = imports.posture(dir_new, 'df-all')
df_dev = imports.posture(dir_new, 'df-all-dev')
df_test = imports.posture(dir_new, 'df-all-test')
df_gr = imports.posture(dir_new, 'df-all-gr')

labels = np.sort(df_test.Position.unique())
sample = pd.DataFrame(index = labels)

for df in [df_dev, df_test, df_gr, df_all]:
       sample = pd.concat([sample,df.Position.value_counts()], axis =1)
print(sample)

df_test.Dog.value_counts()
