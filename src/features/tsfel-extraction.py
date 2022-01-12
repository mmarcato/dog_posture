''' Copyright (C) 2021 by Marinara Marcato
         <marinara.marcato@tyndall.ie>, Tyndall National Institute
        University College Cork, Cork, Ireland.
'''
# ------------------------------------------------------------------------- #
#                           Importing Global Modules                        #
# ------------------------------------------------------------------------- #
import os, sys
from datetime import timedelta
import numpy as np
import pandas as pd
# ------------------------------------------------------------------------- #
#                           Importing Local Modules                         #
# ------------------------------------------------------------------------- #
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# %load_ext autoreload
# %autoreload 2
from __modules__ import imports
from __modules__ import process


# ------------------------------------------------------------------------- #
#                              Importing Raw Dataset                        #
# ------------------------------------------------------------------------- #

# directory to retrieve raw file
dir_raw = 'C:\\Users\\marinara.marcato\\Project\\Scripts\\dog_posture\\data\\raw'
# importing created raw dataset 
df_raw = imports.posture(dir_raw, 'df_raw4')


# ------------------------------------------------------------------------- #
#             Decomposing Raw -> Body & Gravitational components            #
# ------------------------------------------------------------------------- #

gravity, body = process.gravity_body_components(df_raw.iloc[:, 0], freq = 100)
df = df_raw.iloc[:100, [0,1]]
df_new = pd.DataFrame(index = df_raw.index)
# funtion that returns a new dataframe with two columns (body and gravitational accelerations)
df_new = df_new.merge(
df.map(process.gravity_body_components)
                # applying on columns
                left_index=True, right_index=True)

df_new = df_new.merge(df.apply(lambda c: pd.Series({'feature1':c+1, 'feature2':c-1})), 
    left_index=True, right_index=True)
# ------------------------------------------------------------------------- #
#                              Creating New Dataset                         #
# ------------------------------------------------------------------------- #

# directory to save new file 
dir_new = 'C:\\Users\\marinara.marcato\\Project\\Scripts\\dog_posture\\data\\tsfel-12'

# window settings
w_size = 100
w_overlap = .75
t_time = timedelta(seconds = .25)

# Selecting specific dog and dc combination
# df_dogs = pd.DataFrame({
#                 'Dog' : ['Dugg', 'Fawn', 'Fawn', 'Joy'],
#                 'DC' : [1, 1, 2, 1]
#         })

# Selecting all Dog and DC combinations
df_dogs = df_raw.groupby(['Dog', 'DC']).count().reset_index()[['Dog', 'DC']]
print(df_dogs)

process.features_tsfel(df_raw, dir_new, df_dogs, w_size, w_overlap, t_time)

# ------------------------------------------------------------------------- #
#                                Other stuff                                #
# ------------------------------------------------------------------------- #

df_new = imports.features_tsfel(dir_new)

##### I'M STILL NOT SURE WHERE TO INCLUDE THIS

# splitting the entire dataset into dev and test sets
#df_dev, df_test = process.split(df_new, 0.2)
# splitting dev dataset into train and val
#df_train, df_val = process.split(df_dev, 0.25)

# saving dataframes
# df_test.to_csv('%s\\%s.csv' % (dir_new, 'data_test'))
# df_dev.to_csv('%s\\%s.csv' % (dir_new, 'data_dev'))
# df_train.to_csv('%s\\%s.csv' % (dir_new, 'data_dev'))
# df_val.to_csv('%s\\%s.csv' % (dir_new, 'data_val'))