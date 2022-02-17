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
from time import time

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
import imports, process


# ------------------------------------------------------------------------- #
#                              Importing Raw Dataset                        #
# ------------------------------------------------------------------------- #


# ------------------------------------------------------------------------- #
#                           Feature Engineering                             #    
# ------------------------------------------------------------------------- # 
'''
parameters: window size, window offset and transition time 
    df_name:    dataset file name 
    t_time:     transition time - between positions used for creating a position window 
    w_size:     window size - for feature calculation, considering that raw data are recorded at 100Hz
    w_offset:   window offset - for resampling, taken from start_time + t_time + w_size/2 * as feature are calculated from centre of window
'''
dir_raw = os.path.join(dir_base, 'data', 'raw')

dir_new =  os.path.join(dir_base, 'data', 'simple')
df_name = 'df5_12'
w_size = 100
w_offset = timedelta(seconds = .50)
t_time = timedelta(seconds = .25)


# importing created raw dataset 
df_raw = imports.posture(dir_raw, 'df_raw5')

# print user defined settings used to create processed dataset
print('\nCreating a new dataset with simple features', 
        '\nDirectory:', dir_new, '\nName:', df_name,
        '\nWindow size:', w_size,'\nWindow Offset:', w_offset, '\nTransition Time:', t_time)


# creates and saves dataset with features with user defined settings 
start_time = time()
df_feat = process.features_simple(df_raw, dir_new, df_name, w_size, w_offset, t_time)
finish_time = time()

print("--- %s seconds ---" % (finish_time - start_time))


df_dev, df_test, df_gr = imports.split(df_feat, dir_new, df_name)

