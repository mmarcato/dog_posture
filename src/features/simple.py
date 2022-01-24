import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

%load_ext autoreload
%autoreload 2

from __modules__ import imports
from __modules__ import process
from datetime import timedelta
import time

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
dir_raw = 'C:\\Users\\marinara.marcato\\Project\\Scripts\\dog_posture\\data\\raw'

dir_new = 'C:\\Users\\marinara.marcato\\Project\\Scripts\\dog_posture\\data\\processed'
df_name = 'df5_11'
w_size = 100
w_offset = timedelta(seconds = .25)
t_time = timedelta(seconds = .25)


# importing created raw dataset 
df_raw = imports.posture(dir_raw, 'df_raw5')

# print user defined settings used to create processed dataset
print('\nCreating a new dataset with simple features', 
        '\nDirectory:', dir_new, '\nName:', df_name,
        '\nWindow size:', w_size,'\nWindow Offset:', w_offset, '\nTransition Time:', t_time)

start_time = time.time()

# creates and saves dataset with features with user defined settings 
df_feat = process.features_simple(df_raw, dir_new, df_name, w_size, w_offset, t_time)

print("--- %s seconds ---" % (time.time() - start_time))