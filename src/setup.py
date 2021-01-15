import os
import pandas as pd
from datetime import timedelta
import logging
import os
from sklearn.feature_selection import SelectKBest, f_regression
    
''' this file contains parameters for the posture recognition algorithm
'''

def log(name, log_file = 'C:\\Users\\marinara.marcato\\Project\\Scripts\\dog_posture\\main.log', level = logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(name)s \n%(message)s')
    handler = logging.FileHandler(log_file)       
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger
    

# ------------------------------------------------------------------------- #
#                            Feature Engineering                            #    
# ------------------------------------------------------------------------- #
'''
parameters: window size, window offset and transition time 
df_name:    dataset file name 
t_time:     transition time - between positions used for creating a position window 
w_size:     window size - for feature calculation, considering that raw data are recorded at 100Hz
w_offset:   window offset - for resampling, taken from start_time + t_time + w_size/2 * as feature are calculated from centre of window
'''
df_dir = 'C:\\Users\\marinara.marcato\\Project\\Scripts\\dog_posture\\dfs'
df_name = 'df_32'
w_size = 25
w_offset = timedelta(seconds = .10)
t_time = timedelta(seconds = .25)

# ------------------------------------------------------------------------- #
#                             Machine Learning                              #    
# ------------------------------------------------------------------------- #

df_fname = 'df_32'
run = 'GS-GB-df_32'
