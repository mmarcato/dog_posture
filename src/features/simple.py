# from src.__modules__ import setup
from src.__modules__ import imports
from src.__modules__ import process
from datetime import timedelta

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
df_dir = 'C:\\Users\\marinara.marcato\\Project\\Scripts\\dog_posture\\dfs'
df_name = 'df_11'
w_size = 100
w_offset = timedelta(seconds = .25)
t_time = timedelta(seconds = .25)


# importing created raw dataset 
df_raw = imports.posture(df_dir, 'df_raw')

# print user defined settings used to create processed dataset
print(df_name, w_size, w_offset, t_time)

# creating dataset with features with user defined settings 
df_feat = process.features(df_raw, df_dir, df_name, w_size, w_offset, t_time)
