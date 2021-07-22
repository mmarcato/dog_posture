import os, sys
from datetime import timedelta
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
import tsfel

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


%load_ext autoreload
%autoreload 2
from __modules__ import imports
from __modules__ import process

# directory where the raw file is at
dir_raw = 'C:\\Users\\marinara.marcato\\Project\\Scripts\\dog_posture\\data\\raw'
# importing created raw dataset 
df_raw = imports.posture(dir_raw, 'df_raw3')


dir_new = 'C:\\Users\\marinara.marcato\\Project\\Scripts\\dog_posture\\data\\processed'
df_name = 'df3_11-tsfel'
w_size = 100
w_offset = timedelta(seconds = .25)
t_time = timedelta(seconds = .25)
print('Processing simple features \n df_name {}, w_size {}, w_offset {}, t_time{}'.format(df_name, w_size, w_offset, t_time))


# Finding transitions in posture
df = process.transitions(df_raw)

df = df[df['Dog'] == 'Diva']
cols = df.columns[:-7]
cols = cols[cols.str.contains('Neck')]
cols = cols[~cols.str.contains('Mag')]
timestamp = df.index
df_feat = pd.DataFrame([])


# Retrieves a pre-defined feature configuration file to extract all available features
cfg = tsfel.get_features_by_domain('statistical')
# Extract features
# X = tsfel.time_series_features_extractor(cfg, df)


# Iterating over the postures, taking the steady periods between transitions
for (s_time, f_time) in zip(df.loc[df['Transition'] == True].index[:-1] + t_time , \
                            df.loc[df['Transition'].shift(-1) == True].index - t_time):
    print(s_time,  f_time)

    idx = timestamp.get_loc(s_time-t_time)

    # if there is not a transition in time 
    if(~df.loc[timestamp[idx + 1], 'Trans-Time']):
    
        print('Calculating Features for {}\n'
                .format(df.loc[s_time, ['Dog','DC','Position']].values))
        print(df.loc[s_time:f_time, 'Neck.Acc.X']) 
        
        # calculating all features and appending to the dataframe
        ############# DROP MAGNETOMETER FEATURES?
        
        ############# PREPROCESSING: ACCELEROMETER FILTERING -> BODY + GRAVITATIONAL COMPONENTS 

        ############# FEATURE EXTRACTION USING TSFEL
        df_feat = df_feat.append(
                    tsfel.time_series_features_extractor(
                        # configuration file with features to be extracted 
                        dict_features = cfg,                            
                        # dataframe window to calculate features window on 
                        signal_windows = df.loc[s_time:f_time, 'Neck.Acc.X'],   
                        # sampling frequency of original signal
                        fs = 100,
                        # sliding window size
                        window_size = w_size, 
                        # overlap between subsequent sliding windows
                        overlap = .5,
                        # using all processors
                        n_jobs = -1 
                        )
                        .assign(Dog = df.loc[s_time,'Dog'], 
                        DC = df.loc[s_time,'DC'], 
                        Type = df.loc[s_time,'Type'], 
                        Position = df.loc[s_time, 'Position']))

# flatten the multiindex column to a simple columns +   add the last 4 columns
df_feat.columns =  [".".join(v) for v in df_feat.columns[:-4]] + ['Dog', 'DC', 'Type', 'Position']

'''
        df_new = df_new.append(df.loc[s_time:f_time, cols]
                # rolling window with size w_size
                .rolling(w_size, center = True)
                # calculating features
                .agg(['min', 'max', 'mean', 'std', 'var',
                        q1, 'median', q3, iqr, mad,
                        'sum', 'skew', 'kurt', 
                        # signal magnitude area
                        # activity (accellerometers)
                        #  
                        ])
                # dropping the first and last values that are NAs
                .dropna(axis = 'index', how = 'any')
                # resample to apply the window offset
                .resample(w_offset).first()
                # creating columns for information and labels
                .assign(Dog = df.loc[s_time,'Dog'], 
                        DC = df.loc[s_time,'DC'], 
                        Type = df.loc[s_time,'Type'], 
                        Position = df.loc[s_time, 'Position']))
'''