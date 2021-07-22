import os, sys
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


%load_ext autoreload
%autoreload 2
from __modules__ import imports
from __modules__ import process
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

# directory where the raw file is at
dir_raw = 'C:\\Users\\marinara.marcato\\Project\\Scripts\\dog_posture\\data\\raw'
# importing created raw dataset 
df_raw = imports.posture(dir_raw, 'df_raw3')


dir_new = 'C:\\Users\\marinara.marcato\\Project\\Scripts\\dog_posture\\data\\processed'
df_name = 'df3_11-intermediate'
w_size = 100
w_offset = timedelta(seconds = .25)
t_time = timedelta(seconds = .25)


print('Processing simple features \n df_name {}, w_size {}, w_offset {}, t_time{}'.format(df_name, w_size, w_offset, t_time))


# Finding transitions in posture
df = process.transitions(df_raw)

df = df[df['Dog'] == 'Diva']
cols = df.columns[:-7]
timestamp = df.index
df_new = pd.DataFrame([])

# Iterating over the postures, taking the steady periods between transitions
for (s_time, f_time) in zip( df.loc[df['Transition'] == True].index[:-1] + t_time , \
                            df.loc[df['Transition'].shift(-1) == True].index - t_time):

    print(s_time,  f_time)
    print(df.loc[s_time-t_time, 'Position'])

    idx = timestamp.get_loc(s_time-t_time)

    # if there is not a transition in time 
    if(~df.loc[timestamp[idx + 1], 'Trans-Time']):
    
        print('Calculating Features for {}\n'
                .format(df.loc[s_time, ['Dog','DC','Position']].values))    
        
        # calculating all features and appending to the dataframe
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

# flatten the multiindex column to a simple columns +   add the last 4 columns
df_new.columns =  [".".join(v) for v in df_new.columns[:-4]] + ['Dog', 'DC', 'Type', 'Position']
# saving new features dataframe
print('Save df to csv')
df_new.to_csv('%s\\%s.csv' % (df_dir, df_name))



####################### NEW FUNCTIONS

z = np.random.uniform(low=-1, high=1, size=(10,))



def n_zero_crossing(x):
        # count number of zero crossings
        return sum(x * np.append(x[1:], [np.nan]) < 0 )

def q1(x):
        # calculate the 1st quartile 
        return x.quantile(0.25)

def q3(x):
        # calculate the 3st quartile
        return x.quantile(0.75)

def iqr(x):
        #calculate interquartile rage (Q3 - Q1)
        return np.subtract(*np.percentile(x, [75, 25]))

def mad(x):
        # median absolute deviation
        np.median(np.abs(x - np.median(x)))

def md(x):
        # mean absolute deviation
        np.mean(np.abs(x - np.median(x)))

        energy(): Energy measure. Sum of the squares divided by the number of values.

def rms(x):
        # calculate root mean square 
        return np.sqrt(np.mean(x**2))


def sma(x, y, z):
        # calculates signal magnitude area
        # t is the time window
        return np.sum(np.abs(x) + np.abs(y) + np.abs(z))/len(x)

def svm(x, y, z):
        # calculates signal vector magnitude
        return np.sum(np.sqrt(x**2 + y**2 + z**2))/len(x)

#    df_logger = log(df_name, log_file = '%s\\%s.log' % (df_dir, df_name))
#    df_logger.info('\n\t Dataset created with simple_feature parameters: \n\ndf_name: {}, w_size: {}, w_offset: {}us, t_time: {}us'.format(df_name, w_size, w_offset, t_time))
 #   df_logger.info('\n\t Number of Examples in raw dataframe \n{} \n\n{}\n'.format(df_new['Position'].value_counts(), df_new['Type'].value_counts()))
  #  df_logger.info('\n\t Including data from  \n{}\n\n'.format( df_new.groupby(['Dog', 'DC']).size() ))
#    logger.info('{}: Dataset created. See log for parameter details'.format(df_name))

#    return (df_new)