# ------------------------------------------------------------------------- #
#                                   Imports                                 #    
# ------------------------------------------------------------------------- # 

import pandas as pd
import numpy as np
from datetime import timedelta
from scipy import signal
import tsfel 

from setup import log


# ------------------------------------------------------------------------- #
#                                  Functions                                #    
# ------------------------------------------------------------------------- # 

def transitions(df): 
    """
    Processes df_raw to create new columns:
        1. 'Trans-Pos' - Transition in Position for consecutive positions performed by same dog
        2. 'Trans-Time' - Transition in Time in case moving in between two positions or different dog
        3. 'Transition' -  Transition column combining both step 1. and 2.
    Parameters
    ----------
    """
    # Finding transitions in posture
    df['Trans-Pos'] = df['Position'].shift()  != df['Position']
    # Finding transitions in time that are bigger than the 100Hz -> 10,000 microseconds
    df['Trans-Time'] = (df.index.to_series().diff() != timedelta(microseconds = 10000)) # + \
                            #(df.index.to_series().diff().shift(-1) != timedelta(microseconds = 10000))
    # Combining the time and position transitions
    df['Transition'] = df['Trans-Pos'] + df['Trans-Time']
    # Changing last row into a transition, Transition column has s_time and f_time of the BT
    df.iloc[-1]['Transition'] = True
    
    return(df)

def features_simple(df_raw, df_dir, df_name, w_size, w_offset, t_time):
    """
    Extracts 'min', 'max', 'mean','std', 'median', 'sum', 'skew', 'kurt' from a window interval in df_raw 
    based on timestamps for positions in df_raw
    Saves transformed df to a file in df_dir with df_name:
        df_raw: dataframe with all raw IMU measurements and info 'Dog', 'DC' & 'Position', 'Type'
        df_imu: dataframe containing Actigraph data (back, chest, neck)*(3-axis)*(acc, gyr, mag)
    Parameters
    ----------
    pr_feat contains columns for
    df_name = dataset name
    w_size = size of the window df_feat.ix[df_feat.index.get_loc(s_time)+1, 'Trans-Time']
    w_offset = offset from start time for the value to be taken
    t_time = transition time between positions
    Returns
    ----------
    df containing features calculated and label 'Position' and 'Type'
    """     
    
    logger = log(__name__)

    print('Processing simple features \n df_name {}, w_size {}, w_offset {}, t_time{}'.format(df_name, w_size, w_offset, t_time))

    # Finding transitions in posture
    df = transitions(df_raw)
    cols = df.columns[:-7]
    timestamp = df.index
    df_new = pd.DataFrame([])

    # Iterating over the periods between transitions
    for (s_time, f_time) in zip( df.loc[df['Transition'] == True].index[:-1] + t_time , \
                                df.loc[df['Transition'].shift(-1) == True].index - t_time):

        #print(s_time,  f_time)
        #print(df.loc[s_time-t_time, 'Position'])

        idx = timestamp.get_loc(s_time-t_time)

        # if there is not a transition in time 
        if(~df.loc[timestamp[idx + 1], 'Trans-Time']):
        
            #print('Calculating Features for {}\n'.format(df.loc[s_time, ['Dog','DC','Position']].values))    
            
            # calculating all features and appending to the dataframe
            df_new = df_new.append(df.loc[s_time:f_time, cols]
                    # rolling window with size w_size
                    .rolling(w_size, center = True)
                    # calculating features
                    .aggregate(['min', 'max', 'mean', 'std', 'median', 'sum', 'skew', 'kurt'])
                    # dropping the first and last values that are NAs
                    .dropna(axis = 'index', how = 'any')
                    # resample to apply the window offset
                    .resample(w_offset).first()
                    # creating columns for information and labels
                    .assign(Dog = df.loc[s_time,'Dog'], 
                            DC = df.loc[s_time,'DC'], 
                            Breed = df.loc[s_time,'Breed'], 
                            Type = df.loc[s_time,'Type'], 
                            Position = df.loc[s_time, 'Position']))

    # flatten the multiindex column to a simple columns
    df_new.columns =  [".".join(v) for v in df_new.columns[:-5]] + \
                        ['Dog', 'DC', 'Breed', 'Type', 'Position'] # adding the last 5 columns

    print('Save df to csv')
    df_new.to_csv('%s\\%s.csv' % (df_dir, df_name))

    df_logger = log(df_name, log_file = '%s\\%s.log' % (df_dir, df_name))
    df_logger.info('\n\t Dataset created with simple_feature parameters: \
                \n\ndf_name: {}, w_size: {}, w_offset: {}us, t_time: {}us'.format(df_name, w_size, w_offset, t_time))
                
    df_logger.info('\n\t Number of Examples in raw dataframe \n{} \n\n{}\n'.format(
                df_new['Position'].value_counts(), df_new['Type'].value_counts()))

    df_logger.info('\n\t Including data from  \n{}\n\n'.format( df_new.groupby(['Dog', 'DC']).size() ))
    logger.info('{}: Dataset created. See log for parameter details'.format(df_name))

    return (df_new)

def features_tsfel(df_raw, df_dir, df_dogs, w_size, w_overlap, t_time):
    """
    Description

    Parameters
    ----------
    variable (type)
            Description
    
    Returns
    ----------
    variable (type)
            Description

    """
    print('**Started processing TSFEL features**\n \
        w_size {}, w_overlap {}, t_time {}'.format(w_size, w_overlap, t_time))
            
    # Finding transitions in posture, adds 3 more columns to original dataframe
    df_raw = transitions(df_raw)
    # Selecting sensor columns
    cols = df_raw.columns[:-8]
    # dropping Magnetometer Features
    cols = cols[~cols.str.contains('Mag')]
    
    # feature extraction settings
    cfg = tsfel.get_features_by_domain()

    for (dog, dc) in zip(df_dogs['Dog'], df_dogs['DC']):
        print(dog, dc)
        df_name = 'df-{}-{}'.format(dog, dc)

        print('\ndf_name: {}\n\n'.format(df_name))

        # selecting dog and dc
        df = df_raw[(df_raw['Dog'] == dog) & (df_raw['DC'] == dc)]

        # list to add tsfel dataframes
        df_list = []
        # Start and Finish Timestamps for each position
        s_times = df.loc[df['Transition'] == True].index[:-1] + t_time 
        f_times = df.loc[df['Transition'].shift(-1) == True].index - t_time

        # Iterating over the postures, taking the steady periods between transitions
        for (s_time, f_time) in zip(s_times, f_times):

            if(df.loc[s_time:f_time].shape[0] >= w_size):

                print('{}\n\tStart: {}\t Finish: {}'
                    .format(df.loc[s_time, 'Position'], s_time, f_time))

                df_list.append(tsfel.time_series_features_extractor(
                                # configuration file with features to be extracted 
                                dict_features = cfg,                            
                                # dataframe window to calculate features window on 
                                signal_windows = df.loc[s_time:f_time, cols],   
                                # name of header columns
                                header_names = cols,
                                # sampling frequency of original signal
                                fs = 100,
                                # sliding window size
                                window_size = w_size, 
                                # overlap between subsequent sliding windows
                                overlap = w_overlap,
                                # do not create a progress bar
                                verbose = 0).assign(
                                Timestamp = s_time, 
                                Breed = df.loc[s_time, 'Breed'],
                                Dog = df.loc[s_time,'Dog'], 
                                DC = df.loc[s_time,'DC'], 
                                Type = df.loc[s_time,'Type'], 
                                Position = df.loc[s_time, 'Position']))
                
        df_feat = pd.concat(df_list)
        print(df_feat.columns)
        df_feat.drop(0)
        df_feat.set_index('Timestamp', inplace = True)
        print('Save {}.csv'.format(df_name))
        df_feat.to_csv('%s\\%s.csv' % (df_dir, df_name))

    print('**Finished saving TSFEL features**')
    return(df_feat)

def gravity_body_components(X):
    """
    Separate acceleration gravity/body components column-wise.

    Uses an elliptic IIR low-pass filter according to Karantonis et
    al. (2006).

    Parameters
    ----------
    X : (np.array [n,p])
            matrix with the data
    freq : (float)
            Sampling frequency in Hertz.

    Returns
    ----------
    G (np.array [n,p]), B (np.array [n,p])
    gravity and body acceleration, respectively.

    """
    print('X:',X)
    freq = 100
    cutoff_hz = 0.25
    nyq = 0.5 * freq
    norm_cutoff = cutoff_hz / nyq
    # Elliptic (Cauer) digital/analog filter, return filter coefficients
    b, a = signal.ellip(
        # order of the filter
        N = 3, 
        # normalised cutoff frequency
        Wn=norm_cutoff, 
        # ripple in the passband in dB
        rp=0.01,
        # attenuation stop band in dB 
        rs=100, 
        # type of filter lowpass
        btype='lowpass')
    def f(x):
            return signal.lfilter(b, a, x)
    G = np.apply_along_axis(func1d=f, axis=0, arr=X)
    print('G:', G)
    B = X.values - G
    print('B:',B)
    col_ga = X.name + '.GA'
    col_ba = X.name + '.BA'
    return pd.Series({col_ga: G, col_ba: B})

# Unused / Under development
def split_old (df, prop):
    '''
        I was using this function to split datasets before starting taking into account the breed balance
        This function splits the dataset into two sets, selecting different dogs for each set
        The criteria ranks dogs according to the (1) number of different positions performed
        and (2) number of total observations. 
        Dogs with more variety in positions are placed in the second set (generally test set)
    
    '''
    df_counts = df.groupby(['Dog','Position']).size().reset_index(name = 'Counts')
    df_summary = df_counts.groupby('Dog')\
        .agg({'Position':'count', 'Counts': 'sum'})\
        .reset_index()
    df_summary.sort_values(['Position', 'Counts'], ascending = False, ignore_index = True, inplace = True)
    df_summary['Cum_Percentage'] = df_summary['Counts'].cumsum()/df_summary['Counts'].sum()

    # adding a column Breed by mapping dog's name to dictionary
    df_summary['Breed'] = df_summary['Dog'].map(dict(zip(df.Dog, df.Breed)))
    # proportion of dogs per breed in the original dataframe
    breed_prop = df_summary.groupby('Breed')['Breed'].count()/df_summary.shape[0]
    
    idx = np.argmin(abs(df_summary['Cum_Percentage'] - prop))
    dogs_chunk = df_summary.Dog[0:idx+1].to_list()
    #print(dogs_chunk)

    df1 = df[~df.Dog.isin(dogs_chunk)]
    df2 = df[df.Dog.isin(dogs_chunk)]

    return(df1, df2)

def balance (df, label):
    '''
        Balances df based on label
        Naive Undersampling, does not take into account the dogs  
    '''
    print('\nBalancing df for label', label , '\n')
    df_list = []
    min_sample = np.min(df[label].value_counts())
    for pos in df[label].unique():
        df_list.append(df[df[label] == pos].sample(min_sample))
    df_balanced = pd.concat(df_list)
    return df_balanced
