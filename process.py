import pandas as pd
import numpy as np
from setup import log
from math import ceil
import datetime

logger = log(__name__)

def label(df_info, df_pos, df_imu, df_dir):
    '''
        Combines data from df_imu and df_info to create a 
            df containing raw df_imu data plus Dog, DC, Type, Position 
                based on the markings df_pos

        df_info: df containing 'Subject', 'DC', 'Data' and 'Start Time'
        df_pos: df containing timestamps data 'Position', 'Pos-VT' and 'Duration'
        df_imu: df containing Actigraph data (back, chest, neck)*(3-axis)*(acc, gyr, mag)
        df_dir: directory to save new dataframe

    '''
    df_list = []  
    for subj in df_info['Subject'].unique():        
        # Iterating through data collections
        for dc in df_info[df_info.Subject == subj]['DC']:     
            print('\t',subj, dc)
            for (s_time, f_time) in zip(df_pos[subj][dc].index.to_series(), \
                                df_pos[subj][dc].index.to_series().shift(-1)):
                #print(s_time, f_time)    
                df_imu[subj][dc]['Dog'] = subj
                df_imu[subj][dc]['DC'] = dc
                df_imu[subj][dc].loc[s_time:f_time,'Type'] = df_pos[subj][dc].loc[s_time, 'Type']
                df_imu[subj][dc].loc[s_time:f_time,'Position'] = df_pos[subj][dc].loc[s_time, 'Position']

               
            df_list.append(df_imu[subj][dc])
    df = pd.concat(df_list)
    # Deleting rows with nan 
    df.dropna(axis = 0, inplace = True)
    # Deleting rows with 'Moving'
    df = df[df['Position'] != 'moving']
    df.to_csv('%s\\%s.csv' % (df_dir, 'df_raw'))
    return(df)



def simple_features(df_raw, df_dir, df_name, w_size, w_offset, t_time):
    '''     
    Extract 'mean','std', 'median', 'min', 'max' from df_imu 
    based on timestamps for positions in df_pos
    Concatenate all the results in a df_raw and add column for 'Type'
    objects:
    df_raw: dataframe with all raw IMU measurements and info 'Dog', 'DC' & 'Position', 'Type'
    df_imu: dataframe containing Actigraph data (back, chest, neck)*(3-axis)*(acc, gyr, mag)
    params: pr_feat contains columns for
    df_name = dataset name
    w_size = size of the window df_feat.ix[df_feat.index.get_loc(s_time)+1, 'Trans-Time']
    w_offset = offset from start time for the value to be taken
    t_time = transition time between positions
    return:
    df containing features calculated and label 'Position' and 'Type'

    '''
    # Finding transitions in posture
    df_raw['Trans-Pos'] = df_raw['Position'].shift()  != df_raw['Position']
    # Finding transitions in time that are bigger than the 100Hz -> 10,000 microseconds
    df_raw['Trans-Time'] = (df_raw.index.to_series().diff() != datetime.timedelta(microseconds = 10000)) + (df_raw.index.to_series().diff().shift(-1) != datetime.timedelta(microseconds = 10000))
    # Combining the time and position transitions
    df_raw['Transition'] = df_raw['Trans-Pos'] + df_raw['Trans-Time']
    # Changing last row into a transition, Transition column has s_time and f_time of the BT
    df_raw.iloc[-1]['Transition'] = True

    df_l2 = []
    # Iterating over the periods between transitions
    t_time = 0 
    for (s_time, f_time) in zip( df_raw.loc[df_raw['Transition'] == True].index[:-1] + t_time , \
                                df_raw.loc[df_raw['Transition'] == True].index[1:] - t_time):

        #print (s_time,  f_time, df_raw.loc[s_time, ['Dog','DC', 'Position']].values)

        if(~df_raw.ix[df_raw.index.get_loc(s_time)+1, 'Trans-Time']):
            #print('\tCalculating Features\n ')
            df_l1 = []        
            
            df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).mean()).resample(w_offset).first())
            df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).std()).resample(w_offset).first())
            df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).median()).resample(w_offset).first())
            df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).min()).resample(w_offset).first())
            df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).max()).resample(w_offset).first())
            #df_l1.append((df_imu[subj][dc][s_time:f_time].rolling(w_size, center = True).sem()).resample(w_offset).first())
        #else:
            #print('Do not calculate features\n\n' )

        df_l2.append( pd.concat(df_l1, axis = 1, keys = ['mean','std', 'median', 'min', 'max'],\
        names = ['Statistics','BodyPart.SensorAxis'])\
        .assign(Dog = df_raw.loc[s_time,'Dog'], DC = df_raw.loc[s_time,'DC'], Type = df_raw.loc[s_time,'Type'], Position = df_raw.loc[s_time, 'Position']))  
        
    df = pd.concat(df_l2)
    # Renaming the columns to contain stats.bodypart.sensor.axis, e.g. mean.Back.Acc.X, keeping last 4 columns (info and label) the same
    df.columns = df.columns[:-4].map('{0[0]}.{0[1]}'.format).append(df.columns[-4:].droplevel(1))
    
    df.to_csv('%s\\%s.csv' % (df_dir, df_name))
    df_logger = log(df_name, log_file = '%s\\%s.log' % (df_dir, df_name))
    df_logger.info('\n\t Dataset created with simple_feature parameters: \n\ndf_name: {}, w_size: {}, w_offset: {}, t_time: {}'.format(df_name, w_size, w_offset, t_time))
    df_logger.info('\n\t Number of Examples in raw dataframe \n {} \n\n {} '.format(df['Position'].value_counts(), df['Type'].value_counts()))
    df_logger.info('\n\t Including data from  \n {}'.format( df.groupby(['Dog', 'DC']).size() ))
    logger.info('\t {}: Dataset created with simple_feature parameters'.format(df_name))

    return (df)

def error_check (df):
    w = df.groupby(['Dog', 'Position', 'DC'], as_index = False).agg(['count'])
    x = df.groupby(['Dog', 'Position'], as_index = False).agg(['count'])
    y = x.groupby(['Position'], as_index = False).agg(['mean', 'median', 'std'])
    return(w,x,y)

def split (df):
    size_total = df['Dog'].unique().size
    size_test = ceil(size_total * 0.2)
    w = df.groupby(['Dog', 'Position'], as_index = False).size().reset_index(name='counts')
    w.groupby(['Dog']).size().reset_index(name='counts')
    x = w.groupby(['Dog'])['Dog', 'Position', 'counts'].filter(lambda x: len(x) == 9)
    if (size_test == x['Dog'].unique().size) :
        x.groupby(['Position'], as_index = False)['counts'].sum().sort_values('counts')
        dogs_test  = x['Dog'].unique()
    df_test = df[df.Dog.isin(dogs_test)]
    df_dev = df[~df.Dog.isin(dogs_test)]
    return(df_test, df_dev)

def balance (df, label):
    '''
        Balances df based on label
    '''
    print('\nBalancing df for label', label , '\n')
    df_list = []
    min_sample = np.min(df[label].value_counts())
    for pos in df[label].unique():
        df_list.append(df[df[label] == pos].sample(min_sample))
    df_balanced = pd.concat(df_list)
    return df_balanced
