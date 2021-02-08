import pandas as pd
import numpy as np
from setup import log
import datetime
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import roll_time_series

import matplotlib.pyplot as plt
import seaborn as sns

logger = log(__name__)

def transitions(df): 
    '''
        Process df_raw to create:
            1. Transition in Position for consective positions performed by same dog
            2. Transition in Time in case moving in between two positions or different dog
            3. Transition column combining both step 1. and 2.
    '''
    # Finding transitions in posture
    df['Trans-Pos'] = df['Position'].shift()  != df['Position']
    # Finding transitions in time that are bigger than the 100Hz -> 10,000 microseconds
    df['Trans-Time'] = (df.index.to_series().diff() != datetime.timedelta(microseconds = 10000)) + (df.index.to_series().diff().shift(-1) != datetime.timedelta(microseconds = 10000))
    # Combining the time and position transitions
    df['Transition'] = df['Trans-Pos'] + df['Trans-Time']
    # Changing last row into a transition, Transition column has s_time and f_time of the BT
    df.iloc[-1]['Transition'] = True
    
    return(df)


def features(df_raw, df_dir, df_name, w_size, w_offset, t_time):
    '''     
    Extracts 'min', 'max', 'mean','std', 'median', 'sum', 'skew', 'kurt' from a window interval in df_raw 
    based on timestamps for positions in df_raw
    Saves transformed df to a file in df_dir with df_name:
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
    print('Processing simple features \n df_name {}, w_size {}, w_offset {}, t_time{}'.format(df_name, w_size, w_offset, t_time))

    # Finding transitions in posture
    df_raw = transitions(df_raw)

    df_l2 = []
    # Iterating over the periods between transitions
    for (s_time, f_time) in zip( df_raw.loc[df_raw['Transition'] == True].index[:-1] + t_time , \
                                df_raw.loc[df_raw['Transition'].shift(-1) == True].index - t_time):
        
        #print(s_time,  f_time)
        # if there is not a transition in time 
        if(~df_raw.ix[df_raw.index.get_loc(s_time-t_time)+1, 'Trans-Time']):
        
            #print('Calculating Features for {}\n'.format(df_raw.loc[s_time, ['Dog','DC', 'Position']].values))    
            df_l1 = []   
            
            df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).min()).resample(w_offset).first())
            df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).max()).resample(w_offset).first())
            df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).mean()).resample(w_offset).first())
            df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).std()).resample(w_offset).first())
            df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).median()).resample(w_offset).first())
            df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).sum()).resample(w_offset).first())
            #df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).corr()).resample(w_offset).first())
            #df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).cov()).resample(w_offset).first())
            df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).skew()).resample(w_offset).first())
            df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).kurt()).resample(w_offset).first())


            df_l2.append( pd.concat(df_l1, axis = 1, keys = ['min', 'max', 'mean','std', 'median', 'sum', 'skew', 'kurt',],\
            names = ['Statistics','BodyPart.SensorAxis'])\
            .assign(Dog = df_raw.loc[s_time,'Dog'], DC = df_raw.loc[s_time,'DC'], Type = df_raw.loc[s_time,'Type'], Position = df_raw.loc[s_time, 'Position']))  
           
        #else:
            #print('Do not calculate features\n' )

    df = pd.concat(df_l2)
    # Renaming the columns to contain stats.bodypart.sensor.axis, e.g. mean.Back.Acc.X, keeping last 4 columns (info and label) the same
    df.columns = df.columns[:-4].map('{0[0]}.{0[1]}'.format).append(df.columns[-4:].droplevel(1))
    print('Shape before dropping NAs', df.shape)
    df = df.dropna()
    print('Shape after dropping NAs', df.shape)

    print('Save df to csv')
    df.to_csv('%s\\%s.csv' % (df_dir, df_name))
    df_logger = log(df_name, log_file = '%s\\%s.log' % (df_dir, df_name))
    df_logger.info('\n\t Dataset created with simple_feature parameters: \n\ndf_name: {}, w_size: {}, w_offset: {}us, t_time: {}us'.format(df_name, w_size, w_offset, t_time))
    df_logger.info('\n\t Number of Examples in raw dataframe \n{} \n\n{}\n'.format(df['Position'].value_counts(), df['Type'].value_counts()))
    df_logger.info('\n\t Including data from  \n{}\n\n'.format( df.groupby(['Dog', 'DC']).size() ))
    logger.info('{}: Dataset created. See log for parameter details'.format(df_name))

    return (df)

def distribution (df, df_desc):
    print(df_desc)
    # checking the number dogs included
    print('\n\nNumber of Dogs: {}'.format(df['Dog'].unique().size))
    # checking the number DCs included
    print('Number of DCs: {}\n'.format(df.groupby(['Dog' ,'DC']).size().count()))
    df_dogs = df['Dog'].value_counts().reset_index(name= 'count')
    df_dogs['percentage'] = df_dogs['count']*100 /df_dogs['count'].sum()
    print(df_dogs)
    # calculating the number of examples per category
    df_sum = df['Position'].value_counts().reset_index(name= 'count')
    # calculating the percentage of examples per category
    df_sum['percentage'] = df_sum['count']*100 /df_sum['count'].sum()
    print(df_sum)

    print('Distribution of Positions per dog')
    plt.figure(figsize=(10,5))
    chart = sns.countplot(x="Position", hue="Dog", data = df, order = df['Position'].value_counts().index)
    chart.set_title('Distribution of Positions per dog')
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    return(df.groupby(['Position', 'Dog']).size().reset_index(name='count'))


def split (df, p):
    '''
        split the dataset into two sets
        selects different dogs for each set
        dogs with most diverse position set are placed in the second set 
    '''
    df = df_dev
    
    df_counts = df.groupby(['Dog','Position']).size().reset_index(name = 'Counts')
    df_summary = df_counts.groupby('Dog').sum()
    df_summary['Positions'] = df_counts.groupby('Dog').size()
    df_summary.sort_values(['Positions', 'Counts'], ascending = False, inplace = True)
    df_summary['Cum_Percentage'] = df_summary['Counts'].cumsum()/df_summary['Counts'].sum()
    idx = np.argmin(abs(df_summary['Cum_Percentage'] - prop))
    dogs_chunk = df_summary[0:idx+1].index.to_list()

    df1 = df[~df.Dog.isin(dogs_chunk)]
    df2 = df[df.Dog.isin(dogs_chunk)]

    return(df1, df2)

def stats(dfs):
  sizes = list(map(len, dfs))
  print(sizes)
  print([size/sum(sizes) for size in sizes])


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
