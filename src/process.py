import pandas as pd
import numpy as np
from setup import log
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logger = log(__name__)


def features(df_raw, df_dir, df_name, w_size, w_offset, t_time):
    '''     
    Extract 'mean','std', 'median', 'min', 'max' from df_raw 
    based on timestamps for positions in df_pos
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
    print('Processing simple features')
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
    for (s_time, f_time) in zip( df_raw.loc[df_raw['Transition'] == True].index[:-1] + t_time , \
                                df_raw.loc[df_raw['Transition'].shift(-1) == True].index - t_time):
        print (s_time,  f_time)
        # if there is not a transition in time -> transitions in time 
        if(~df_raw.ix[df_raw.index.get_loc(s_time-t_time)+1, 'Trans-Time']):
        #if(s_time < f_time):
            print('\tCalculating Features\n ')
            df_l1 = []   
            
            df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).mean()).resample(w_offset).first())
            df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).std()).resample(w_offset).first())
            df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).median()).resample(w_offset).first())
            df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).min()).resample(w_offset).first())
            df_l1.append((df_raw.ix[s_time:f_time, :-7].rolling(w_size, center = True).max()).resample(w_offset).first())
            #df_l1.append((df_imu[subj][dc][s_time:f_time].rolling(w_size, center = True).sem()).resample(w_offset).first())
            #print(df_l1)

            print (df_raw.loc[s_time, ['Dog','DC', 'Position']].values)

            df_l2.append( pd.concat(df_l1, axis = 1, keys = ['mean','std', 'median', 'min', 'max'],\
            names = ['Statistics','BodyPart.SensorAxis'])\
            .assign(Dog = df_raw.loc[s_time,'Dog'], DC = df_raw.loc[s_time,'DC'], Type = df_raw.loc[s_time,'Type'], Position = df_raw.loc[s_time, 'Position']))  
           
        else:
            print('Do not calculate features\n\n' )

    df = pd.concat(df_l2)
    # Renaming the columns to contain stats.bodypart.sensor.axis, e.g. mean.Back.Acc.X, keeping last 4 columns (info and label) the same
    df.columns = df.columns[:-4].map('{0[0]}.{0[1]}'.format).append(df.columns[-4:].droplevel(1))
    print('Shape before before dropping NAs', df.shape)
    df = df.dropna()
    print('Shape before after dropping NAs', df.shape)

    print('Save df to csv')
    df.to_csv('%s\\%s.csv' % (df_dir, df_name))
    df_logger = log(df_name, log_file = '%s\\%s.log' % (df_dir, df_name))
    df_logger.info('\n\t Dataset created with simple_feature parameters: \n\ndf_name: {}, w_size: {}, w_offset: {}us, t_time: {}us'.format(df_name, w_size, w_offset, t_time))
    df_logger.info('\n\t Number of Examples in raw dataframe \n {} \n\n {} '.format(df['Position'].value_counts(), df['Type'].value_counts()))
    df_logger.info('\n\t Including data from  \n {}'.format( df.groupby(['Dog', 'DC']).size() ))
    logger.info('\t {}: Dataset created with simple_feature parameters'.format(df_name))

    return (df)


def distribution (df):
    print('Distribution of Positions per dog')
    plt.figure(figsize=(10,5))
    chart = sns.countplot(x="Position", hue="Dog", data = df)
    chart.set_title('Distribution of Positions per dog')
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    return(df.groupby(['Position', 'Dog']).size().reset_index(name='count'))

def split (df):
    '''
        split the dataset into development and test sets respecting the groups (dogs)
        selects dogs for the test set that have all 9 body positions 
        this function is not optimal really, see the if condition
    '''
    # total number of unique dogs
    size_total = df['Dog'].unique().size
    # 20% of total number of unique dogs
    size_test = round(size_total * 0.2)
    print(size_test)

    df_counts = df.groupby(['Dog','Position']).size().reset_index(name = 'Counts')
    df_summary = df_counts.groupby('Dog').sum()
    df_summary['Positions'] = df_counts.groupby('Dog').size()
    df_summary.sort_values(['Positions', 'Counts'], ascending = False, inplace = True)

    dogs_test = list(df_summary.index[0:size_test])

    df_test = df[df.Dog.isin(dogs_test)]
    df_dev = df[~df.Dog.isin(dogs_test)]

    #logger = log(df_name, log_file = '%s\\%s.log' % (df_dir, df_name))
    logger.info('\n\n Percentage in Test Set: {} \n Percentage in Dev Set: {}'.format(df_test.shape[0]/(df_test.shape[0]+df_dev.shape[0]), df_dev.shape[0]/(df_test.shape[0]+df_dev.shape[0])) )
    
    return(df_dev, df_test)


def balance (df, label):
    '''
        Balances df based on label
    '''
    print('\nBalancing df for label', label , '\n')
    #df = df_feat
    label = 'Position'
    df_list = []
    min_sample = np.min(df[label].value_counts())
    
    for pos in df[label].unique():
        df_list.append(df[df[label] == pos].sample(min_sample))
    df_balanced = pd.concat(df_list)
    return df_balanced


def naive_balance (df, label):
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
