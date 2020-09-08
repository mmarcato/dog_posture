import pandas as pd
import numpy as np
from setup import log

logger = log(__name__)

def dummy():
    print('Hello dummy')
    logger.info('\n\t Creating Dataset with the following feature parameters: ')

def simple_features(subjects, dcs, df_pos, df_imu, df_dir, df_name, w_size, w_offset, t_time):
    '''     Extract 'mean','std', 'median', 'min', 'max' from df_imu 
            based on timestamps for positions in df_pos
        Concatenate all the results in a df_feat and add column for 'Type'
    objecst:
        subjects: list of all name of dogs
        dcs: list of all dcs to be included
        df_pos: dataframe with columns 'Position', 'Pos-VT'and 'Duration'
        df_imu: dataframe containing Actigraph data
    params: pr_feat contains columns for
        df_name = dataset name
        w_size = size of the window 
        w_offset = offset from start time for the value to be taken
        t_time = transition time between positions
    return:
        df containing features calculated and label 'Position' and 'Type'

    '''
    df_l2 = []  
    print('Calculating simple_features for df_feat')
    for subj in subjects:
        for dc in dcs:      
            # Checking if df_pos and df_imu exists 
            if df_pos[subj][dc] is not None and df_imu[subj][dc] is not None:
                print('\t',subj, dc)
                for (s_time, f_time) in zip( df_pos[subj][dc].index.to_series() + t_time , \
                                df_pos[subj][dc].index.to_series().shift(-1) - t_time):
                        
                    df_l1 = []        
                    
                    df_l1.append((df_imu[subj][dc][s_time:f_time].rolling(w_size, center = True).mean()).resample(w_offset).first())
                    df_l1.append((df_imu[subj][dc][s_time:f_time].rolling(w_size, center = True).std()).resample(w_offset).first())
                    df_l1.append((df_imu[subj][dc][s_time:f_time].rolling(w_size, center = True).median()).resample(w_offset).first())
                    df_l1.append((df_imu[subj][dc][s_time:f_time].rolling(w_size, center = True).min()).resample(w_offset).first())
                    df_l1.append((df_imu[subj][dc][s_time:f_time].rolling(w_size, center = True).max()).resample(w_offset).first())
                    #df_l1.append((df_imu[subj][dc][s_time:f_time].rolling(w_size, center = True).sem()).resample(w_offset).first())
                    

                    df_l2.append( pd.concat(df_l1, axis = 1, keys = ['mean','std', 'median', 'min', 'max'], \
                        names = ['Statistics','BodyParts', 'Sensor Axis'])\
                        .assign(Dog = subj, DC = dc, Position = df_pos[subj][dc]['Position'][(s_time-t_time)], Type = df_pos[subj][dc]['Type'][(s_time-t_time)]))  
        
    df = pd.concat(df_l2)
    # Deleting rows with nan 
    df.dropna(axis = 0, inplace = True)
    # Deleting rows with 'Moving'
    df = df[df['Position'] != 'moving']

    #df.to_csv('%s\\%s.csv' % (df_dir, df_name))
    
    #logger.info('\n\t Dataset created with feature parameters: \n df_name: {}, w_size: {}, w_offset: {}, t_time: {}'.format(df_name, w_size, w_offset, t_time))
    #logger.info('\n\t Number of Examples in raw dataframe \n {} \n\n {} '.format(df['Position'].value_counts(), df['Type'].value_counts()))

    return (df)

def error_check (df):
    x = df.groupby(['Dog', 'Position'], as_index = False).agg(['count'])
    y = x.groupby(['Position'], as_index = False).agg(['mean', 'median', 'std'])
    print(y)

def split (df, test_dogs ):
    df_test  = 0
    df_train = 0


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
