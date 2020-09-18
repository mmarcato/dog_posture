import os
import pandas as pd
import glob
from setup import log

logger = log(__name__)

def timestamps(subjects, dcs, base_dir): 
    '''
    import timestamps files and organised data into and return:

    df_ep: df indexed by'Timestamps' containing'Episode', 'Ep-VT' and 'Duration'
    df_pos: df indexed by 'Timestamps' containing 'Position', 'Pos-VT', 'Duration', 'Type'
    df_stats: df containing 'Subject', 'DC', 'Date', 'Start time' 
    '''
    print('\nImporting Timestamp files - Episode and Position Data') 
    stats = []
    df_ep, df_pos = {},{}
    for subj in subjects:
        df_ep[subj], df_pos[subj] = {},{}
        for dc in dcs:
            df_ep[subj][dc], df_pos[subj][dc] = None, None
            f_name = '%s\\%s\\%s_Timestamps.csv' % (base_dir, subj, dc[-1])  
            if os.path.exists(f_name):
                # Read the information about the behaviour test 
                df_info = pd.read_csv(f_name, index_col = 0, nrows = 4, usecols = [0,1])
                date = df_info[subj]['Date']
                time = df_info[subj]['Start time']
                dt = pd.to_datetime(date + time, format = '%d/%m/%Y%H:%M:%S' )            
                stats.append([subj, dc, date, time])
                # Read the episode Virtual Time (VT) 
                df_ep[subj][dc] = pd.read_csv(f_name, skiprows = 6, usecols = ['Episode', 'Ep-VT']).dropna()
                # Create new column for the episode Real Time (RT)
                df_ep[subj][dc].index = dt + pd.to_timedelta(df_ep[subj][dc]['Ep-VT'])         
                # Create new column for the episode Duration
                df_ep[subj][dc]['Duration'] = df_ep[subj][dc].index.to_series().diff().shift(-1)
                df_ep[subj][dc]['Episode'] = df_ep[subj][dc]['Episode'].str.lower()
                
                # Read the position Virtual Time (VT) 
                df_pos[subj][dc] = pd.read_csv(f_name, skiprows = 6, usecols = ['Position', 'Pos-VT']).dropna()
                # Create new column for the position Real Time (RT) and sets it as the index
                df_pos[subj][dc].index = dt + pd.to_timedelta(df_pos[subj][dc]['Pos-VT'])         
                # Create new column for the position Duration
                df_pos[subj][dc]['Duration'] = df_pos[subj][dc].index.to_series().diff().shift(-1) 
                df_pos[subj][dc]['Position'] = df_pos[subj][dc]['Position'].str.lower()
                
                pos_type = {'walking': 'dynamic', 'w-sniffing floor': 'dynamic',
                            'standing':'static', 'sitting':'static', 
                            'lying down': 'static', 'jumping up': 'dynamic',
                            'jumping down':'dynamic', 'body shake':'dynamic',
                            's-sniffing floor': 'static', 'Pull on leash': 'dynamic',
                            'moving': 'dynamic'}

                df_pos[subj][dc]['Type'] = df_pos[subj][dc]['Position'].map(pos_type)

    df_info = pd.DataFrame(stats, columns = ['Subject', 'DC', 'Date', 'Start time'])
    logger.info('\t Imported Timestamps for \n{}'.format(df_info))

    return(df_info, df_pos, df_ep)

def actigraph(df_info, base_dir):
    df_imu = {}
    bps = ['Back', 'Chest', 'Neck']
    print('\nImporting Actigraph files - IMU Data')
    logger.info('\t Started Importing Actigraph data')
    for subj in df_info['Subject'].unique():
        df_imu[subj] = {}       
        # Iterating through data collections
        for dc in df_info[df_info.Subject == subj]['DC']:
            df_list= []
            df_imu[subj][dc] = None
            # If this the path to data exists
            if os.path.isdir('%s\\%s\\%s_Actigraph' % (base_dir, subj, dc[-1])):
                # Looping through all bps
                for bp in bps:   
                        # Find file path for each bp
                        f_name =  glob.glob('%s\\%s\\%s_Actigraph\\*_%s.csv' % (base_dir, subj, dc[-1], bp))
                    
                        df_list.append(pd.read_csv(f_name[0], index_col = ['Timestamp'], parse_dates = [0], \
                                date_parser = lambda x: pd.to_datetime(x, format = '%Y-%m-%d %H:%M:%S.%f'))\
                                .drop(['Temperature'], axis = 1))
                # Concatenating dataframes for different body parts in one single dataframe
                # Results in one dataframe per dog per data collection
                df_imu[subj][dc] = pd.concat(df_list, axis = 1, keys = bps, \
                names = ['Body Parts', 'Sensor Axis'])
                # Change column names to be bodypart.sen.axis (Back.Acc.X)
                df_imu[subj][dc].columns = [f'{i}.{j[:3]}.{j[-1]}' for i,j in df_imu[subj][dc].columns]
    logger.info('\t Finished Importing Actigraph data')
    return(df_imu)


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


def posture(df_dir, df_name = 'df_raw'):
    return(pd.read_csv( '%s\\%s.csv' % (df_dir, df_name), index_col = ['Timestamp'], parse_dates = [0], \
                                date_parser = lambda x: pd.to_datetime(x, format = '%Y-%m-%d %H:%M:%S.%f')))
