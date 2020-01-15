import os
import pandas as pd
import glob
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('data_import.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

def timestamps (subjects, dcs, dir_base): 
    '''
    import timestamps files and organised data into and return:

    df_ep: df indexed by'Timestamps' containing'Episode', 'Ep-VT' and 'Duration'
    df_pos: df indexed by 'Timestamps' containing 'Position', 'Pos-VT', 'Duration', 'Type'
    df_stats: df containing 'Subject', 'DC', 'Date', 'Start time' 
    '''
    print('\nRetrieving Timestamp files - Episode and Position Data') 
    stats = []
    df_ep, df_pos = {},{}
    for subj in subjects:
        df_ep[subj], df_pos[subj] = {},{}
        for dc in dcs:
            df_ep[subj][dc], df_pos[subj][dc] = None, None
            f_name = '%s\\%s\\%s_Timestamps.csv' % (dir_base, subj, dc[-1])  
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
                # Create new column for the position Real Time (RT)
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
    #logger.info('Added data for {}'.format(df_info['Subject'].values))
    return(df_info, df_pos, df_ep)
        
def actigraph(subjects, dcs, dir_base):
    df_imu = {}
    bps = ['Back', 'Chest', 'Neck']
    print('\nRetrieving Actigraph files - IMU Data')
    for subj in subjects:
        df_imu[subj] = {}       
        # Iterating through data collections
        for dc in dcs:
                df_list= []
                df_imu[subj][dc] = None
                # If this the path to data exists
                if os.path.isdir('%s\\%s\\%s_Actigraph' % (dir_base, subj, dc[-1])):
                    print('\t', subj, dc)
                    # Looping through all bps
                    for bp in bps:   
                            # Find file path for each bp
                            f_name =  glob.glob('%s\\%s\\%s_Actigraph\\*_%s.csv' % (dir_base, subj, dc[-1], bp))
                        
                            df_list.append(pd.read_csv(f_name[0], index_col = ['Timestamp'], parse_dates = [0], \
                                    date_parser = lambda x: pd.to_datetime(x, format = '%Y-%m-%d %H:%M:%S.%f'))\
                                    .drop(['Temperature'], axis = 1))
                    # Concatenating dataframes for different body parts in one single dataframe
                    # Results in one dataframe per dog per data collection
                    df_imu[subj][dc] = pd.concat(df_list, axis = 1, keys = bps, \
                    names = ['Body Parts', 'Sensor Axis'])
    return(df_imu)