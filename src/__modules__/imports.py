# ------------------------------------------------------------------------- #
#                                   Imports                                 #    
# ------------------------------------------------------------------------- # 
 
import os
import pandas as pd
import glob


# ------------------------------------------------------------------------- #
#                                  Functions                                #    
# ------------------------------------------------------------------------- # 


def timestamps(df_data, df_dir): 
    """
    Imports data from timestamps files, organise them in dictionaries and return

    Parameters
        -------
        df_data : DataFrame
            columns are subjects, dcs: unique combinations of dog name & dc number
        df_dir : str
            directory where timestamps are located

    Returns
        -------
        df_ep: DataFrame
            indexed by'Timestamps' containing 'Episode', 'Ep-VT' and 'Duration'
        df_pos: DataFrame
            indexed by 'Timestamps' containing 'Position', 'Pos-VT', 'Duration', 'Type'
        df_stats: DataFrame
            containing 'Subject', 'DC', 'Date', 'Start time' 
    """

    print('\nImporting Timestamp files - Episode and Position Data') 
    stats = []
    df_ep, df_pos = {},{}

    for subj in df_data['Dog'].unique():
        df_ep[subj], df_pos[subj] = {},{}
        for dc in df_data.loc[df_data['Dog'] == subj, 'DC']:
            df_ep[subj][dc], df_pos[subj][dc] = None, None
            f_name = '%s\\%s\\%s_Timestamps.csv' % (df_dir, subj, dc) 
            # if the timestamp file is found 
            if os.path.exists(f_name):            
                # Read the information about the behaviour test 
                df_info = pd.read_csv(f_name, index_col = 0, nrows = 4, usecols = [0,1], dayfirst = False)
                date = df_info[subj]['Date']
                time = df_info[subj]['Start time']     
                stats.append([subj, dc, date, time])
                
                dt = pd.to_datetime(date + time, format = '%d/%m/%Y%H:%M:%S' )       

                # Read the EPISODE Virtual Time (VT) 
                df_ep[subj][dc] = pd.read_csv(f_name, skiprows = 6, usecols = ['Episode', 'Ep-VT']).dropna()
                # Create new column for the episode Real Time (RT)
                df_ep[subj][dc].index = dt + pd.to_timedelta(df_ep[subj][dc]['Ep-VT'])         
                # Create new column for the episode Duration
                df_ep[subj][dc]['Duration'] = df_ep[subj][dc].index.to_series().diff().shift(-1)
                df_ep[subj][dc]['Episode'] = df_ep[subj][dc]['Episode'].str.lower()
                
                # Read the POSITION Virtual Time (VT) 
                df_pos[subj][dc] = pd.read_csv(f_name, skiprows = 6, usecols = ['Position', 'Pos-VT']).dropna()
                # Create new column for the position Real Time (RT) and sets it as the index
                df_pos[subj][dc].index = dt + pd.to_timedelta(df_pos[subj][dc]['Pos-VT'])         
                # Create new column for the position Duration
                df_pos[subj][dc]['Duration'] = df_pos[subj][dc].index.to_series().diff().shift(-1) 

                if any(df_pos[subj][dc]['Duration'] <= pd.Timedelta(0,'seconds')):
                # Flagging positions that are shorter than 0s
                    print('Warning: Position duration is shorter than 0s for', subj, dc, '\n', 
                        df_pos[subj][dc].loc[df_pos[subj][dc]['Duration'] <= pd.Timedelta(0,'seconds'),'Position'],
                    '\n\n')

                df_pos[subj][dc]['Position'] = df_pos[subj][dc]['Position'].str.lower()
                
                pos_type = {'walking': 'dynamic', 'w-sniffing floor': 'dynamic',
                            'standing':'static', 'sitting':'static', 
                            'lying down': 'static', 'jumping up': 'dynamic',
                            'jumping down':'dynamic', 'body shake':'dynamic',
                            's-sniffing floor': 'static', 'Pull on leash': 'dynamic',
                            'moving': 'dynamic'}

                df_pos[subj][dc]['Type'] = df_pos[subj][dc]['Position'].map(pos_type)
            else:
                print('Error loading', subj, dc)
    df_info = pd.DataFrame(stats, columns = ['Subject', 'DC', 'Date', 'Start time'])
    #logger.info('\t Imported Timestamps for \n{}'.format(df_info))

    return(df_info, df_pos, df_ep)

def actigraph(df_info, df_dir):
    df_imu = {}
    bps = ['Back', 'Chest', 'Neck']
    print('\nStarted Importing Actigraph files')
    #logger.info('\t Started Importing Actigraph data')
    for subj in df_info['Subject'].unique():
        df_imu[subj] = {}       
        # Iterating through data collections
        for dc in df_info[df_info.Subject == subj]['DC']:
            df_list= []
            df_imu[subj][dc] = None
            # If this the path to data exists
            if os.path.isdir('%s\\%s\\%s_Actigraph' % (df_dir, subj, dc)):
                # Looping through all bps
                for bp in bps:   
                        # Find file path for each bp
                        f_name =  glob.glob('%s\\%s\\%s_Actigraph\\*_%s.csv' % (df_dir, subj, dc, bp))
                        df_list.append(pd.read_csv(f_name[0], 
                                                    index_col = ['Timestamp'], 
                                                    parse_dates = [0],
                                                    date_parser = lambda x: pd.to_datetime(x, format = '%Y-%m-%d %H:%M:%S.%f'))\
                                        .drop(['Temperature'], 
                                        axis = 1))
                # Concatenating dataframes for different body parts in one single dataframe
                # Results in one dataframe per dog per data collection
                print('\t', subj, dc)
                df_imu[subj][dc] = pd.concat(df_list, axis = 1, keys = bps, \
                        names = ['Body Parts', 'Sensor Axis'])
                # Change column names to be bodypart.sen.axis (Back.Acc.X)
                df_imu[subj][dc].columns = [f'{i}.{j[:3]}.{j[-1]}' for i,j in df_imu[subj][dc].columns]
    print(' Finished Importing Actigraph files')
    #logger.info('\t Finished Importing Actigraph data')
    return(df_imu)

def label(df_info, df_pos, df_imu):
    '''
        Combines data from df_imu and df_info to create a 
        df containing raw df_imu data plus Dog, DC, Type, Position 
        based on the markings df_pos

        df_info: df containing 'Subject', 'DC', 'Data' and 'Start Time'
        df_pos: df containing timestamps data 'Position', 'Pos-VT' and 'Duration'
        df_imu: df containing Actigraph data (back, chest, neck)*(3-axis)*(acc, gyr, mag)
    '''
    print('Started creating labeled raw data')
    df = pd.DataFrame()
    for (subj, dc) in zip(df_info['Subject'], df_info['DC']):    
        
        print('\t',subj, dc)

        # iterating over all postures annotated 
        for (s_time, f_time) in zip(df_pos[subj][dc].index.to_series(), \
                            df_pos[subj][dc].index.to_series().shift(-1)):
            print(s_time, f_time)    
            df_imu[subj][dc].loc[s_time:f_time,'Dog'] = subj
            df_imu[subj][dc].loc[s_time:f_time,'DC'] = dc
            df_imu[subj][dc].loc[s_time:f_time,'Type'] = df_pos[subj][dc].loc[s_time, 'Type']
            df_imu[subj][dc].loc[s_time:f_time,'Position'] = df_pos[subj][dc].loc[s_time, 'Position']
         
        print(df_imu[subj][dc].shape)
        df = df.append(df_imu[subj][dc])
        print(df.shape)
    # Deleting rows with nan 
    df.dropna(axis = 0, inplace = True)
    # Deleting rows with 'Moving'
    df = df[df['Position'] != 'moving']
    print('Finished creating labeled raw data')
    
    return(df)
 
def posture(df_dir, df_name = 'df_raw'):
    df_path = os.path.join(df_dir, "{}.csv".format(df_name))
    df = pd.read_csv( df_path, 
                index_col = ['Timestamp'], 
                parse_dates = ['Timestamp'],
                dayfirst = True,
                date_parser = lambda x: pd.to_datetime(x, format = '%Y-%m-%d %H:%M:%S.%f')    )
    return(df)

def split(df, df_dir, df_name):
    # split dataset taking dog and breed into account

    # separating golden retriever 'Tosh'
    df_gr = df.loc[df['Breed'] == 'GR']

    # test set with 20% of observations, 60% LRxGR (Douglas, Elf, Goober) 40% LR (Meg, July)
    df_test = df[df.Dog.isin(['Douglas', 'Elf', 'Goober', 'Meg', 'July'])]

    # dogs for dev set for 80% observation, 60% LRxGR (Douglas, Elf, Goober) 40% LR (Meg, July) 
    df_dev = df[~df.Dog.isin(['Tosh', 'Douglas', 'Elf', 'Goober', 'Meg', 'July'])]

    df_dev.to_csv(os.path.join(df_dir, df_name +'-dev.csv'))
    df_test.to_csv(os.path.join(df_dir, df_name + '-test.csv'))
    df_gr.to_csv(os.path.join(df_dir,df_name+ '-gr.csv'))

    return(df_dev, df_test, df_gr)

def dogs(df_dir, df_name):
    df_dogs = pd.read_csv( '%s\\%s.csv' % (df_dir, df_name), \
        usecols = ['Intake', 'Code', 'Name', 'DOB', 'DOA', 'PR Sup', 'Sex', 'Source', 'Breed', 'DC1', 'DC2'],\
        parse_dates = ['DOB', 'DOA', 'DC1', 'DC2'])
    return(df_dogs)

def features_tsfel(dir_new):
    
    # view all dataframes in directory
    print(os.listdir(dir_new))

    # read all dataframes into only one dataframe df_new
    df = pd.DataFrame()
    for df_name in os.listdir(dir_new):
        df = df.append(
                    posture(dir_new, os.path.splitext(df_name)[0]))
        print(df_name, df.shape)
    return df