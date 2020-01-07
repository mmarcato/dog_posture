# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:18:58 2019

@author: marinara.marcato
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# ------------------------------------------------------------------------- #
#                               Parametes                                   #    
# ------------------------------------------------------------------------- #

# file path to csv file Summary 
dir_sum = "Z:\\Tyndall\\IGDB\\Observational Study\\Data Collection\\Study\\Data Collection - Summary.csv"
# directory containing the raw Actigraph files 
dir_raw = "Z:\\Tyndall\\IGDB\\Observational Study\\Data Collection\\Study\\Actigraph\\"
# directory where new split files will be created
dir_new = "Z:\\Tyndall\\IGDB\\Observational Study\\Data Collection\\Study\\Subjects\\"

# all data when Actigraph data was collected
dates = os.listdir(dir_raw)
# select dates when the Inertial data was collected
dates = ['2019-05-23']

# ------------------------------------------------------------------------- #
#                       Importing Summary data                              #    
# ------------------------------------------------------------------------- #
df_sum = {}

df_sum['info'] = pd.read_csv(dir_sum, skiprows = 1,index_col = [1], usecols = [0,1,2,3,4,5,17])
df_sum['info']['Intake'] = df_sum['info']['Intake'].fillna(method = 'ffill', inplace = True)

parse_dict = { 'DT-Pre-Saliva': [4,5],'DT-BT Start': [4,6], 'DT-BT Finish': [4,8],'DT-Post-Saliva': [4,9], 'DT': [4]} 

df_sum['dc1'] = pd.read_csv(dir_sum, skiprows = 1, index_col = 'Name', usecols = [1]+list(range(6,16)), parse_dates = parse_dict, dayfirst = True)
df_sum['dc2'] = pd.read_csv(dir_sum, skiprows = 1, index_col = 'Name', usecols = [1]+list(range(17,27)), parse_dates = parse_dict, dayfirst = True)
df_sum['dc2'].columns = df_sum['dc1'].columns

# ------------------------------------------------------------------------- #
#             Converting DataFrame columns to correct datatype              #    
# ------------------------------------------------------------------------- #

for dc in ['dc1', 'dc2']:
    for col in ['BT Duration', 'Total Duration']:
        df_sum[dc][col] = pd.to_timedelta(df_sum[dc][col])
    for col in ['DT-Pre-Saliva','DT-BT Start', 'DT-BT Finish','DT-Post-Saliva']: 
        df_sum[dc][col] = pd.to_datetime(df_sum[dc][col], format = '%d/%m/%Y %H:%M:%S',dayfirst= True, errors='coerce')
    # this was already transformed to datetime when read_csv
    df_sum[dc]['DT'] = pd.to_datetime(df_sum[dc]['DT'], format =  '%d/%m/%Y', dayfirst= True, errors='coerce')

# ------------------------------------------------------------------------- #
#            Still working on this              #    
# ------------------------------------------------------------------------- #
# Given a data collection day, get the name of the dogs, upload the Artigraph files (back, chest and neck)
# the start and finish time for their BT
# slice the original dataframes 
df_raw = {}
for date in dates:
    print(date)
    for dc in ['dc1', 'dc2']:
        subjects =   list(df_sum[dc].index[df_sum[dc]['DT'] == date].values)
        print(subjects)
        for subj in subjects:
            print(subj)                
            start = df_sum[dc]['DT-BT Start'][subj].time()
            finish = df_sum[dc]['DT-BT Finish'][subj].time()
            for sensor in ['Back','Chest','Neck']:
                file_raw = glob.glob ('%s\\%s*-IMU.csv' % (dir_raw + date, sensor))
                df_raw[sensor] = pd.read_csv(file_raw[0], skiprows = 10, index_col = 'Timestamp', parse_dates = [0], infer_datetime_format = True) 
                df_raw[sensor].between_time(start,finish).to_csv('%s\\%s_Actigraph\\%s_%s.csv' % (dir_new + subj, dc[-1], date, sensor))
