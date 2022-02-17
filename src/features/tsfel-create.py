''' Copyright (C) 2022 by Marinara Marcato
         <marinara.marcato@tyndall.ie>, Tyndall National Institute
        University College Cork, Cork, Ireland.
'''
# ------------------------------------------------------------------------- #
#                           Importing Global Modules                        #
# ------------------------------------------------------------------------- #
import os, sys
import numpy as np
import pandas as pd
from datetime import timedelta

# ------------------------------------------------------------------------- #
#                           Importing Local Modules                         #
# ------------------------------------------------------------------------- #

dir_current = os.path.dirname(os.path.realpath(__file__))
dir_parent = os.path.dirname(dir_current)
dir_base = os.path.dirname(dir_parent)
dir_modules = os.path.join(dir_base, 'src', '__modules__')
# Set path variable
sys.path.append(dir_modules)

# Local Modules
# %load_ext autoreload
# %autoreload 2
import imports, process


# ------------------------------------------------------------------------- #
#                           Import Dasets from Folder                       #
# ------------------------------------------------------------------------- #

dir_df = os.path.join(dir_base, 'data', 'tsfel')

# import all 
df_new = imports.features_tsfel(dir_df)
df_new.to_csv(os.path.join(dir_df, 'df-all.csv'))


imports.split(df_new, dir_df, 'df5-all')

# # splitting the entire dataset into dev and test sets
# df_dev, df_test = process.split(df_new, 0.2)
# # splitting dev dataset into train and val
# df_train, df_val = process.split(df_dev, 0.25)

# # saving dataframes
# df_test.to_csv('%s\\%s.csv' % (dir_df, 'data_test'))
# df_dev.to_csv('%s\\%s.csv' % (dir_df, 'data_dev'))
# df_train.to_csv('%s\\%s.csv' % (dir_df, 'data_dev'))
# df_val.to_csv('%s\\%s.csv' % (dir_df, 'data_val'))
