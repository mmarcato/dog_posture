''' Copyright (C) 2021 by Marinara Marcato
         <marinara.marcato@tyndall.ie>, Tyndall National Institute
        University College Cork, Cork, Ireland.
'''
# ------------------------------------------------------------------------- #
#                           Importing Global Modules                        #
# ------------------------------------------------------------------------- #
import os, sys
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
#                              Importing Raw Dataset                        #
# ------------------------------------------------------------------------- #

# directory to retrieve raw file
dir_raw = os.path.join(dir_base, 'data', 'raw')
# importing created raw dataset 
df_raw = imports.posture(dir_raw, 'df_raw5')


# ------------------------------------------------------------------------- #
#             Decomposing Raw -> Body & Gravitational components            #
# ------------------------------------------------------------------------- #

# gravity, body = process.gravity_body_components(df_raw.iloc[:, 0], freq = 100)
# df = df_raw.iloc[:100, [0,1]]
# df_new = pd.DataFrame(index = df_raw.index)
# # funtion that returns a new dataframe with two columns (body and gravitational accelerations)
# df_new = df_new.merge(
#             df.map(process.gravity_body_components)
#                 # applying on columns
#                 left_index=True, right_index=True)

# df_new = df_new.merge(df.apply(lambda c: pd.Series({'feature1':c+1, 'feature2':c-1})), 
#     left_index=True, right_index=True)

# ------------------------------------------------------------------------- #
#                              Creating New Dataset                         #
# ------------------------------------------------------------------------- #

# directory to save new file 
dir_new = os.path.join(dir_base, 'data', 'tsfel')

# window settings
w_size = 100
w_overlap = .5
t_time = timedelta(seconds = .25)

# Select specific dog and dc combination for fast code completion
# df_dogs = pd.DataFrame({
#                 'Dog' : ['Fawn', 'Fawn', 'Joy'],
#                 'DC' : [1, 2, 1]
#         })

# Select all Dog and DC combinations
df_dogs = df_raw.groupby(['Dog', 'DC']).count().reset_index()[['Dog', 'DC']]
print("Number of unique data collections: ", df_dogs.shape[0])
print("Number of dogs: ", len(df_dogs['Dog'].unique()))

# There is a bug in this package with the progress bar
# it won't run in the interactive window
process.features_tsfel(df_raw, dir_new, df_dogs, w_size, w_overlap, t_time)

