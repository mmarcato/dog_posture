# ------------------------------------------------------------------------- #
#                                  Imports                                  #    
# ------------------------------------------------------------------------- # 
from src.__modules__ import imports
from src.__modules__ import process 

# directory where the dataset is located
df_dir = '..//data//processed//'

# importing previously created dataset
df_feat = imports.posture(df_dir, 'df_12')  

# creating dev and test sets
df_dev, df_test = process.split(df_feat, 0.2)
df_train, df_val = process.split(df_dev, 0.25)

# ------------------------------------------------------------------------- #
#                            Data Visualisations                            #    
# ------------------------------------------------------------------------- # 
# visualising feature distribution  
df_dist = process.distribution(df_feat, 'Original Dataset')

# visualising feature distribution for dev and test sets
process.distribution(df_dev, 'Development Dataset')
process.distribution(df_test, 'Test Dataset')

# visualising feature distribution for dev and test sets
process.distribution(df_train, 'Train Dataset')
process.distribution(df_val, 'Validation Dataset')
process.distribution(df_test, 'Test Dataset')