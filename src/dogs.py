import pandas as pd
df_dir = 'C:\\Users\\marinara.marcato\\Project\\Scripts\\dog_posture\\dfs'
df_name = 'df_dogs'
df_dogs = pd.read_csv( '%s\\%s.csv' % (df_dir, df_name), \
    usecols = ['Intake', 'Code', 'Name', 'DOB', 'DOA', 'PR Sup', 'Sex', 'Source'],\
    parse_dates = ['DOB', 'DOA']) #, \
    #date_parser = lambda x: pd.to_datetime(x, format = '%Y-%m-%d')))
