# ------------------------------------------------------------------------- #
#                                  Imports                                  #    
# ------------------------------------------------------------------------- # 
## General imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)

 

def stats(dfs):
  sizes = list(map(len, dfs))
  print(sizes)
  print([size/sum(sizes) for size in sizes])


# ------------------------------------------------------------------------- #
#                           EDA - Dog Imbalance                             #    
# ------------------------------------------------------------------------- #

def distribution (df, df_desc):

    print(df_desc)
    # checking the number dogs included
    print('\n\nNumber of Dogs: {}'.format(df['Dog'].unique().size))
    # checking the number DCs included
    print('Number of DCs: {}\n'.format(df.groupby(['Dog' ,'DC']).size().count()))

    df_dogs = df['Dog'].value_counts().reset_index(name= 'count')
    df_dogs['percentage'] = df_dogs['count']*100 /df_dogs['count'].sum()
    print(df_dogs)
    # number of observations per category
    df_sum = df['Position'].value_counts().reset_index(name= 'count')
    # percentage of observations per category
    df_sum['percentage'] = df_sum['count']*100 /df_sum['count'].sum()
    print(df_sum)

    print('Distribution of Positions per dog')
    plt.figure(figsize=(10,5))
    chart = sns.countplot(x="Position", hue="Dog", data = df, order = df['Position'].value_counts().index)
    chart.set_title('Distribution of Positions per dog')
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    return(df.groupby(['Position', 'Dog']).size().reset_index(name='count'))


# ------------------------------------------------------------------------- #
#                       EDA - Breed Balance in dataset                      #    
# ------------------------------------------------------------------------- #

def breed(df):
    '''
        Displays breed information about the dataset
    '''
    # calculating number of dogs per breed and proportion
    df_summary = df.groupby(['Dog', 'Breed'])['Position'].count()
    df_breed = df_summary.groupby(['Breed']).size().reset_index(name = 'Count')
    df_breed['Proportion'] = df_breed['Count'] / df_breed['Count'].sum()
    print(df_breed)
 
# ------------------------------------------------------------------------- #
#                          EDA - Class Imbalance                            #    
# ------------------------------------------------------------------------- #
def label(df):


    # class balance
    df['Type'].value_counts()
    # class balance
    df['Position'].value_counts()

    # Development set Position - calculating the number of examples per category
    df_pos = df['Position'].value_counts().reset_index(name= 'count')
    # Development set Position - calculating the percentage of examples per category
    df_pos['percentage'] = df_pos['count']*100 /df_pos['count'].sum()

    # Plot percentage of points per category
    plt.bar(df_pos['index'], df_pos['count'])
    plt.xlabel('Position')
    plt.xticks(rotation = 45)
    plt.ylabel('Number of examples')


    # Development set Position - calculating the number of examples per category
    df_type = df['Type'].value_counts().reset_index(name= 'count')
    # Development set Position - calculating the percentage of examples per category
    df_type['percentage'] = df_pos['count']*100 /df_pos['count'].sum()

    # Plot percentage of points per category
    plt.bar(df_type['index'], df_type['count'])
    plt.xlabel('Type')
    plt.xticks(rotation = 45)
    plt.ylabel('Number of examples')


# ------------------------------------------------------------------------- #
#                               EDA - p-values                              #    
# ------------------------------------------------------------------------- #
 


 
# ------------------------------------------------------------------------- #
#                        EDA - Variable Correlation                         #    
# ------------------------------------------------------------------------- #
def correlation(df):
    '''
        Displays breed information about the dataset
    '''
    df_corr = df.iloc[:,:-4].corr().stack().reset_index()

    # rename the columns
    df_corr.columns = ['f1', 'f2', 'correlation']

    # create a mask to identify rows with duplicate features
    mask_dups = (df_corr[['f1', 'f2']]\
                    .apply(frozenset, axis=1).duplicated()) | \
                    (df_corr['f1']==df_corr['f2']) 

    # apply the mask to clean the correlation dataframe
    df_corr = df_corr[~mask_dups]

    print(df_corr['correlation'].describe())
    df_corr['correlation'].hist()

    df_corr.loc[df_corr['correlation']>0.99, :].sort_values('correlation', ascending = False)

    idx = (df_corr['correlation']<0.99) & (df_corr['correlation']>0.95)
    df_corr.loc[idx, :].sort_values('correlation', ascending = False)


