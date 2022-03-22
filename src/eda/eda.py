
# ------------------------------------------------------------------------- #
#                      Exploratory Data Analysis                            #    
# ------------------------------------------------------------------------- # 

# breeds
analyse.breed(df_feat)
analyse.breed(df_test)
analyse.breed(df_dev)



# ------------------------------------------------------------------------- #
#                        Exploratory Data Analysis                          #    
# ------------------------------------------------------------------------- #

 
# --------------------------- EDA - Dog Imbalance ------------------------- #   

# Development set Position - calculating the number of examples per category
df_dog = df['Dog'].unique()
 

# -----------------------   EDA - Class Imbalance  ------------------------ # 

### Position
# class balance
print(df['Position'].value_counts())

# Development set Position - calculating the number of examples per category
df_pos = df['Position'].value_counts().reset_index(name= 'count')
# Development set Position - calculating the percentage of examples per category
df_pos['percentage'] = df_pos['count']*100 /df_pos['count'].sum()

# Plot percentage of points per category
plt.bar(df_pos['index'], df_pos['count'])
plt.xlabel('Position')
plt.xticks(rotation = 45)
plt.ylabel('Number of examples')


### Type
# class balance
print(df['Type'].value_counts())

# Development set - calculating the number of examples per category
df_type = df['Type'].value_counts().reset_index(name= 'count')
# Development set - calculating the percentage of examples per category
df_type['percentage'] = df_pos['count']*100 /df_pos['count'].sum()

# Plot percentage of points per category
plt.bar(df_type['index'], df_type['count'])
plt.xlabel('Type')
plt.xticks(rotation = 45)
plt.ylabel('Number of examples') 
            