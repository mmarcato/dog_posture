
# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
#                   Machine Learning - Label 'Type'                         #    
# ------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------- #

#                          Evaluating Dataset                               #
df_feat.loc[:, 'Type'].value_counts()

df_bal_type = balance_df(df_feat, 'Type')
df_bal_type['Type'].value_counts()

x_bal_type = df_bal_type.loc[:, feat]
y_bal_type = df_bal_type.loc[:, 'Type']



pipe_performance = []
print ('LR, RF, SV classifier for Type Positions')
for pipe in [LR, RF, SV]:
    print(np.mean(cross_val_score(pipe, x_bal_type , y_bal_type , scoring="accuracy", cv=kf)))
    score = cross_validate(pipe, x_bal_type, y_bal_type, cv=kf, return_train_score=True)
    
    pipe_performance.append([100*np.mean(score['test_score']), 100*np.mean(score['train_score']), 
                                 np.mean(score['fit_time']), np.mean(score['score_time'])])
print(pipe_performance)    

#                          Logistic Regression                              #
# We can see that the model is underfitting, so no point in using regularization or PCA
LR = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('estimator', LogisticRegression() )       
        ])
Ridge = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('estimator', RidgeClassifier() )       
        ]) 
    
    
# Evaluating pipelines
pipes = [LR, Ridge]
X_b = [x_bal_type] * 2
y_b = [y_bal_type] * 2
title = 'Difference between LogReg, Ridge'
label = ['LogReg', 'Ridge']
LR = evaluate_pipeline(pipes, X_b, y_b, kf, title, label)
  
#                     Linear Support Vector Classifier                      #
# Creating pipeline

SV = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('estimator', LinearSVC() )       
        ])
LS2 = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components = 120)),
        ('estimator', LinearSVC() )       
        ]) 
LS3 = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components = 100)),
        ('estimator', LinearSVC() )       
        ])
LS4 = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components = 50)),
        ('estimator', LinearSVC() )       
        ]) 
    
pipes = [SV]
title = 'Difference between no PCA, PCA (120), PCA(100), PCA (50) using Linear SCV'
label = ['no PCA', 'PCA - 120 comp',  'PCA - 100 comp', 'PCA - 50 comp']
LS = evaluate_pipeline(pipes, X_b, y_b, kf, title, label)         
    
#                         Random Forest Classifier                          #
RF = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('estimator', RandomForestClassifier() )       
        ]) 
RF2 = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components = 120)),
        ('estimator', RandomForestClassifier() )       
        ])    
RF3 = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components = 100)),
        ('estimator', RandomForestClassifier() )       
        ]) 
RF4 = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components = 50)),
        ('estimator', RandomForestClassifier() )       
        ]) 

pipes = [RF]
title = 'Random Forest - No PCA, PCA (120), PCA(100), PCA (50) '
label = ['No PCA', 'PCA - 120 comp',  'PCA - 100 comp', 'PCA - 50 comp']
RF_Results = evaluate_pipeline(pipes, X_b, y_b, kf, title, label)
      

   
# ------------------------------------------------------------------------- #
# Machine Learning - Tutorial https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60                          #    
# ------------------------------------------------------------------------- # 

#       Splitting DF into Test and Train before starting                   #

df_feat['Type'].value_counts()

X = df_feat.loc[:, feat].values
Y = df_feat.loc[:, 'Type'].values

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size= 0.2, stratify = Y, random_state=0)

# Build new data frame with Scaling and PCA on X_train
scaler = StandardScaler()

scaler.fit(X_train)      
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

pca_50 = PCA(n_components = 50)
pca_50.fit(X_train)                

X_train = pca_50.transform(X_train)     
X_test = pca_50.transform(X_test)  

LogReg = LogisticRegression()

LogReg.fit(X_train, Y_train)
LogReg.score(X_test, Y_test)

explained_variance = pca.explained_variance_ratio_

RF = RandomForestClassifier(max_depth = 5, random_state = 0)
RF.fit(X_train, Y_train)               
RF.score(X_test, Y_test)

# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
#                   Machine Learning - Label 'Position'                     #    
# ------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------- #

# ------------------------------------------------------------------------- #
#                     Classifier for 'Static' Positions                     #    
# ------------------------------------------------------------------------- # 
df_static = df_feat[ df_feat['Type'] == 'Static' ]
df_static['Position'].value_counts()
# Balancing dataframe
df_static_bal = balance_df(df_static, 'Position')
df_static_bal['Position'].value_counts()

X_static_bal = df_static_bal.loc[:, feat]
Y_static_bal = df_static_bal.loc[:, 'Position']

pipe_performance = []
print ('LR, RF, SV classifier for Static Positions')
for pipe in [LR, RF, SV]:
    print(np.mean(cross_val_score(pipe, X_static_bal, Y_static_bal, scoring="accuracy", cv=kf)))
    score = cross_validate(pipe, X_static_bal, Y_static_bal, cv=kf, return_train_score=True)
                                                
    pipe_performance.append([100*np.mean(score['test_score']), 100*np.mean(score['train_score']), 
                                 np.mean(score['fit_time']), np.mean(score['score_time'])])
 
    #print(confusion_matrix( Y_static_bal, (cross_val_predict(pipe, X_static_bal, Y_static_bal, cv= kf))))

print('test_score', 'train_score','fit_time', 'score_time')
print(pipe_performance)


    
# ------------------------------------------------------------------------- #
#                    Classifier for 'Dynamic' Positions                     #    
# ------------------------------------------------------------------------- # 
df_dynamic = df_feat[ df_feat['Type'] == 'Dynamic' ]
df_dynamic['Position'].value_counts()

df_dynamic_bal = balance_df(df_dynamic, 'Position')
df_dynamic_bal['Position'].value_counts()

X_dynamic_bal = df_dynamic_bal.loc[:, feat]
Y_dynamic_bal = df_dynamic_bal.loc[:, 'Position']


pipe_performance = []
print ('LR, RF, SV classifier for Dynamic Positions')
for pipe in [LR, RF, SV]:
    print(np.mean(cross_val_score(pipe, X_dynamic_bal, Y_dynamic_bal, scoring="accuracy", cv=kf)))
    score = cross_validate(pipe, X_dynamic_bal, Y_dynamic_bal, cv=kf, return_train_score=True)
    
    pipe_performance.append([100*np.mean(score['test_score']), 100*np.mean(score['train_score']), 
                                 np.mean(score['fit_time']), np.mean(score['score_time'])])
print(pipe_performance)    
    
