

# ------------------------------------------------------------------------- #
#                   Load and Evaluate Grid Search results                   #
# ------------------------------------------------------------------------- #

# Loading Grid Search Results from Pickle file
run = 'RF-Test3'
gs = joblib.load('{}/{}.pkl'.format(dir_model, run))
evaluate.gs_output(gs)

# Calculate mean test score value while maintaining one parameter constant at a time
df_cv = pd.DataFrame(gs.cv_results_)
print(df_cv.groupby(['param_estimator__max_depth'])['mean_test_score'].mean())
print(df_cv.groupby(['param_estimator__n_estimators'])['mean_test_score'].mean())
print(df_cv.groupby(['param_estimator__max_features'])['mean_test_score'].mean())

print(df_cv.groupby(['param_estimator__max_depth'])['mean_train_score'].mean())
print(df_cv.groupby(['param_estimator__n_estimators'])['mean_train_score'].mean())
print(df_cv.groupby(['param_estimator__max_features'])['mean_train_score'].mean())

for depth in df_cv['param_estimator__max_depth'].unique():
    print(depth)
    df = df_cv.loc[df_cv['param_estimator__max_depth']== depth]
    sns.catplot(data=df, kind="bar",
        x="param_estimator__n_estimators", y="mean_test_score", hue="param_estimator__max_features",
        ci="sd", palette="dark", alpha=.6, height=6 )



# Evaluate Random Forest feature importance
df_ft = pd.DataFrame({'Feature': gs_rf.best_estimator_['selector'].attribute_names, 
        'Importance' : gs_rf.best_estimator_['slt'].scores_})
df_ft.sort_values(by = 'Importance', ascending = False, inplace = True, ignore_index = True)

# Plotting feature importance
slt_ft = gs_rf.best_params_['slt__k']
plt.figure(figsize= (20, 8))
plt.bar(df_ft.loc[:slt_ft,'Feature'],df_ft.loc[:slt_ft, 'Importance'])
plt.xticks(rotation = 45)
plt.title('Best {} Features and their importances using SKB'.format(slt_ft))
plt.xlabel('Features')
plt.ylabel('Importance')
plt.savefig('{}/results/SKB-BestEstimator_BestFeatures'.format(dir_base))
# important features from the best 
rf_ft = list(df_ft.loc[:14, 'Feature'])


