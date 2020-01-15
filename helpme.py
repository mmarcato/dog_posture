from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import data_prep


from sklearn.model_selection import cross_validate


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names, dtype=None):
        self.attribute_names = attribute_names
        self.dtype = dtype
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_selected = X[self.attribute_names]
        if self.dtype:
            return X_selected.astype(self.dtype).values
        return X_selected.values
 
 
feat = ['mean','std', 'median', 'min', 'max']
LR = Pipeline([
        ('selector', DataFrameSelector(feat,'float64')),
        ('scaler', StandardScaler()),
        ('estimator', LogisticRegression() )       
        ])
RF = Pipeline([
            ('selector', DataFrameSelector(feat,'float64')),
            ('scaler', StandardScaler()),
            ('estimator', RandomForestClassifier() )       
            ]) 
SV = Pipeline([
            ('selector', DataFrameSelector(feat,'float64')),
            ('scaler', StandardScaler()),
            ('estimator', LinearSVC() )       
            ])
cv = StratifiedKFold(n_splits = 10)

df_pos_bal = data_prep.balance_df(df_feat, 'Position')
X = df_pos_bal.loc[:, feat]
y = df_pos_bal.loc[:, 'Position'].values
print( df_pos_bal['Position'].value_counts(), '\n')
print(cross_validate(SV, X, y, cv= cv, return_train_score=True ))