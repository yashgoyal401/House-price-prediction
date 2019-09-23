import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(context="notebook", style="darkgrid", palette="deep", font="sans- serif", font_scale=1, color_codes=True)
## Original data sets
House_train = pd.read_csv("C:\\Users\\yashgoyal\\House_train.csv")
House_test = pd.read_csv("C:\\Users\\yashgoyal\\House_test.csv")
#### Output of train data
y_train = House_train.SalePrice
### train data without output
House_train = House_train.drop(["SalePrice"],axis=1)
### Train + Test data
Combined_data = pd.concat([House_train,House_test],axis=0)

Combined_data.columns
Q = Combined_data.isnull().sum()

New = Combined_data.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1)
Z = New.isnull().sum()
New.LotFrontage.unique()
New.describe()

New["LotFrontage"] = New["LotFrontage"].fillna(New["LotFrontage"].mean())
sns.heatmap(New.isnull(),cmap="coolwarm")

New.columns
New.GarageCond.unique()
New["GarageCond"].value_counts()

New["GarageCond"] = New["GarageCond"].fillna(New["GarageCond"].mode()[0])
New['GarageQual'].value_counts()
New['GarageQual'] = New['GarageQual'].fillna(New['GarageQual'].mode()[0])

New['GarageFinish'].value_counts()
New['GarageFinish'] = New['GarageFinish'].fillna(New['GarageFinish'].mode()[0])

New['GarageYrBlt'].value_counts()
New['GarageYrBlt'] = New['GarageYrBlt'].fillna(New['GarageYrBlt'].mean())

New["GarageType"].value_counts()
New["GarageType"] = New["GarageType"].fillna(New["GarageType"].mode()[0])

New['BsmtQual'].value_counts()
New['BsmtQual'] = New['BsmtQual'].fillna(New['BsmtQual'].mode()[0])

New['BsmtCond'].value_counts()
New['BsmtCond'] = New['BsmtCond'].fillna(New['BsmtCond'].mode()[0])

New['BsmtExposure'].value_counts()
New['BsmtExposure'] = New['BsmtExposure'].fillna(New['BsmtExposure'].mode()[0])

New['BsmtFinType1'].value_counts()
New['BsmtFinType1'] = New['BsmtFinType1'].fillna(New['BsmtFinType1'].mode()[0])

New['BsmtFinType2'].value_counts()
New['BsmtFinType2'] = New['BsmtFinType2'].fillna(New['BsmtFinType2'].mode()[0])

New['MasVnrType'].value_counts()
New['MasVnrType'] = New['MasVnrType'].fillna(New['MasVnrType'].mode()[0])

New['MasVnrArea'].value_counts()
New['MasVnrArea'] = New['MasVnrArea'].fillna(New['MasVnrArea'].mode()[0])

New["SaleType"] = New["SaleType"].fillna(New["SaleType"].mode()[0])
New["GarageArea"] = New["GarageArea"].fillna(New["GarageArea"].mean())
New["GarageCars"] = New["GarageCars"].fillna(New["GarageCars"].mode()[0])
New["Functional"] = New["Functional"].fillna(New["Functional"].mode()[0])
New["KitchenQual"] = New["KitchenQual"].fillna(New["KitchenQual"].mode()[0])
New["BsmtFullBath"] =New["BsmtFullBath"].fillna(New["BsmtFullBath"].mode()[0])
New["BsmtHalfBath"] = New["BsmtHalfBath"].fillna(New["BsmtHalfBath"].mode()[0])
New["MSZoning"] = New["MSZoning"].fillna(New["MSZoning"].mode()[0])
New["Utilities"] = New["Utilities"].fillna(New["Utilities"].mode()[0])
New["Exterior1st"] = New["Exterior1st"].fillna(New["Exterior1st"].mode()[0])
New["Exterior2nd"] = New["Exterior2nd"].fillna(New["Exterior2nd"].mode()[0])
New["BsmtFinSF1"] = New["BsmtFinSF1"].fillna(New["BsmtFinSF1"].mean())
New["BsmtFinSF2"] = New["BsmtFinSF2"].fillna(New["BsmtFinSF2"].mean())
New["BsmtUnfSF"] = New["BsmtUnfSF"].fillna(New["BsmtUnfSF"].mean())
New["TotalBsmtSF"] = New["TotalBsmtSF"].fillna(New["TotalBsmtSF"].mean())
New["Electrical"] = New["Electrical"].fillna(New["Electrical"].mode()[0])

New_dummies = pd.get_dummies(New,drop_first=True)
final_train = New_dummies.loc[:1459]
final_test = New_dummies.loc[1459:]
final_test = final_test.drop(final_test.index[0])

X_train = final_train
X_test = final_test
y_train
sns.distplot(y_train)
log = np.log(y_train)
sns.distplot(log)
plt.hist(y_train)

import xgboost
from sklearn.metrics import accuracy_score
predictor = xgboost.XGBRegressor()
predictor.fit(X_train,log)
pred_train = predictor.predict(X_train)
pred_test =predictor.predict(X_test)
pred_test_original = np.exp(pred_test)


PP = pd.concat([X_test.Id],axis=1)
PP["SalePrice"] = pred_test_original
PP.head()

PP.to_csv("HouselogXgboost.csv",index=False)

d = New.describe(include = 'all')
N = House_train.corr()

from sklearn.model_selection import RandomizedSearchCV

params = {'nthread':[4],'learning_rate': [.03, 0.05, .07],'max_depth': [5, 6, 7],'min_child_weight': [4],'silent': [1],'subsample': [0.7],'colsample_bytree': [0.7],'n_estimators': [500]}

predictor = xgboost.XGBRegressor()
RandomS = RandomizedSearchCV(predictor,param_distributions=params,n_jobs=-1,cv=5,verbose=3)
RandomS.fit(X_train,log)
RandomS.best_estimator_

XGB = xgboost.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,colsample_bynode=1, colsample_bytree=0.7, gamma=0,importance_type='gain', learning_rate=0.05, max_delta_step=0,max_depth=5, min_child_weight=4, missing=None, n_estimators=500,n_jobs=1, nthread=4, objective='reg:linear', random_state=0,reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None, silent=1,subsample=0.7, verbosity=1)
XGB.fit(X_train,log)
Xpred_train = XGB.predict(X_train)
pred1 = XGB.predict(X_test)
pred1_original = np.exp(pred1)
PP = pd.concat([X_test.Id],axis=1)
PP["SalePrice"] = pred1_original
PP.head()

PP.to_csv("Houselog1Xgboost.csv",index=False)

# Features selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func = f_classif, k='all')
fit = bestfeatures.fit(X_train,y_train)
bestfeatures1 = SelectKBest(score_func = chi2, k='all')
fit1 = bestfeatures1.fit(X_train,y_train)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X_train.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']

dfscores1 = pd.DataFrame(fit1.scores_)
dfcolumns1 = pd.DataFrame(X_train.columns)
featureScores1 = pd.concat([dfcolumns1,dfscores1],axis=1)
featureScores1.columns = ['Specs','Score']
### SKlearn method

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,log)
lm.coef_
lm.score(X_train,log) ###0.94
lm_pred_test = lm.predict(X_test)
lm_pred_test_original = np.exp(lm_pred_test)

PP = pd.concat([X_test.Id],axis=1)
PP["SalePrice"] = lm_pred_test_original
PP.head()

PP.to_csv("Houselog1sklearn.csv",index=False)
####  Ridge technique 
from sklearn.linear_model import Ridge
L1 = Ridge(alpha=1.0)
L1.fit(X_train,log)
L1.score(X_train,log)  ### 0.9280

L1_pred = L1.predict(X_test)
L1_pred_original = np.exp(L1_pred)

PP = pd.concat([X_test.Id],axis=1)
PP["SalePrice"] = L1_pred_original
PP.head()

PP.to_csv("Houselog1Ridge.csv",index=False)


###  Lasso technique
from sklearn import linear_model
L2 = linear_model.Lasso(alpha=0.1)
L2.fit(X_train,log)
L2.score(X_train,log)  ### 0.7970


###  ElasticNet
from sklearn.linear_model import ElasticNet
L3 = ElasticNet(random_state=500)
L3.fit(X_train,log)
L3.score(X_train, log)   ## 0.7873

## ADboost
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostRegressor
predictor = AdaBoostRegressor()
params={'n_estimators':[500,1000,2000],'learning_rate':[.001,0.01,.1],'random_state':[1]}
RandomS = RandomizedSearchCV(predictor,param_distributions=params,n_jobs=-1,cv=5,verbose=3)
RandomS.fit(X_train,log)
RandomS.best_estimator_

ada = AdaBoostRegressor(base_estimator=None, learning_rate=0.01, loss='linear', n_estimators=2000, random_state=1)
ada.fit(X_train,log)
pr = ada.predict(X_train)
preddd  = ada.predict(X_test)

preddd_original = np.exp(preddd)
PP = pd.concat([X_test.Id],axis=1)
PP["SalePrice"] = preddd_original
PP.head()

PP.to_csv("Houselog1adaboost.csv",index=False)


