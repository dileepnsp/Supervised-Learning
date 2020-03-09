import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import statsmodels.api as sm
#Load dataset
data=pd.read_csv(r"C:\Users\cdileepkumar\Documents\02 Python_Practice\Python_Practice\boston_house_price.csv")
# Split features and Target
features=data[data.columns[:-1]]
target=pd.DataFrame(data[data.columns[-1]],columns=[data.columns[-1]])
# Dist Plot the Target variable
sns.distplot(target)
plt.show()
## Target values are normally distributed.
# Check nulls
data.isnull().sum()
#heat map generate
corr=data.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr,mask=mask,annot=True)
plt.show()
#RM and LSTAT are highly correlated to target. Hence need to preseve them alone.Rest of the features are ignored
features=features[['RM','LSTAT']]
## Run the VIF
#vif=pd.DataFrame()
#vif['VIF Factor']=[ variance_inflation_factor(features.values,i) for i in range(features.shape[1])]
#vif['features']=features.columns
#print(vif)

#Plot the RM v/s Target and LSTAT v/s Target
fig,(ax1,ax2)=plt.subplots(1,2)
ax1.scatter(features['RM'],target)
ax1.set_xlabel("RM")
ax1.set_ylabel("MEDV")
ax2.scatter(features['LSTAT'],target)
ax2.set_xlabel("LSTAT")
ax2.set_ylabel("MEDV")
plt.show()

# Coefficient of correlation
print("Correlation between RM and target:"+str(features['RM'].corr(target['MEDV'])))
print("Correlation between LSTAT and target:"+str(features['LSTAT'].corr(target['MEDV'])))

# Linear Regression using stats model.
X_stat = sm.add_constant(features)
stats_model = sm.OLS(target,X_stat).fit()
print(stats_model.summary())

# Split test and Train data sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(features[['RM','LSTAT']],target['MEDV'],test_size=0.2)
#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)

#Linear Regression using Linear Model
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
pred=lr.predict(x_test)

#Calculate RMSE,r2_score
from sklearn.metrics import mean_squared_error,r2_score
lr_rmse=np.sqrt(mean_squared_error(y_test,pred))
print('Linear_regression model using sklearn..')
print("linear regression RMSE:"+str(lr_rmse))
lr_r2=r2_score(y_test,pred)
lr_coef=lr.coef_
lr_intercept=lr.intercept_
print("linear regression R2_score:"+str(lr_r2))
print("linear regression Coefficient:"+str(lr_coef))
print("linear regression Intercept:"+str(lr_intercept))

# Ridge Regression alpha=0.3
from sklearn.linear_model import Ridge
ridge_reg=Ridge(alpha=3)
ridge_reg.fit(x_train,y_train)
ridge_pred=ridge_reg.predict(x_test)
ridge_rmse=np.sqrt(np.mean((y_test-ridge_pred)** 2))
ridge_score=ridge_reg.score(x_test,y_test)
ridge_coef=ridge_reg.coef_
ridge_intercept=ridge_reg.intercept_
print("ridge rmse:"+str(ridge_rmse))
print("ridge score:"+str(ridge_score))
print("ridge coefficients:"+str(ridge_coef))
print("ridge intercepts:"+str(ridge_intercept))

# Ridge Model coefficents.
import matplotlib.pyplot as plt
rdge_coef = pd.Series(ridge_reg.coef_,x_train.columns).sort_values()
rdge_coef.plot(kind='bar',fontsize=15)
plt.title("Ridge Model Coefficients",fontsize=15)
plt.xticks(rotation=55)
#plt.show()

# Lasso Regression alpha=3
from sklearn.linear_model import Lasso
lasso_reg=Lasso(alpha=3)
lasso_reg.fit(x_train,y_train)
lasso_pred=lasso_reg.predict(x_test)
lasso_rmse=np.sqrt(np.mean((y_test-lasso_pred)** 2))
lasso_score=lasso_reg.score(x_test,y_test)
lasso_coef=lasso_reg.coef_
lasso_intercept=lasso_reg.intercept_

print("Lasso rmse:"+str(lasso_rmse))
print("Lasso score:"+str(lasso_score))
print("Lasso coefficients:"+str(lasso_coef))
print("Lasso intercepts:"+str(lasso_intercept))

# Bar chart
#labels = ['LR', 'RIDGE', 'LASSO']
labels = ['RMSE','R2_SCORE','INTERCEPT']
lr_group=[lr_rmse,lr_r2,lr_intercept]
ridge_group=[ridge_rmse,ridge_score,ridge_intercept]
lasso_group=[lasso_rmse,lasso_score,lasso_intercept]

print(lr_group)
print(ridge_group)
print(lasso_group)

#data = [lr_group,ridge_group,lasso_group]
#X = np.arange(3)
#fig = plt.figure()
#ax = fig.add_axes([0,0,1,1])
#ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
#ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
#ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)
#ax.set_ylabel('Metrics')
#ax.set_title('Metrics v/s regression')
#ax.set_xticks(labels)
#ax.set_xticklabels(labels)
#ax.legend()
#plt.show()
####
