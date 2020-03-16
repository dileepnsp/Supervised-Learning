import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv(r"C:\Users\cdileepkumar\Documents\02 Python_Practice\Python_Practice\breast-cancer-data.csv")
data.drop(data.filter(regex="Unnamed"),axis=1,inplace=True)
print(data.columns)
print(data.head(5))
#exit(0)
#1. Univariant Analysis on Target column ( Diagnosis) - Generate Pie chart.
sizes=[data.diagnosis[data['diagnosis']=='M'].count(),data.diagnosis[data['diagnosis'] =='B'].count()]
print(sizes)
fig,axs=plt.subplots()
labels='Malignant','Benign'
axs.pie(sizes,labels=labels,autopct='%1.1f%%')
axs.legend(loc='upper right')
plt.show()

#2.Distribute the features and observe the normal distribution.
sns.distplot(data.radius_mean)
#plt.show()

sns.distplot(data.perimeter_mean)
#plt.show()

sns.distplot(data.compactness_mean)
#plt.show()

sns.distplot(data.area_worst)
#plt.show()

sns.distplot(data.fractal_dimension_worst)
#plt.show()

sns.distplot(data.symmetry_worst)
#plt.show()

sns.distplot(data.texture_se)
#plt.show()

sns.distplot(data.symmetry_se)
#plt.show()

#3.Perform Bi-variate analysis on atleast 5-6 features
# Binning Features
data['radious_mean_bins']=pd.cut(data.radius_mean,bins=[0,5,10,15,20,25,30],labels=['0-5','5-10','10-15','15-20','20-25','25-30'])
data['perimeter_mean_bins']=pd.cut(data.perimeter_mean,bins=[0,50,100,150,200],labels=['0-50','50-100','100-150','150-200'])
data['compactness_mean_bins']=pd.cut(data.compactness_mean,bins=[0,0.05,0.1,0.15,0.20,0.25,0.30,0.35],labels=['0-0.05','0.05-0.10','0.10-0.15','0.15-0.20','0.20-0.25','0.25-0.30','0.30-0.35'])
data['area_worst_bins']=pd.cut(data.area_worst,bins=[0,500,1000,1500,2000,2500,3000],labels=['0-500','500-1000','1000-1500','1500-2000','2000-2500','2500-3000'])
data['fractal_dimension_worst_bins']=pd.cut(data.fractal_dimension_worst,bins=[0,0.05,0.1,0.15,0.20,0.25],labels=['0-0.05','0.05-0.10','0.10-0.15','0.15-0.20','0.20-0.25'])
data['symmetry_worst_bins']=pd.cut(data.symmetry_worst,bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7],labels=['0-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.6-0.7'])
data['texture_se_bins']=pd.cut(data.texture_se,bins=[1,2,3,4,5],labels=['1-2','2-3','3-4','4-5'])
data['symmetry_se_bins']=pd.cut(data.symmetry_se,bins=[0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08],labels=['0-0.01','0.01-0.02','0.02-0.03','0.03-0.04','0.04-0.05','0.05-0.06','0.06-0.07','0.07-0.08'])

def bivariate_cat(data,col1,col2,rot):
    cross_tab = pd.crosstab(data[col1], data[col2]).apply(lambda x: x/x.sum() * 100, axis=1).round(2)
    #ct_attr = cross_tab['Yes'].sort_values(ascending=False)
    cross_tab.plot.bar(figsize=(12,5))
    plt.xlabel('{}'.format(col1))
    plt.ylabel('% of Diagnosis'.format(col1))
    plt.title('{} Vs Diagnosis'.format(col1))
    plt.xticks(rotation=rot)
    plt.legend(loc='upper right')
    plt.show()
##bivariate_cat(data,'perimeter_mean','diagnosis',45)
#bivariate_cat(data,'compactness_mean','diagnosis',45)
#bivariate_cat(data,'area_worst','diagnosis',45)
#bivariate_cat(data,'radious_mean_bins','diagnosis',45)
#bivariate_cat(data,'fractal_dimension_worst','diagnosis',45)
#bivariate_cat(data,'symmetry_worst','diagnosis',45)
#bivariate_cat(data,'texture_se','diagnosis',45)
#bivariate_cat(data,'symmetry_se','diagnosis',45)

# Generate Heat map
data.replace({'diagnosis':'M'},1,inplace=True)
data.replace({'diagnosis':'B'},0,inplace=True)

##data.drop(['concave points_worst','perimeter_worst','radius_worst'],axis=1,inplace=True)
#data.drop(['area_se','concave points_mean','area_mean'],axis=1,inplace=True)
#data.drop(['perimeter_se','texture_worst','concavity_worst'],axis=1,inplace=True)
#data.drop(['concavity_se','smoothness_mean','concavity_mean','area_worst','compactness_mean','concavity_mean','perimeter_mean','radius_mean','compactness_worst'],axis=1,inplace=True)

import numpy as np
corr=data.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr,mask=mask,annot=True)
plt.show()
print(data.columns)
#table,result=pd.crosstab(data['radious_mean_bins'],data['diagnosis'],test='chi-square')
#print(table)

# Scaling Features
cols=[c for c in data.columns if c.endswith("_bins")]
data.drop(cols,axis=1,inplace=True)
data.replace({'diagnosis':'B'},0,inplace=True)
data.replace({'diagnosis':'M'},1,inplace=True)
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
data[data.columns]=minmax.fit_transform(data.values)

## Model Building

features=[i for i in data.columns if i !='diagnosis']
x=data[features]
y=pd.DataFrame(data['diagnosis'])

import statsmodels.formula.api as sm
import statsmodels.discrete.discrete_model as sm
model=sm.Logit(y,x)
result=model.fit(method='ncg')
print(result.summary())

#ROC Curve
from sklearn.metrics import roc_curve,auc
x['predict']=result.predict(x)
fpr,tpr,thresholds=roc_curve(y,x['predict'])
roc_auc=auc(fpr,tpr)
print("area under the ROC curve:%f" % roc_auc)

# Optimal Cutoff
i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr': pd.Series(fpr,index=i),'tpr':pd.Series(tpr,index=i),
                    '1-fpr':pd.Series(1-fpr, index=i), 'tf':pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

#Plot tpr vs 1-fpr
fig, ax = plt.subplots(figsize=(10,5))
plt.plot(roc['tpr'])
plt.plot(roc['1-fpr'],color='red')
plt.xlabel('1-False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic',fontsize=15)
ax.set_xticklabels([])
plt.show()

#Logistic Regression using sklearn

x = x.drop("predict", axis=1)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train,X_test,Y_train,Y_test = train_test_split(x ,y,test_size=0.3, stratify=y,random_state=0)
logReg = LogisticRegression()
logReg.fit(X_train, Y_train)
y_pred = logReg.predict(X_test)
score = logReg.score(X_test,Y_test)
print("Accuracy score of model:",score.round(2))
Y_pred_proba = logReg.predict_proba(X_test)
print(Y_pred_proba)

from sklearn.metrics import precision_score, recall_score, precision_recall_curve

pred_proba_df = pd.DataFrame(logReg.predict_proba(X_test))
threshold_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, .7, .75, .8, .85, .9, .95,.99]
precision_lst = []
recall_lst = []
from sklearn.metrics import confusion_matrix

# Confustion Matrix
from sklearn.metrics import confusion_matrix
labels=['M','B']
conf_matrix = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix :",conf_matrix)
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
cax = ax.matshow(conf_matrix)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#Classification Report
from yellowbrick.classifier import ClassificationReport
visualizer = ClassificationReport(logReg, classes=['Attrition','No-Attrition'])
visualizer.fit(X_train, Y_train)
visualizer.score(X_test, Y_test)
g = visualizer.poof()
