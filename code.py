import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df=pd.read_csv('/content/drive/MyDrive/Weather Pridiction/seattle-weather.csv')
df.head()
df.isna().any()
sns.pairplot(data=df,hue='weather')
sns.countplot(data=df, x='weather')

df.info()
df.describe()
fig,axes = plt.subplots(2,2, figsize=(10,10))
cols = ['precipitation', 'temp_max', 'temp_min', 'wind']
for i in range(4):
    sns.scatterplot(data=df, x='date', y=cols[i], hue='weather', ax=axes[i%2,i//2])

fig.clear()
fig,axes = plt.subplots(2,2, figsize=(10,10))
for i in range(4):
    sns.histplot(kde=True, data=df, x=cols[i], hue='weather', ax= axes[i%2, i//2])

fig,axes = plt.subplots(2,2, figsize=(10,10))
cols = ['precipitation', 'temp_max', 'temp_min', 'wind']
for i in range(4):
    sns.boxplot(x='weather', y=cols[i], data=df, ax=axes[i%2,i//2])
fig, axes = plt.subplots(2,2, figsize=(12,6))
for i in range(len(cols)):
    sns.scatterplot(data=df.pivot(index='date',columns='weather', values=cols[i]).fillna(0), x='drizzle', y='rain', ax=axes[i%2,i//2])
    axes[i%2,i//2].set_title(cols[i])
    axes[i%2,i//2].set_xticks([])
plt.show()

countrain=len(df[df.weather=="rain"])
countsun=len(df[df.weather=="sun"])
countdrizzle=len(df[df.weather=="drizzle"])
countsnow=len(df[df.weather=="snow"])
countfog=len(df[df.weather=="fog"])
print("Percent of Rain:{:2f}%".format((countrain/(len(df.weather))*100)))
print("Percent of Sun:{:2f}%".format((countsun/(len(df.weather))*100)))
print("Percent of Drizzle:{:2f}%".format((countdrizzle/(len(df.weather))*100)))
print("Percent of Snow:{:2f}%".format((countsnow/(len(df.weather))*100)))
print("Percent of Fog:{:2f}%".format((countfog/(len(df.weather))*100)))

import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import re
import missingno as mso
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

lc=LabelEncoder()# Scaling the weather variables using label Encoder:
df["weather"]=lc.fit_transform(df["weather"])

df=df.drop(["date"],axis=1) #preprocessing
df.head()

x=((df.loc[:,df.columns!="weather"]).astype(int)).values[:,0:]
y=df["weather"].values
df.weather.unique()

knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
print("KNN Accuracy:{:.2f}%".format(knn.score(x_test,y_test)*100))

svm=SVC()
svm.fit(x_train,y_train)
print("SVM Accuracy:{:.2f}%".format(svm.score(x_test,y_test)*100))

gbc=GradientBoostingClassifier()
gbc.fit(x_train,y_train)
print("Gradient Boosting Accuracy:{:.2f}%".format(gbc.score(x_test,y_test)*100))

import warnings
warnings.filterwarnings('ignore')
xgb=XGBClassifier()
xgb.fit(x_train,y_train)
print("XGB Accuracy:{:.2f}%".format(xgb.score(x_test,y_test)*100))

input=[[1.140175,8.9,2.8,2.469818]]
ot=xgb.predict(input)
print("The weather is:")
if(ot==0):
    print("Drizzle")
elif(ot==1):
    print("Fog")
elif(ot==2):
    print("Rain")
elif(ot==3):
    print("snow")
else:
    print("Sun")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

lr = LogisticRegression()
rf = RandomForestClassifier(bootstrap=False)
gbc = GradientBoostingClassifier()
dt = DecisionTreeClassifier()
svc = SVC()
knn= KNeighborsClassifier()

lr.fit(x_train,y_train)
rf.fit(x_train,y_train)
gbc.fit(x_train,y_train)
dt.fit(x_train,y_train)
svc.fit(x_train, y_train)
knn.fit(x_train, y_train)


y_pred_lr = lr.predict(x_test)
y_pred_rf = rf.predict(x_test)
y_pred_gbc = gbc.predict(x_test)
y_pred_dt = dt.predict(x_test)
y_pred_svc = svc.predict(x_test)
y_pred_knn = knn.predict(x_test)
print('LogReg Accuracy = {:.2f}'.format(lr.score(x_test,y_test)*100))
print('RandFor Accuracy = {:.2f}'.format(rf.score(x_test,y_test)*100))
print('GBC Accuracy = {:.2f}'.format(gbc.score(x_test,y_test)*100))
print('DT Accuracy = {:.2f}'.format(dt.score(x_test,y_test)*100))
print('SVC Accuracy = {:.2f}'.format(svc.score(x_test,y_test)*100))
print('KNN Accuracy = {:.2f}'.format(knn.score(x_test,y_test)*100))


from sklearn.metrics import confusion_matrix

print('LogReg\n',confusion_matrix(y_pred_lr,y_test))
print('RandFor\n', confusion_matrix(y_pred_rf,y_test))
print('GBC\n', confusion_matrix(y_pred_gbc, y_test))

from sklearn.metrics import classification_report

print('LogReg\n',classification_report(y_test,y_pred_lr, zero_division=0))
print('GBC\n',classification_report(y_test,y_pred_gbc, zero_division=0))
print('RF\n',classification_report(y_test,y_pred_rf, zero_division=0))
print('DT\n',classification_report(y_test,y_pred_dt, zero_division=0))
print('KNN\n',classification_report(y_test,y_pred_knn, zero_division=0))

# Model training
gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)

# Model evaluation
accuracy = gbc.score(x_test, y_test)
print("Gradient Boosting Accuracy: {:.2f}%".format(accuracy * 100))
