# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 19:45:07 2021

@author: barsuraj1
"""


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd
import math 

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Modelling Helpers
import sklearn
from sklearn.preprocessing import Normalizer , scale
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns


df = pd.read_csv(r'C:\Users\barsuraj1\Desktop\Sales Prediction\rossman\rossmann.csv')
new = df.copy()
store = pd.read_csv(r'C:\Users\barsuraj1\Desktop\Sales Prediction\rossman\store.csv')
test = pd.read_csv(r'C:\Users\barsuraj1\Desktop\Sales Prediction\rossman\test.csv')
df.head()
df.nunique()
df.dtypes
df.info()
df.shape
df.describe()
print("date ranges from", df.Date.min(),"to", df.Date.max())
print("approximately 2.5 years")

li = ["DayOfWeek" , "StateHoliday" , "SchoolHoliday"]

for i in li:
  print(i)
  print(df[i].unique())
  print("-----------------------")
  

### a = public holiday, b = Easter holiday, c = Christmas, 0 = None
##Indicates if the (Store, Date) was affected by the closure of public schools

df.Store.nunique()
df.nunique()
df.info()

test.info()
store.info()

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(20,4))
sns.countplot(x='Open',hue='DayOfWeek', data=df, ax=axis1)
sns.pointplot(x='Open',hue='DayOfWeek', data=df, ax=axis2)
#The store is mainly closed on day 7 which is Sunday


def extract1(x):
    return int(str(x)[:4])

def extract2(x):
    return int(str(x)[5:7])

def extract3(x):
    return (str(x)[:7])


df['Date'] = df['Date'].apply(extract3)
test['Date'] = test['Date'].apply(extract3)
df['Year']  = df['Date'].apply(extract1)
df['Month'] = df['Date'].apply(extract2)
test['Year']  = test['Date'].apply(extract1)
test['Month'] = test['Date'].apply(extract2)
avgsales    = df.groupby('Date')["Sales"].mean()
percentchngsales = df.groupby('Date')["Sales"].sum().pct_change()

#Separating (year + month) in the Date attribute AND Year and Month also in separate column
#Extracting month and year feature from the date
    

sns.factorplot(x="Date" ,y = "Sales" , data=df, kind="point", aspect=2,size=12)

#Heat-Map to show correlation b/w numerical attributes
correlation_map = df[df.columns].corr()
obj = np.array(correlation_map)
obj[np.tril_indices_from(obj)] = False
fig,ax= plt.subplots()
fig.set_size_inches(9,9)
sns.heatmap(correlation_map, mask=obj,vmax=.7, square=True,annot=True)


def plotter(StringA , StringB):
  fig, axes = plt.subplots(2,1)
  fig.set_size_inches(15, 10)
  sns.barplot(x=StringA, y=StringB, data=df ,hue="DayOfWeek", ax = axes[0])
  sns.boxplot(x=StringA, y=StringB, data=df ,hue="DayOfWeek", ax=axes[1])

plotter("Year" , "Sales")
plotter("Year" , "Customers")

# Encoding Stateholiday similarly

df["StateHoliday"] = df["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})
test["StateHoliday"] = test["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})
df.StateHoliday.value_counts()

labels = 'Not-Affected' , 'Affected'
sizes = df.SchoolHoliday.value_counts()
colors = ['gold', 'silver']
explode = (0.1, 0.0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=180)
plt.axis('equal')
plt.title("Sales Affected by Schoolholiday or Not ?")
plt.plot()
fig=plt.gcf()
fig.set_size_inches(6,6)
plt.show()

df.SchoolHoliday.value_counts()

df["Sales"].plot(kind='hist',bins=100,xlim=(0,15000))
## 0 is raised because most of the times store was closed

store.head()

temp = []
for i in df.groupby('Store')[["Sales", "Customers"]].mean()["Sales"]:
  temp.append(i)
store["Sales"] = temp

temp = []
for i in df.groupby('Store')[["Sales", "Customers"]].mean()["Customers"]:
  temp.append(i)
store["Customers"] = temp

store.head()


labels = 'a' , 'b' , 'c' , 'd'
sizes = store.StoreType.value_counts()
colors = ['orange', 'green' , 'red' , 'pink']
explode = (0.1, 0.0 , 0.15 , 0.0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=180)
plt.axis('equal')
plt.title("Distribution of different StoreTypes")
plt.plot()
fig=plt.gcf()
fig.set_size_inches(6,6)
plt.show()


def plotmaster(StringA , StringB):
  fig, axes = plt.subplots(3,1)
  fig.set_size_inches(12, 15)
  sns.barplot(x=StringA, y=StringB, data=store ,hue="StoreType", ax = axes[0])
  sns.boxplot(x=StringA, y=StringB, data=store ,hue="StoreType", ax=axes[1])
  sns.violinplot(x=StringA, y=StringB, data=store, hue="StoreType" , ax=axes[2])

plotmaster("Assortment" , "Sales")

plotmaster("Promo2" , "Sales")

store.isnull().sum()

## stores which are opened on Sundays
df[(df.Open == 1) & (df.DayOfWeek == 7)]['Store'].unique()

df = new.copy()

test.fillna(1, inplace=True)
df = df[df["Open"] != 0]
df = df[df["Sales"] > 0]
df['log_sales'] = np.log(df['Sales'])
df = pd.merge(df, store, on='Store')
test = pd.merge(test, store, on='Store')
df.fillna(0,inplace=True)
test.fillna(0,inplace=True)

new.head().Date

df["year"]=df.Date.apply(extract1)
df["month"]=df.Date.apply(extract2)

df["Day"]=df.Date.apply(lambda x: int(str(x)[8:10])) 

test["year"]=test.Date.apply(extract1)
test["month"]=test.Date.apply(extract2)
test["Day"]=test.Date.apply(lambda x: int(str(x)[8:10]))

#Now getting dummies

df = pd.get_dummies(df,columns=['StoreType','Assortment','year'])
test = pd.get_dummies(test,columns=['StoreType','Assortment','year'])
test['year_2013']=0
test['year_2014']=0

df.columns


X = df.drop(['Sales_x','log_sales','Store','Date','Customers_x','CompetitionOpenSinceYear','Promo2SinceYear','PromoInterval'] , axis = 1)
y = df['log_sales']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=7)
X_test = test.drop(['Id','Store','Date' ,'CompetitionOpenSinceYear','Promo2SinceYear','PromoInterval'] , axis = 1)

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X_train , y_train)
pred = reg.predict(X_val)
 
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_val,pred))
print(rmse)