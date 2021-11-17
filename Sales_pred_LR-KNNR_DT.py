# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 20:07:18 2021

@author: barsuraj1
"""

#For Ignoring Warning
import warnings
warnings.filterwarnings("ignore")

## Handle table-like data and matrices
import numpy as np
import pandas as pd
import math

#visualisation
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
sns.set_style("whitegrid")

#modelling our algorithm
import sklearn

store=pd.read_csv("store.csv")
train=pd.read_csv("rossmann.csv",parse_dates = True, index_col = 'Date')

## Tabular sample representation of store dataset of first and last 5 rows

store.head().append(store.tail())

train.head().append(train.tail())

#### Step 2- Handling Data Description
## Dealing with missing Values


#Count sum of missing values in store dataset
print("Store Dataset:\n\n",store.isnull().sum())


#Count missing values in train dataset
print("Train Dataset:\n\n",train.isnull().sum())


#we can see that some features have a high percentage of missing values and they won't be accurate as indicators, 
#so we will remove features with more than 30% missing values.
store = store.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear', 'PromoInterval'], axis=1)

#### Visualising the Distribution of data

# CompetitionDistance is distance in meters to the nearest competitor store
# let's first have a look at its distribution
sns.distplot(store.CompetitionDistance.dropna())
plt.title("Distributin of Store Competition Distance")
## As we can see from the following graph plot that the distribution has its tail on the right side, so it is a Right Skewed Distribution. so we'll replace missing values with the median.

# replace missing values in CompetitionDistance with median for the store dataset
store.CompetitionDistance.fillna(store.CompetitionDistance.median(), inplace=True)
### Extracting the Days, Month, day and week of year from "Date"

train["Year"]=train.index.year
train["Month"]=train.index.month
train["Day"]=train.index.day
train["WeekOfYear"]=train.index.week
train = train.reset_index()
train

### Merging The tables
df = pd.merge(train, store, how='left', on='Store')
df.head()

len(df)

### Drop subsets of data which might cause Bias
## Bias- it is the simplifying assumptions made by the model to make the target function easier to approximate and thus make the end prediction biased.


# where stores are closed, they won't generate sales, so we will remove this part of the dataset
df = df[df.Open != 0]

# Open isn't a variable anymore, so we'll drop it
df = df.drop('Open', axis=1)

# see if there's any opened store with zero sales
print("Number of stores with zero sales:",df[df.Sales == 0].shape[0])

# remove this part of data to avoid bias
df = df[df.Sales != 0]

df.head()

### Handling Categorical Data
df.info()

# see what's in nominal varibles 
set(df.StateHoliday), set(df.StoreType), set(df.Assortment)


#StateHoliday indicates a state holiday where
#a = public holiday,
#b = Easter holiday,
#c = Christmas,
#0 = None
#convert number 0 to string 0

df.loc[df.StateHoliday == 0,'StateHoliday'] = df.loc[df.StateHoliday == 0,'StateHoliday'].astype(str)
set(df.StateHoliday)
# 0 - not a state holiday; 1- is on a state holiday
df['StateHoliday'] = df.StateHoliday.map({'0':0, 'a':1 ,'b' : 1,'c': 1})
set(df.StateHoliday)

## Step-3: Adding Additional Features

#Making a Copy of the existing dataset
df1 = df.copy()

# calculate weekly average sales
sales = df1[['Year','Month','Store','Sales']].groupby(['Year','Month','Store']).mean()
sales = sales.rename(columns={'Sales':'AvgSales'})
sales = sales.reset_index()


sales.head()
df1['sales_key']=df1['Year'].map(str) + df1['Month'].map(str) + df1['Store'].map(str)
sales['sales_key']=sales['Year'].map(str) + sales['Month'].map(str) + sales['Store'].map(str)

# drop extra columns
sales = sales.drop(['Year','Month','Store'], axis=1)
# merge
df1 = pd.merge(df1, sales, how='left', on=('sales_key'))

#create a variable that calculates Monthly average number of customers for each store.
cust = df1[['Year','Month','Store','Customers']].groupby(['Year','Month', 'Store']).mean()
cust = cust.rename(columns={'Customers':'AvgCustomer'})
cust = cust.reset_index()

df1['cust_key']=df1['Year'].map(str) + df1['Month'].map(str) + df1['Store'].map(str)
cust['cust_key']=cust['Year'].map(str) + cust['Month'].map(str) + cust['Store'].map(str)


# drop original feature Customers
df1 = df1.drop('Customers', axis=1)

# drop extra columns
cust = cust.drop(['Year', 'Month', 'Store'], axis=1)

# merge
df1 = pd.merge(df1, cust, how="left", on=('cust_key'))

# drop extra columns
df1 = df1.drop(['cust_key','sales_key','Store','Date'], axis=1)


df1.head()


### Step-4: Exploratory Data Analysis

dfv = df.copy()
{"Mean":np.mean(dfv.Sales),"Median":np.median(dfv.Sales)}

## Sales Distribution
plt.figure(figsize=(10,5))
plt.hist(x=dfv.Sales, bins=30,color = "green")
plt.ylabel('number of observations')
plt.xlabel('daily sales in $')
plt.title('Sales Distribution')

## Customer Distribution
{"Mean":np.mean(dfv.Customers),"Median":np.median(dfv.Customers)}

plt.figure(figsize=(10,5))
plt.hist(x=dfv.Customers , bins=30,color = "pink")
plt.ylabel('number of observations')
plt.xlabel('daily total number of customers')
plt.title('Customer Distribution')

### Sales over a time

store1_2015 = dfv.query('Store == 1 and Year == 2015')
store1_2013 = dfv.query('Store == 1 and Year == 2013')
store1_2014 = dfv.query('Store == 1 and Year == 2014')

plt.figure(figsize=(12,5))
sns.lineplot(x=store1_2013.Date, y=store1_2013.Sales, data=store1_2013)
sns.lineplot(x=store1_2014.Date, y=store1_2014.Sales, data=store1_2014)
sns.lineplot(x=store1_2015.Date, y=store1_2015.Sales, data=store1_2015)
plt.title('Sales Over Time')

### The average of the sales over a time period is given in the figure We cas see that the sales are at its peak near every new year i.e around the time of christmas

###Sales over week
 
plt.figure(figsize=(15,5))
sns.barplot(x=dfv['WeekOfYear'],y=dfv['Sales'],data=dfv)
plt.xlabel('Week Of Year')
plt.ylabel('Total Sales')
plt.title('Sales Over Weeks')

## Sales by store type

# StoreType - differentiates between 4 different store models: a, b, c, d
plt.figure(figsize=(8,5))
sns.boxplot(x=dfv.StoreType, y=dfv.Sales, data=dfv)
plt.ylabel('Total Sales')
plt.title('Sales By Store Type')


###  Sales by Assortment 
plt.figure(figsize=(8,5)) 
sns.boxplot(x=dfv.Assortment, y=dfv.Sales, data=dfv) 
plt.ylabel('Total Sales') 
plt.title('Sales By Assortment')

sns.factorplot(data = dfv, x = 'Month', y = "Sales", 
               col = 'Assortment',
               palette = 'plasma',
               hue = 'StoreType')

## Sales vs. Competition Distance
plt.figure(figsize=(10,5))
plt.scatter(x=dfv.CompetitionDistance, y=dfv.Sales , c=dfv.Customers)
plt.ylabel('Sales')
plt.xlabel('Competition Distance')
plt.title('Sales vs. Competition Distance')
cbr= plt.colorbar()
cbr.set_label('Number Of Customers')

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


plt.figure(figsize=(15,5))
a = dfv[dfv['SchoolHoliday'] != 'regular_day']
plt.subplot(3,2,1)
sns.countplot(a['SchoolHoliday'])

sns.kdeplot(dfv[dfv['SchoolHoliday'] == 'public_holiday']['Sales'],label = 'public_holiday',shade = True)
sns.kdeplot(dfv[dfv['SchoolHoliday'] == 'easter_holiday']['Sales'],label = 'easter_holiday',shade = True)
sns.kdeplot(dfv[dfv['SchoolHoliday'] == 'christmas']['Sales'],label = 'christmas',shade = True)

#store_type

plt.subplot(3,2,3)
sns.countplot(dfv['StoreType'])

#assortment
plt.subplot(3,2,5)
sns.countplot(dfv['Assortment'])

"""
From all the plots above, we can get some conclusions:

state_holiday -> We have a much larger amount of sales on public holidays, but at Christmas, which has a smaller amount of sales than easter_holiday, it has a higher peak.

store_type -> The store_type "a" that sells more, does not have such a peak compared to the others.

assortment -> We see that stores with the "extra" type assortment sell less, but have a higher distribution. So, there are stores that sell more with the "extra" assortment and stores that sell less.
"""
## Impact of promotion on sales over days of a week
print ("Number of Stores opened on Sundays:{}" .format(dfv[dfv.DayOfWeek == 7]['Store'].unique().shape[0]))
sns.factorplot(data = dfv, x ="DayOfWeek", y = "Sales", hue='Promo',sharex=False)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
aux1 = dfv[['CompetitionDistance','Sales']].groupby('CompetitionDistance').sum().reset_index()
sns.scatterplot(x = 'CompetitionDistance', y = 'Sales', data = aux1);

plt.subplot(1,3,2)
bins = list(np.arange(0,20000,1000))
aux1['CompetitionDistance_binned'] = pd.cut(aux1['CompetitionDistance'], bins = bins)
aux2 = aux1[['CompetitionDistance_binned','Sales']].groupby('CompetitionDistance_binned').sum().reset_index()
sns.barplot(x = 'CompetitionDistance_binned', y = 'Sales', data = aux2);
plt.xticks(rotation = 90)
    
plt.subplot(1,3,3)
sns.heatmap(aux1.corr(method = 'pearson'), annot = True);

##### from above figure we conclude that Stores with longer competitors sell less.

## sale of store after the 10th of each month

plt.figure(figsize=(20,8))
aux1 = dfv[['Day','Sales']].groupby('Day').sum().reset_index()

plt.subplot(2,2,1)
sns.barplot(x = 'Day',y = 'Sales', data = aux1)

plt.subplot(2,2,2)
sns.regplot(x = 'Day',y = 'Sales', data = aux1)

plt.subplot(2,2,3)
sns.heatmap(aux1.corr(method = 'pearson'), annot = True)

plt.subplot(2,2,4)
aux1['before_after'] = aux1['Day'].apply(lambda x:'before_10_days' if x <= 10 else 'after_10_days')
aux2 = aux1[['before_after','Sales']].groupby('before_after').sum().reset_index()
sns.barplot(x = 'before_after', y = 'Sales', data = aux2)

### Stores sell more after the 10th of each month.


plt.figure(figsize=(20,8))
aux1 = dfv[['Month','Sales']].groupby('Month').sum().reset_index()

plt.subplot(1,3,1)
sns.barplot(x = 'Month',y = 'Sales', data = aux1)

plt.subplot(1,3,2)
sns.regplot(x = 'Month',y = 'Sales', data = aux1)

plt.subplot(1,3,3)
sns.heatmap(aux1.corr(method = 'pearson'), annot = True)

### Sale of store is less in 2nd half of the year

### Sale of Store over the Year

plt.figure(figsize=(20,8))
aux1 = dfv[['Year','Sales']].groupby('Year').sum().reset_index()

plt.subplot(1,3,1)
sns.barplot(x = 'Year',y = 'Sales', data = aux1)

plt.subplot(1,3,2)
sns.regplot(x = 'Year',y = 'Sales', data = aux1)

plt.subplot(1,3,3)
sns.heatmap(aux1.corr(method = 'pearson'), annot = True)

### Sale of Store has decreased over the year


plt.figure(figsize=(20,8))
aux1 = dfv[['DayOfWeek','Sales']].groupby('DayOfWeek').sum().reset_index()

plt.subplot(1,3,1)
sns.barplot(x = 'DayOfWeek',y = 'Sales', data = aux1)

plt.subplot(1,3,2)
sns.regplot(x = 'DayOfWeek',y = 'Sales', data = aux1)

plt.subplot(1,3,3)
sns.heatmap(aux1.corr(method = 'pearson'), annot = True)
## Stores sell less on weekends.

### Sale during School Holiday
plt.figure(figsize=(20,8))
plt.subplot(2,1,1)
aux1 = dfv[['SchoolHoliday','Sales']].groupby('SchoolHoliday').sum().reset_index()
sns.barplot(x = 'SchoolHoliday',y = 'Sales', data = aux1)

plt.subplot(2,1,2)
aux2 = dfv[['Month','SchoolHoliday','Sales']].groupby(['Month','SchoolHoliday']).sum().reset_index()
sns.barplot(x = 'Month',y = 'Sales',hue = 'SchoolHoliday', data = aux2)
## Stores sell less during school holidays except in July and August.

### Final Correalation HeatMap


# Converting categorial features Assortment and StoreType  
dfv['Assortment']=dfv['Assortment'].astype('category').cat.codes
dfv['StoreType']=dfv['StoreType'].astype('category').cat.codes

# Adding new feature SalesperCustomer to get better correlation with StoreType and Assortment
dfv['SalesperCustomer']=dfv['Sales']/dfv['Customers']
corr = dfv.corr()

mask = np.zeros_like(corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize = (11, 9))
sns.heatmap(corr, mask = mask, annot= True,
            square = True, linewidths = .5, ax = ax, cmap = "BuPu")
plt.title("Correlation Heatmap", fontsize=20)

### Step-5: Machine Learning Data Modelling
# split features and labels
X = df1.drop('Sales', axis=1)
y = df1.Sales

# get dummy variables for categorical features for linear regression
xd = X.copy()
xd = pd.get_dummies(xd)
xd.head()

# split training and test datasets
from sklearn.model_selection import train_test_split
xd_train,xd_test,yd_train,yd_test = train_test_split(xd,y,test_size=0.3, random_state=1)    


import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

## Linear Regression
model_lr= LinearRegression()
result = model_lr.fit(xd_train, yd_train)

# definte RMSE function
from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(x, y):
    return sqrt(mean_squared_error(x, y))

# definte MAPE function
def mape(x, y): 
    return np.mean(np.abs((x - y) / x)) * 100 

# get cross validation scores 
yd_predicted = result.predict(xd_train)
yd_test_predicted = result.predict(xd_test)

print("Regresion Model Score" , ":" , result.score(xd_train, yd_train) , "," ,
      "Out of Sample Test Score" ,":" , result.score(xd_test, yd_test))
print("Training RMSE", ":", rmse(yd_train, yd_predicted),
      "Testing RMSE", ":", rmse(yd_test, yd_test_predicted))
print("Training MAPE", ":", mape(yd_train, yd_predicted),
      "Testing MAPE", ":", mape(yd_test, yd_test_predicted))

regression_train = result.score(xd_train, yd_train)
regression_test = result.score(xd_test, yd_test)

## KNN (k-nearest-neighbour)

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()
knnreg = knn.fit(xd_train, yd_train)   
print("KNRegresion Model Score" , ":" , knnreg.score(xd_train, yd_train) , "," ,
      "Out of Sample Test Score" ,":" , knnreg.score(xd_test, yd_test))

yd_predicted = knnreg.predict(xd_train)
yd_test_predicted = knnreg.predict(xd_test)

print("Training RMSE", ":", rmse(yd_train, yd_predicted),
      "Testing RMSE", ":", rmse(yd_test, yd_test_predicted))
print("Training MAPE", ":", mape(yd_train, yd_predicted),
      "Testing MAPE", ":", mape(yd_test, yd_test_predicted))

Knn_train = knnreg.score(xd_train, yd_train)
Knn_test =  knnreg.score(xd_test, yd_test)


for x in range(1,5):
    knn = KNeighborsRegressor(n_neighbors = x)
    knnreg = knn.fit(xd_train, yd_train)
    print("Regresion Model Score" , ":" , knnreg.score(xd_train, yd_train) , "," ,
      "Out of Sample Test Score" ,":" , knnreg.score(xd_test, yd_test))
    

#### Desicion Tree Model

#Build decision tree model
tree_model = DecisionTreeRegressor(min_samples_leaf=20)

#Train model on training dataset
dt= tree_model.fit(xd_train, yd_train)
print(dt)

#Predict using model

yd_pred = tree_model.predict(xd_train)


#Calculate RMSE and MAPE

print("The model performance for training dataset:\n")
print("RMSE :",rmse(yd_train, yd_pred))
print("MAPE :",mape(yd_train, yd_pred))

dtree_train = dt.score(xd_train, yd_train)
dtree_test =  dt.score(xd_test, yd_test)

## Feature Importance

features = xd_train.columns
importances = dt.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8,8))
plt.title('Feature Importance', fontsize=20)
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')

### Comparision between the model
import pandas as pd

train_error=[regression_train,dtree_train,Knn_train]

test_error=[regression_test,dtree_test,Knn_test]

col={'Train Error':train_error,'Test Error':test_error}
models=['Linear Regression','Dtree','Knn']
dfm=pd.DataFrame(data=col,index=models)
dfm