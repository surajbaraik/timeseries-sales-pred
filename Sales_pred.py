# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:29:17 2021

@author: barsuraj1
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
#%matplotlib inline
warnings.filterwarnings('ignore')

train = pd.read_csv("storedata.csv")
test = pd.read_csv("test.csv")

#Combine test and train into one file
train['source']='train'
test['source']='test'
df = pd.concat([train, test],ignore_index=True)
print(train.shape, test.shape, df.shape)

#### total of 12 attributes, output is Item_Outlet_sales
## Check for unique values in dataset
df.nunique
df.apply(lambda x: len(x.unique()))
## More Number of uniques values indicates Numerical variables Like Item_Identifier, Item_Weight, Item_Visibility, Item_MRP, Item_Outlet_Sales
### Less Number shows Categorical Values

### basic analysis for statistical info(Numerical values)
df.describe()

## checking the datatypes
df.info

## Data Preprocesing
###check up for null Values
df.isnull().sum()
### @ Columns, Item Weights and Outlet Size with more number of null values

## check for categorical (Objective) data types
df.dtypes

cat_col = []

## loop for append all column name in one list
for x in df.dtypes.index:
    if df.dtypes[x] =='object':
        cat_col.append(x)
        
cat_col

#### REMOVING Item_Identifier & Outlet_Identifier as they are of not much use

cat_col.remove('Item_Identifier')
cat_col.remove('Outlet_Identifier')
cat_col


#categorical columns detail, number of counts

for col in cat_col:
    print(col)
    print(df[col].value_counts())
    print()

### for Item Fat content, low fat,LF Low Fat represnts same thing, so we will combine the attributes
## reg & Regular also represents same thing so will combine them
## Item types are reasonalable for modelling
### Outlet , Location and Outlet type are also given with good numbers


## Filling the missing values first, Item Weight
item_weight_mean = df.pivot_table(values="Item_Weight", index='Item_Identifier')
item_weight_mean
                                               
miss_val = df['Item_Weight'].isnull()
miss_val

for i, item in enumerate(df['Item_Identifier']):
    if miss_val[i]:
        if item in item_weight_mean:
            df['Item_Weight'][i] = item_weight_mean.loc[item]['Item_Weight']
        else:
            df['Item_Weight'][i] = np.mean(df['Item_Weight'])

df['Item_Weight'].isnull().sum()

### Check for outlet size missing values based on outlet type by using pivot and lamba function,

outlet_size_mode = df.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0])) # using mode as its categorial
outlet_size_mode

missing_boolean = df['Outlet_Size'].isnull()
df.loc[missing_boolean, 'Outlet_Size']= df.loc[missing_boolean, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
df['Outlet_Size'].isnull().sum()

#### Replace 0 with mean of Item_Visibility
sum(df['Item_Visibility']==0)
df.loc[:,'Item_Visibility'].replace([0], [df['Item_Visibility'].mean()], inplace=True)
sum(df['Item_Visibility']==0)

### combining item fat content for Low Fat and Regular using.replace or .map
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})
df['Item_Fat_Content'].value_counts()

#### Creating new attributes from the available content

df['New_Item_Type'] = df['Item_Identifier'].apply(lambda x: x[:2])
df.New_Item_Type

### creating New_Item_Type fat content for Food Non-Consumable & Drinks using.replace or .map
df['New_Item_Type']= df['New_Item_Type'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
df['New_Item_Type'].value_counts()

### Creating new category for Non-Edible
df.loc[df['New_Item_Type']=='Non-Consumable','Item_Fat_Content'] ='Non-Edible'
df['Item_Fat_Content'].value_counts()

### create small values for Establishment year
df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']
df.Outlet_Years

#### EDA
df.head()
sns.displot(df['Item_Weight'])
#### Item weight near mean is higher as we replace by mean

sns.displot(df['Item_Visibility'])
#almost left skewed , value is peaked at centre due to mean replcement

sns.displot(df['Item_MRP'])
###range of price

sns.displot(df['Item_Outlet_Sales'])
# sales is range from 0 to 14000
## using log transformation to normalise it as left skewed
##LOG transformation
df['Item_Outlet_Sales'] = np.log(1+df['Item_Outlet_Sales'])
sns.displot(df['Item_Outlet_Sales'])
## almost normal dist

##### for categorical attribute
sns.countplot(df['Item_Fat_Content'])
plt.figure(figsize=(15,5))
l = list(df['Item_Type'].unique())
chart = sns.countplot(df["Item_Type"])
chart.set_xticklabels(labels=l, rotation=90)

sns.countplot(df['Outlet_Establishment_Year'])
### only 1985 more no of stores were established

sns.countplot(df['Outlet_Size'])
## small Outlet size is more, High is less
sns.countplot(df['Outlet_Location_Type'])
### amost same
sns.countplot(df['Outlet_Type'])
plt.figure(figsize=(15,5))
l = list(df['Outlet_Type'].unique())
chart = sns.countplot(df["Outlet_Type"])
chart.set_xticklabels(labels=l, rotation=90)
### supermarket type 1 is more

#### Correlation Matrix
corr = df.corr()
corr
sns.heatmap(corr, annot=True, cmap='coolwarm')
## Item_MRP is more correlated

mask = np.zeros_like(corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize = (11, 9))
sns.heatmap(corr,square = True,mask=mask, linewidths = .5,ax=ax, cmap = "BuPu")
plt.title("Correlation Heatmap", fontsize=20)


###Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.columns

### First we convert OutletIdentifier into numerical values
df['Outlet'] = le.fit_transform(df['Outlet_Identifier'])
cat_col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type']

for i in cat_col:
    df[i] = le.fit_transform(df[i])

#for col in cat_col:
 #   df[col]=le.fit_transform(df[col])
df

### ONE HOT ENCODING
df = pd.get_dummies(df, columns=['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type'])

#df = pd.get_dummies(df, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','New_Item_Type'])
df.head()

df.dtypes

#### Input SPLIT
#X = df.drop(columns=['Outlet_Establishment_Year','Item_Identifier','Outlet_Identifier','Item_Outlet_Sales','Item_Type'])
#y = df['Item_Outlet_Sales']

import warnings
warnings.filterwarnings('ignore')
#Drop the columns which have been converted to different types:
df.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

#Divide into test and train:
train = df.loc[df['source']=="train"]
test = df.loc[df['source']=="test"]

#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

#Export files as modified versions:
train.to_csv("train_modified.csv",index=False)
test.to_csv("test_modified.csv",index=False)


#### MODEL TRAINING

# Reading modified data
train2 = pd.read_csv("train_modified.csv")
test2 = pd.read_csv("test_modified.csv")

train2.head()

X_train = train2.drop(['Item_Outlet_Sales', 'Outlet_Identifier','Item_Identifier'], axis=1)
y_train = train2.Item_Outlet_Sales

X_test = test2.drop(['Outlet_Identifier','Item_Identifier'], axis=1)
X_train.head()

y_train.head()


"""
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

def train(model, X,y):
    model.fit(X,y)
    #train the model
    
    pred = model.predict(X)
    #predicting the Trainaing set
    
    #Perform cross-validation
    cv_score = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=5)
    cv_score = np.abs(np.mean(cv_score))
    print("Model Report")
    print("MSE", mean_squared_error(y, pred))
    print("CV Score:", cv_score)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
model = LinearRegression(normalize=True)
train(model, X, y)
coeff = pd.Series(model.coef_, X.columns).sort_values()
coeff.plot(kind='bar', title="Model Coefficients")


model = Ridge(normalize=True)
train(model, X, y)
coeff = pd.Series(model.coef_, X.columns).sort_values()
coeff.plot(kind='bar', title="Model Coefficients")

model = Lasso()
train(model, X, y)
coeff = pd.Series(model.coef_, X.columns).sort_values()
coeff.plot(kind='bar', title="Model Coefficients")

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
train(model, X, y)
coeff = pd.Series(model.feature_importances_, X.columns).sort_values()
coeff.plot(kind='bar', title="Feature Importance")


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
train(model, X, y)
coeff = pd.Series(model.feature_importances_, X.columns).sort_values()
coeff.plot(kind='bar', title="Feature Importance")

from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
train(model, X, y)
coeff = pd.Series(model.feature_importances_, X.columns).sort_values()
coeff.plot(kind='bar', title="Feature Importance")

"""

#####
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
#X = df.drop(columns=['Outlet_Establishment_Year','Item_Identifier','Outlet_Identifier','Item_Outlet_Sales','Item_Type'])
#y = df['Item_Outlet_Sales']
#################################################



######### Feature Importance
from sklearn.ensemble import RandomForestRegressor
rdf = RandomForestRegressor(n_estimators=30)
rdfreg = rdf.fit(X_train, y_train)

features = X_train.columns
importances = rdfreg.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(8,10))
plt.title('Feature Importances', fontsize=20)
plt.barh(range(len(indices)), importances[indices], color='pink', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')


################333333# Fitting  Linear Regression to the training set
from sklearn.linear_model import  LinearRegression
# Measuring Accuracy
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn import metrics

regressor = LinearRegression()
regressor.fit(X_train, y_train)
# Predicting the test set results
y_pred = regressor.predict(X_test)
y_pred

lr_accuracy = round(regressor.score(X_train,y_train) * 100,2)
lr_accuracy
r2_score(y_train, regressor.predict(X_train))

#Perform cross-validation:
cv_score = cross_val_score(regressor, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
#print(np.sqrt(np.abs(cv_score)))
cv_score = np.abs(np.mean(cv_score))
print("CV Score:", cv_score)
print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(y_train, regressor.predict(X_train))))


################################### Ridge
from sklearn.linear_model import Ridge, Lasso
ridge_m = Ridge()
ridge_m.fit(X_train, y_train)
# Predicting the test set results
y_pred = ridge_m.predict(X_test)
y_pred

ridge_m_accuracy = round(ridge_m.score(X_train,y_train) * 100,2)
ridge_m_accuracy
r2_score(y_train, ridge_m.predict(X_train))

#Perform cross-validation:
cv_score = cross_val_score(ridge_m, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_score = np.abs(np.mean(cv_score))
print("CV Score:", cv_score)
print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(y_train, ridge_m.predict(X_train))))


############################## Lasso() 
lasso = Lasso()
lasso.fit(X_train, y_train)
# Predicting the test set results
y_pred = lasso.predict(X_test)
y_pred

lasso_accuracy = round(lasso.score(X_train,y_train) * 100,2)
lasso_accuracy
r2_score(y_train, lasso.predict(X_train))

#Perform cross-validation:
cv_score = cross_val_score(lasso, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_score = np.abs(np.mean(cv_score))
print("CV Score:", cv_score)
print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(y_train, lasso.predict(X_train))))


############################ Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
DTregressor = DecisionTreeRegressor(max_depth=15,min_samples_leaf=300)
DTregressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = DTregressor.predict(X_test)
y_pred

tree_accuracy = round(DTregressor.score(X_train,y_train),2)
tree_accuracy
r2_score(y_train, DTregressor.predict(X_train))

#Perform cross-validation:
cv_score = cross_val_score(DTregressor, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_score = np.abs(np.mean(cv_score))
print("CV Score:", cv_score)
print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(y_train, DTregressor.predict(X_train))))



###################### Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
RFregressor = RandomForestRegressor(n_estimators=100,max_depth=6, min_samples_leaf=50,n_jobs=4)
RFregressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = RFregressor.predict(X_test)
y_pred

rf_accuracy = round(RFregressor.score(X_train,y_train),2)
rf_accuracy
r2_score(y_train, RFregressor.predict(X_train))

#Perform cross-validation:
cv_score = cross_val_score(RFregressor, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_score = np.abs(np.mean(cv_score))
print("CV Score:", cv_score)
print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(y_train, RFregressor.predict(X_train))))


################################ Extra Tree Regressor
from sklearn.ensemble import ExtraTreesRegressor
ExtraReg = ExtraTreesRegressor()
ExtraReg.fit(X_train, y_train)

# Predicting the test set results
y_pred = ExtraReg.predict(X_test)
y_pred

ExtraReg_accuracy = round(ExtraReg.score(X_train,y_train),2)
ExtraReg_accuracy
r2_score(y_train, ExtraReg.predict(X_train))

#Perform cross-validation:
cv_score = cross_val_score(ExtraReg, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_score = np.abs(np.mean(cv_score))
print("CV Score:", cv_score)
print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(y_train, ExtraReg.predict(X_train))))

################# K NEAREST NEIGHBOUR REGRESSOR
from sklearn.neighbors import KNeighborsRegressor
knnreg = KNeighborsRegressor(n_neighbors = 30)
knnreg.fit(X_train, y_train)

# Predicting the test set results
y_pred = knnreg.predict(X_test)
y_pred

rf_accuracy = round(knnreg.score(X_train,y_train),2)
rf_accuracy
r2_score(y_train, knnreg.predict(X_train))

#Perform cross-validation:
cv_score = cross_val_score(knnreg, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_score = np.abs(np.mean(cv_score))
print("CV Score:", cv_score)
print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(y_train, knnreg.predict(X_train))))







