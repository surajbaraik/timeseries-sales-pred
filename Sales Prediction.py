# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 16:45:29 2021

@author: barsuraj1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

#### Loading the dataset

sales_df = pd.read_csv("train.csv")
sales_df.shape
sales_df.dtypes
sales_df.head(5)

# Check columns list and missing values
sales_df.isnull().sum()
sales_df.nunique()
sales_df.describe()
sales_df.sales.min()
sales_df.date.min()
sales_df.date.max()
sales_df.item.max()
###Dataset contanins historical sales records from year 2013-2017 of 10 stores and 50 products.

## Data Preprocessing
#selecting item 1 from store 1

sales_df['date'] = pd.to_datetime(sales_df['date'], format='%Y-%m-%d') # convert date column to datatime object
sales_df

# Filter records for store 1 and item 1 -> to be able to scale to other items in the future
sales_df = sales_df[sales_df['store'] == 1]     
sales_df = sales_df[sales_df['item'] == 1 ]

##sales_df = sales_df[sales_df['store'] == 1]

# Create Date-related Features to be used for EDA and Supervised ML: Regression
sales_df['year'] = sales_df['date'].dt.year
sales_df['month'] = sales_df['date'].dt.month
sales_df['day'] = sales_df['date'].dt.day
sales_df['weekday'] = sales_df['date'].dt.weekday
sales_df['weekday'] = np.where(sales_df.weekday ==0, 7,sales_df.weekday)

# Split the series to predict the last 3 months of 2017
## train set contains sales record from January 2013 to September 2017 and 
## the test set (validation set) contains sales records of the last three month of 2017.

temp_df = sales_df.set_index('date')
train_df = temp_df.loc[:'2017-09-30'].reset_index(drop=False)                         
test_df = temp_df.loc['2017-10-01':].reset_index(drop=False)

train_df.head()
test_df.head()

#Data Exploration
## The plots below try to capture the trend and distribution of sales through weeks, months and years


monthly_agg = sales_df.groupby('month')['sales'].sum().reset_index()
yearly_agg = sales_df.groupby('year')['sales'].sum().reset_index()

plot = sns.boxplot(x='weekday', y='sales', data=sales_df)
_ = plot.set(title='Weekly sales distribution')
### The average number of sales increases over the week, is maximum on Day 6 (Saturday) , and takes a sharp fall on Day7(Sunday)

fig, axs = plt.subplots(nrows=2, figsize=(9,7))
sns.boxplot(x='month', y='sales', data=sales_df, ax=axs[0])
_ = sns.lineplot(x='month', y='sales', data=monthly_agg, ax=axs[1])

### Inference: The number of sales gradually ascends in the first half of the year starting February (2), peaks in July (7), and then gradually descends, before slightly increasing in November (11) and then dropping again in December (12).

fig, axs = plt.subplots(nrows=2, figsize=(9,7))
sns.boxplot(x='year', y='sales', data=sales_df, ax=axs[0])
_ = sns.lineplot(x='year', y='sales', data=yearly_agg, ax=axs[1])

## Inference: From the number of sales vs. year plot, we can infer an increasing trend over the years. The aggregate number of sales has increased from approximately 6000 in 2013 to slightly over 8000 in 2017, i.e. a 33.3% increase in the number of sales approximately. A clear trend is captured by the lineplot above.

plot = sns.lineplot(x='date', y='sales', data=sales_df)
_ = plot.set(title='Sales for Store 1, Item 1 over the years')

#### Inference: There is a seasonal pattern in the number of sales of 'item' - 1 at 'store' - 1. As also infered in the plot for sales vs. month above, we can see an increase in the sales in the first half of the year, peaking in July, and then a gradual decrease till December. This pattern is repeated each year, 2013 onwards.

###Forecast Sales
##  there is seasonality present in the product sales data, along with a greneral increase in the number of sales over the years.
# Therefore, in order to forecast the number of sales for the last three months of 2017, we will keep in mind the linear trend and seasonality present in the product sales.
## 

sales_df.columns
######################## DATA is non stationary, going up and upward trend
#### Checking Seasonality of data
sales_df = sales_df.drop(['store', 'item', 'year', 'month', 'day', 'weekday'], axis=1)
sales_df = sales_df.set_index('date')
sales_df.dtypes

### Determining Rolling Statistics
rolmean = sales_df.rolling(window=365).mean()
rolstd = sales_df.rolling(window=365).std()
print(rolmean, rolstd)

##plot rolling statictcs 
orig = plt.plot(sales_df, color='blue', label='Orignial')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Std Deviation')
plt.show(block=False)
##### mean and std deviation is not constant, hence data is not stationary

### The Augmented Dickey-Fuller
##### Perform Dicky-Fuller Test
from statsmodels.tsa.stattools import adfuller


dftest = adfuller(sales_df['sales'], autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistics','p-value','#Lags Used','Numbers of observation Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value

print('Result of Dicky-Fuller Test' )
print(dfoutput)

### estimating Trend
sales_df_LogScale = np.log(sales_df)
plt.plot(sales_df_LogScale)

movingAverage = sales_df_LogScale.rolling(window=365).mean()
movingstd = sales_df_LogScale.rolling(window=365).std()
plt.plot(sales_df_LogScale)
plt.plot(movingAverage, color='red')
##### Mean is not stationary, its moving upward with time so upward trend. Again it says data is not stationary


##### Difference Between Moving Average and actual sales

sales_df_LogScaleMINUSmovingAverage = sales_df_LogScale - movingAverage
sales_df_LogScaleMINUSmovingAverage.head()

## REMOVING NAN VALUES
sales_df_LogScaleMINUSmovingAverage.dropna(inplace=True)
sales_df_LogScaleMINUSmovingAverage.head()

from statsmodels.tsa.stattools import adfuller
def test_stationary(timeseries):
    
    ## Determining Rolling Statistics
    rolmean = timeseries.rolling(window=365).mean()
    rolstd = timeseries.rolling(window=365).std()
            
    ##plot rolling statictcs 
    orig = plt.plot(timeseries, color='blue', label='Orignial')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Std Deviation')
    plt.show(block=False)
    
    ##### Perform Dicky-Fuller Test
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistics','p-value','#Lags Used','Numbers of observation Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    
    print('Result of Dicky-Fuller Test' )
    print(dfoutput)  
    
test_stationary(sales_df_LogScaleMINUSmovingAverage)

#### Visually we can see there is no such trend or lets say better than previous cases

##### Calculated Weighted Average to Check the trend inside the time series

exponentialDecayWeightedAverage = sales_df_LogScale.ewm(halflife=12, min_periods=0, adjust=True).mean()
plt.plot(sales_df_LogScale)
plt.plot(exponentialDecayWeightedAverage, color='red')

### ANOTHER Transformation where LOGsclae - Weighted AVG
sales_df_LogScaleMINUSexponentialDecayAverage = sales_df_LogScale - exponentialDecayWeightedAverage
test_stationary(sales_df_LogScaleMINUSexponentialDecayAverage)

datasetLogDiffShifting = sales_df_LogScale - sales_df_LogScale.shift()
plt.plot(datasetLogDiffShifting)

datasetLogDiffShifting.dropna(inplace=True)
test_stationary(datasetLogDiffShifting)

#### COMPONENTS OF TIME SERIES
### ARIMA

from statsmodels.tsa.seasonal import seasonal_decompose
decompostion = seasonal_decompose(sales_df_LogScale)

trend = decompostion.trend
seasonal = decompostion.seasonal
residual = decompostion.resid

plt.subplot(411)
plt.plot(sales_df_LogScale, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='residual')
plt.legend(loc='best')
plt.tight_layout()

decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
test_stationary(decomposedLogData)




