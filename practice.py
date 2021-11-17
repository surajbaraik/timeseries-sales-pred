# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 15:20:38 2021

@author: barsuraj1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("train.csv")
df.head()

## Data Preprocessing
#selecting item 1 from store 1

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d') # convert date column to datatime object
df

# Filter records for store 1 and item 1 -> to be able to scale to other items in the future
df = df[df['store'] == 1]     
df = df[df['item'] == 1 ]

df = df.drop(['store', 'item'], axis=1)

df.set_index('date',inplace=True)
df.head()
df.describe()


#### Step 2: Visualize the Data

### Testing For Stationarity

from statsmodels.tsa.stattools import adfuller

test_result=adfuller(df['sales'])
    
#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")



adfuller_test(df['sales'])


##But for ACF and PACF  we need to do differencing

df['Seasonal First Difference']=df['sales']-df['sales'].shift()
df.head()
df['Seasonal First Difference'].plot()


## Again test dickey fuller test
adfuller_test(df['Seasonal First Difference'].dropna())

df['Seasonal First Difference'].plot()


## Auto Regressive Model

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['sales'])
plt.show()


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = plot_acf(df['sales'],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = plot_pacf(df['sales'],lags=40,ax=ax2)

## p,d,q
## p = AR model lags, d = differencing, q = Moving Average lags
## p = 08, q = 1,0 d=1 (since one seasonal differencing)

from statsmodels.tsa.arima_model import ARIMA

model=ARIMA(df['sales'],order=(8,1,1))
model_fit=model.fit()
model_fit.summary()

df['forecast']=model_fit.predict()
df[['sales','forecast']].plot(figsize=(12,8))



import statsmodels.api as sm
model =sm.tsa.statespace.SARIMAX(df['sales'],order=(1, 1, 1),seasonal_order=(1,1,1,2))
results=model.fit()

df['forecast']=results.predict(dynamic=True)
df[['sales','forecast']].plot(figsize=(12,8))
