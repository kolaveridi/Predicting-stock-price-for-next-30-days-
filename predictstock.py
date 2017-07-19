# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 22:11:16 2017

@author: User
"""
#prediction of stock price for next 30 days without feature engineering

import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
#data reading
dataset=pd.read_csv('Tesla_stocks.csv')
#we define our own 
#we had our column where we vacated last 30 rows and thees are thr rows for which we will predict thr stock price 
dataset["PriceNextMonth"] = dataset["Adj Close"].shift(-30)

dates = np.array(dataset["Date"])
dates_check = dates[-30:]# dates_check contains the last thirty dates for which label will be predicted.
dates = dates[:-30]#this conatins every date except last 30 

X = np.array(dataset.drop(["PriceNextMonth", "Date"], axis=1)) #we dropped two columns out of our dataset as they are not usefulin predicting stock price
X_Check = X[-30:]#this conatins last 30 rows 
X = X[:-30]#every row except last 30 row
dataset.dropna(inplace = True)#to drop those rows which conatains NAn
y = np.array(dataset["PriceNextMonth"])# ok so this conatins the column price next month from our dataset and it excludes the last 30 rows for which we have to predict .think why ?

# training the datset
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size = 0.2)
#fittng the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

#checking
conf = model.score(X_test, y_test)
print(conf)
