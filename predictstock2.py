# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 00:39:12 2017

@author: User
"""

#it involves feature enginnering for a better result and the real motive behind workig on this dataset is to learn faeture engineering 
import pandas as pd
import numpy as np
import seaborn as sns #Seaborn will be used for making visualizations.
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
dataset = pd.read_csv("Tesla_stocks.csv")
#we are defining new column which goes something like percentage change of high and low values 

dataset["HL_Perc"] = (dataset["High"]-dataset["Low"]) / dataset["Low"] * 100
dataset["CO_Perc"] = (dataset["Close"] - dataset["Open"]) / dataset["Open"] * 100

#so our dataset now conatins two new columnn and they are change from low to high percentage and closed and open percentage
dataset = dataset[["HL_Perc", "CO_Perc", "Adj Close", "Volume"]] #we keep only those columns in our datasheet whih we need and remove the rest

#here we define our new dataframe which is going to be predicnextmonth and we laos shifted our datframe

dataset["PriceNextMonth"] = dataset["Adj Close"].shift(-30)

X = np.array(dataset.drop(["PriceNextMonth"], axis=1))# is our cut off datset and we don't have pricenextmonth in it

X_Check = X[-30:] #the last 30 rows in dataset that requires predition
X = X[:-30]#the entire datset except last 30 rows
dataset.dropna(inplace = True)#now dataset doesn't conati those last 30 NAn
y = np.array(dataset["PriceNextMonth"])# except last 30 rows it contains predict 
#scaling
X = preprocessing.scale(X)
#training and fitting in this part 
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size = 0.2)
model = RandomForestRegressor()

model.fit(X_train, y_train)

conf = model.score(X_test, y_test)
print(conf)

