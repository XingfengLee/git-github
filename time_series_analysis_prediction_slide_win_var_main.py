#!/bin/bash/python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet,HuberRegressor,Lars,LassoLars,PassiveAggressiveRegressor,RANSACRegressor,SGDRegressor
from collections import defaultdict 
import time

# slide window size 
winSize = 10

#read data
df = pd.read_csv('drug_time_series_data.csv', parse_dates=['date'])

models            = defaultdict(list) # create empty dictionary
models['LR']      = LinearRegression()
models['LASSO']   = Lasso()
models['Ridge']   = Ridge()
models['EN']      = ElasticNet()
models['Huber']   = HuberRegressor()
models['Lars']    = Lars()
models['LLars']   = LassoLars()
models['PA']      = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3)
models['Ranscac'] = RANSACRegressor()

list1 = [] 
for model in models:
	list1.append(model)

nModel = len(list1)

ones   = np.ones(winSize)
one1   = ones.reshape(winSize,1)
xTime  = np.array(range(0,df.shape[0]))
xTime1 = xTime.reshape(df.shape[0],1)

prediction = np.zeros([df.shape[0],nModel]) 
timeMat    = prediction

for i in range(winSize,df.shape[0]):
	y    = df.iloc[i-winSize:i,1]
	x    = xTime1[i-winSize:i]
	X1   = xTime1[i-winSize+1:i+1] # move to next value

	yHatAll = []
	mn   = 0;
	for model in models:
		fitModel = models[model]
		t1 = time.time()		
		fitModel.fit(x,y)				
		yhat = fitModel.predict(X1)[-1] # predict next value , and use the last value 
		yHatAll.append(yhat)		
		timeMat[i,mn] = time.time() - t1 
		mn += 1	
	prediction[i,:] = yHatAll
	
# time used

avgTime = np.mean(timeMat,axis=0)
mn = 0
for model in models:
	print(model + ' average time used:')
	print(str(np.round(avgTime[mn],3) ) + 'seconds')
	mn += 1

# Plot
fig, axes = plt.subplots(nrows=3, ncols=3, dpi=120, figsize=(10,6))
for i, ax in enumerate(axes.flatten()):    
    data = prediction[:,i]    
    ax.plot(data, color='red', linewidth=1)
    ax.plot(df['value'])
    MSE  = (sum((df['value']-data)**2))
    mse1 = np.round(MSE,1)
    # ax.set_title()
    ax.text(2,30, list1[i] + ' MSE: '+ str(mse1) )
    ax.text(2,25,'Avg time used: ' + str(np.round(avgTime[i],3))  + ' s')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)
plt.show()

