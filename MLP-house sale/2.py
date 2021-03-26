# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:45:03 2020

@author: Mahsa Abedi
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

names = ['id' ,'date',	'price'	,'bedrooms'	,'bathrooms'	,'sqft_living',	'sqft_lot'	,'floors'	,'waterfront'	,'view','condition','grade','sqft_above','sqft_basement',	'yr_built'	,'yr_renovated'	,'zipcode'	,'lat'	,'long'	,'sqft_living15',	'sqft_lot15']

dataset=pd.read_csv('house sales.csv' , names=names)

dataset=dataset.drop('date',1)
X=dataset.drop('price' , 1)
array=X.values
X=array[0:500,:]
#print(X.shape)


Y=dataset['price']
array=Y.values
Y=array[0:500,]
#print(Y)

test_size = 0.2
seed = 7

X_train ,X_test , Y_train, Y_test = train_test_split(X , Y ,test_size=test_size, random_state=seed)
# X_train = StandardScaler().fit_transform(X_train)
# X_test =StandardScaler().fit(X_test)


#Initializing the MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(3,),activation = 'relu',solver='adam',random_state=1)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
print(y_pred)



