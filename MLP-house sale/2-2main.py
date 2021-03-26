# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:50:42 2020

@author: Mahsa Abedi
"""


from google.colab import files
uploaded = files.upload()
import io
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras import optimizers
import matplotlib.pyplot as plt
from keras.optimizers import adagrad
from tensorflow.keras.losses import mean_squared_logarithmic_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


names = ['id' ,'date',	'price'	,'bedrooms'	,'bathrooms'	,'sqft_living',	'sqft_lot'	,'floors'	,'waterfront'	,'view','condition','grade','sqft_above','sqft_basement',	'yr_built'	,'yr_renovated'	,'zipcode'	,'lat'	,'long'	,'sqft_living15',	'sqft_lot15']

dataset=pd.read_csv(io.BytesIO(uploaded['House Sales.csv']), names=names)
#dataset=pd.read_csv('House Sales.csv', names=names)

dataset = dataset.drop('date',1)
datasetX = dataset.drop('id',1)
#datasetX =dataset.drop('price',1)
#print(datasetX.shape)
datasetX = np.array(datasetX)


datasetY=dataset['price']
print(datasetY.shape)
datasetY=np.array(datasetY)

sc=StandardScaler()
dataset = sc.fit_transform(dataset)
X=datasetX[0:500,:]

Y=datasetY[0:500,]
print(dataset)

test_size = 0.2
seed = 7

X_train ,X_test , Y_train, Y_test = train_test_split(X , Y ,test_size=test_size, random_state=seed)


mymodel=Sequential()
mymodel.add(Dense(10,activation='relu',input_shape=(19,)))
mymodel.add(Dense(10,activation='relu'))
mymodel.add(Dense(1,kernel_initializer='normal'))
mymodel.summary()

mymodel.compile(optimizer="adam" , loss=mean_squared_logarithmic_error, metrics=["accuracy"])
trained_model=mymodel.fit(X_train,Y_train , batch_size=4 , epochs=60, validation_split=0.2)
history=trained_model.history

losses=history['loss']
val_losses=history['val_loss']

predicted_labels=mymodel.predict(X_test)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(Y_test)
plt.plot(predicted_labels)
plt.legend(['ytest','predicted'])

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(losses)
plt.plot(val_losses)
plt.legend(['loss','val_loss'])

plt.figure()
plt.xlabel('real price')
plt.ylabel('predicted price')
plt.scatter(Y_test , predicted_labels ,s=None ,)



test_loss=mymodel.evaluate(X_test,Y_test)
test_loss , test_acc =mymodel.evaluate(X_test,Y_test)
print("test loss is " ,test_loss)
print("test acc is",test_acc)

