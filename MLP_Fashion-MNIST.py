# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:44:01 2020

@author: Mahsa Abedi
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import Sequential
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report , confusion_matrix
import itertools
import ipykernel.pylab.backend_inline
from sklearn.metrics import plot_confusion_matrix


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)

# 
dict = {0:"T-shirt/top",1:"Trouser",2:"Pullover",3:"Dress",4:"Coat",5:"Sandal",6:"Shirt",7:"Sneaker",8:"Bag",9:"Ankle boot"}
# fig = plt.figure(figsize=(10,4))
# for i in range(10):
#     plt.subplot(2,5,i+1)
#     plt.imshow(X_train[y_train == i][0] ,cmap=plt.cm.binary)
#     plt.title((dict[i]))
#     plt.axis('off')


X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print("Training shape", X_train.shape)
#print(X_train.shape[0])
print("Testing shape", X_test.shape)

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)

model =Sequential()
model.add(Dense(512,activation='relu',input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))

from keras import optimizers
sgd = optimizers.SGD(lr=0.01)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


import datetime
start=datetime.datetime.now()

trained_model=model.fit(X_train,Y_train , batch_size=32 , epochs=60 ,validation_split = 0.2)
end=datetime.datetime.now()
Total_time_training=end-start

history=trained_model.history

losses=history['loss']
val_losses=history['val_loss']

ac=history['accuracy']
val_ac=history['val_accuracy']

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(losses)
plt.plot(val_losses)
plt.legend(['loss','val_loss'])


plt.figure()
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.plot(ac)
plt.plot(val_ac)
plt.legend(['acc','val_acc'])


predicted_labels=model.predict(X_test)
y_pred = model.predict_classes(X_test)
test_loss,test_acc=model.evaluate(X_test,Y_test)
print("test_loss is" , test_loss)
print("test_acc is ",test_acc)
print("Total_time_training",Total_time_training)

pd.set_option('display.max_columns', None)




target_names = [dict[x] for x in range(0,10)]
print("classification_report")
print(classification_report(y_test, y_pred, target_names=target_names))

plt.figure()
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
cnf_matrix = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cnf_matrix, classes=target_names , normalize=True, title='matrix')




# matrix = pd.DataFrame(confusion_matrix(y_test, y_pred), index=target_names, columns=target_names)
# matrix.index.name = 'Predicted'
# matrix.columns.name = 'Actual'
# print("confusion matrix " )
# print(matrix)
