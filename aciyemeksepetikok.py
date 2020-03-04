# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:50:49 2020

@author: aidata
"""
import numpy as np
import pandas as pd
import os 
import re
import math,time
import itertools
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn import metrics
from string import punctuation
from email.parser import Parser
from sklearn.model_selection import cross_val_predict
from snowballstemmer import stemmer
import numpy.matlib as npm
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import keras
from keras.layers import LSTM,GlobalMaxPool1D,Dropout,GlobalAveragePooling1D
from keras.layers import Flatten,Embedding,Conv1D, MaxPooling1D
from keras import backend as K
import matplotlib.pyplot as plt
from keras.optimizers import Adam,Adadelta
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def mcor(y_true, y_pred):
     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
     y_pred_neg = 1 - y_pred_pos 
     y_pos = K.round(K.clip(y_true, 0, 1))
     y_neg = 1 - y_pos  
     tp = K.sum(y_pos * y_pred_pos)
     tn = K.sum(y_neg * y_pred_neg) 
     fp = K.sum(y_neg * y_pred_pos)
     fn = K.sum(y_pos * y_pred_neg) 
     numerator = (tp * tn - fp * fn)
     denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
     return numerator / (denominator + K.epsilon())
def precisionn(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())
# def recall1(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall
def f1score(y_true, y_pred):
    precision = precisionn(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall


L=1
R=1

def aciLR(mesaj,classification):
    Entropy=np.array([])  
    df=np.array([])
    df=np.array([np.append(df,x) for x in map(ord,mesaj)])
    Entropy=np.append(Entropy,acibul(df))    
    
    Entropy=np.append(Entropy,classification) #sınıflandırma için   
    return Entropy

def acibul(veri):
    acilar=[]
    for i in range(L,len(veri)-R) :
        P=veri[[i-L,i,i+R],:]
        z=np.array([[1],[2],[3]])
        z=np.append(z,P,axis=1)
        dd=np.linalg.det([(z[0,:]-z[1,:]),(z[1,:]-z[2,:])])
        dtt=np.dot(z[0,:]-z[1,:],z[1,:]-z[2,:])        
        ang = math.atan2(dd,dtt);
        ac=round(ang*180/math.pi)+180
        acilar=np.append(acilar, ac)    
    
    say=360      
    count=np.zeros(say)
    for k in range (say) :
        ss=len(np.where(acilar==(k))[0])
        if ss >0 :
            count[k]=ss
    return count 

data = pd.read_csv(r'D:\Dropbox\Doktora\veriseti\yemeklerin_sepeti\data.csv')
new = pd.read_csv(r'D:\Dropbox\Doktora\veriseti\yemeklerin_sepeti\new.csv')

Entropiler=np.zeros(361)[np.newaxis]  
df=np.array([])

for num, satir in enumerate(new.values):
    if (pd.isnull(satir[1])==False):
        Entropiler=np.concatenate([Entropiler,aciLR(satir[1],data['isPositive'][num])[np.newaxis, :]])
Entropiler = np.delete(Entropiler, (0), axis=0)

df=pd.DataFrame(Entropiler)

X = df.iloc[:,0:len(df.columns)-1]
Y = df.iloc[:, -1]
from sklearn.model_selection  import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=42)  


rd = ReduceLROnPlateau(monitor='val_accuracy', factor=np.sqrt(0.1), patience=4,verbose=1, mode='max')    
es = EarlyStopping(monitor="val_accuracy", mode="max", patience=10,restore_best_weights=True) 
mc = ModelCheckpoint("best_val_loss", monitor='val_accuracy', verbose=0, save_best_only=True, mode='auto',
                                 save_weights_only=True)

# model = Sequential()
# model.add(Embedding(25, 32, input_length=x_train.shape[2]))
# model.add(Dropout(0.25))
# model.add(Conv1D(64, 3, padding='valid', activation='relu', strides=1))
# model.add(MaxPooling1D())
# model.add(LSTM(70))
# model.add(Dense(1,activation='sigmoid'))
# model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
# model.summary()
# model.fit(x_train, y_train, batch_size=4, epochs=50,shuffle=True, 
#           validation_data=(x_test, y_test),
#           callbacks=[rd,es,mc])

# t0=time.time() 
# final_loss,final_acc,  mcor1,precisi, reca,f1,specificitytt = model.evaluate(x_test, y_test, verbose=0)
# t1=(time.time()-t0)/60
# print("Test süresi\t"+str(t1))
# print("Test accuracy\t{0:.6f} \nTest f1_score\t{1:.6f} \nTest precision\t{2:.6f} \nTest recall(Sensitivity)\t{3:.6f} \nTest specificity\t{4:.6f}\n".format(final_acc,f1, precisi,reca,specificitytt))


x_train=np.array(x_train)
x_test=np.array(x_test)

train_vecs = np.reshape(x_train, (x_train.shape[0], 1,x_train.shape[1]))
test_vecs = np.reshape(x_test, (x_test.shape[0], 1,x_test.shape[1]))  




# %87
# model = Sequential()
# model.add(LSTM(128,dropout=0.05,recurrent_dropout=0.05,activation="relu",return_sequences=True,input_shape=(1,train_vecs.shape[2])))
# model.add(LSTM(128,dropout=0.05,recurrent_dropout=0.05,activation="relu",return_sequences=False))
# model.add(Dense(144, activation="relu"))
# model.add(Dropout(0.15))
# model.add(Dense(1, activation="sigmoid"))

model = Sequential()
model.add(LSTM(72,dropout=0.2,recurrent_dropout=0.1,activation="relu",return_sequences=True,input_shape=(1,train_vecs.shape[2])))
model.add(GlobalAveragePooling1D())
# GlobalMaxPool1D
model.add(Dense(36, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))

learning_rate=0.001
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate),  metrics=['accuracy',mcor,precisionn, recall_m, f1score,specificity])
model.summary()



history = model.fit(train_vecs, y_train,
          shuffle=False,
          epochs=100,batch_size=2, verbose=2,
          validation_split=0.15,
          callbacks=[rd,es,mc])


t0=time.time() 
final_loss,final_acc,  mcor1,precisi, reca,f1,specificitytt = model.evaluate(test_vecs, y_test, verbose=0)
t1=(time.time()-t0)/60
print("Test süresi\t"+str(t1))
print("Test accuracy\t{0:.6f} \nTest f1_score\t{1:.6f} \nTest precision\t{2:.6f} \nTest recall(Sensitivity)\t{3:.6f} \nTest specificity\t{4:.6f}\n".format(final_acc,f1, precisi,reca,specificitytt))


# y_pred=model.predict(test_vecs,
#                       batch_size=1,
#                       verbose=1, steps=None)
# y_pred=y_pred>0.5

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_pred=y_pred,y_true=y_test)
# print('True Positives: ',cm[1,1])
# print('False Positives: ',cm[0,1])
# print('True Negatives: ',cm[0,0])
# print('False Negatives: ',cm[1,0])

# from sklearn.metrics import accuracy_score
# accuracy=accuracy_score(y_test,y_pred)
# print('Accuracy: %f' % accuracy)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], )
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Açı  Model loss  ')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()