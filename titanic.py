#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ArthurYang
"""


import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import models,layers
import operator
from functools import reduce



dftrain_raw = pd.read_csv('./data/train.csv')
dftest_raw = pd.read_csv('./data/test.csv')

def preprocessing(dfdata):

    dfresult= pd.DataFrame()

    #Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_' +str(x) for x in dfPclass.columns ]
    dfresult = pd.concat([dfresult,dfPclass],axis = 1)

    #Sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult,dfSex],axis = 1)

    #Age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    #SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    #Carbin
    dfresult['Cabin_null'] =  pd.isna(dfdata['Cabin']).astype('int32')

    #Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'],dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult,dfEmbarked],axis = 1)

    return(dfresult)

x_train = preprocessing(dftrain_raw)
y_train = dftrain_raw['Survived'].values

x_test = preprocessing(dftest_raw)

print("x_train.shape =", x_train.shape )
print("x_test.shape =", x_test.shape )

tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Dense(20,activation = 'relu',input_shape=(15,)))
model.add(layers.Dense(10,activation = 'relu' ))
model.add(layers.Dense(1,activation = 'sigmoid' ))

model.summary()


model.compile(optimizer='nadam',
            loss='binary_crossentropy',
            metrics=['AUC'])

history = model.fit(x_train,y_train,
                    batch_size= 64,
                    epochs=30,
                    validation_split=0.2)

#print(model.predict(x_test[0:10]), model.predict_classes(x_test[0:10]))

predictions = model.predict_classes(x_test)
predictions = predictions.reshape(len(predictions))




result = pd.DataFrame({'PassengerId':dftest_raw['PassengerId'], 
                       'Survived':predictions})
result.to_csv('submission.csv',index=False)
