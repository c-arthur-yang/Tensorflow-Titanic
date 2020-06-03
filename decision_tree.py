#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
import operator
from functools import reduce
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier




dftrain_raw = pd.read_csv('./data/train.csv')
dftest_raw = pd.read_csv('./data/test.csv')

dftrain_raw.isna().sum()

dftest_raw.isna().sum()

dftest_raw[['Pclass', 'Fare']].groupby('Pclass').mean()


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
    dfresult['Fare'] = dfdata['Fare'].fillna(12)

    #Carbin
    dfresult['Cabin_null'] =  pd.isna(dfdata['Cabin']).astype('int32')
    #Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'],dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult,dfEmbarked],axis = 1)

    return(dfresult)

x_train = preprocessing(dftrain_raw)
y_train = dftrain_raw['Survived'].values
dftrain_raw.head(10)

x_test = preprocessing(dftest_raw)

print("x_train.shape =", x_train.shape )
print("y_train.shape =", y_train.shape )
print("x_test.shape =", x_test.shape )


x_train.isna().sum()

x_test.isna().sum()

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
predictions = dt.predict(x_test)

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
predictions = rfc.predict(x_test)


gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)
predictions = gbc.predict(x_test)

result = pd.DataFrame({'PassengerId':dftest_raw['PassengerId'], 
                       'Survived':predictions})
result.to_csv('submission.csv',index=False)


