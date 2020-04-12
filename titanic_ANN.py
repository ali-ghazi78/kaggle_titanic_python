# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from numba import jit, cuda
import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import validation_curve
import time
import seaborn as sn
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import tensorflow
import keras
import theano
from keras.models import Sequential
from keras.layers import Dense

s_time = time.time()
data = pd.read_csv("train.csv")
data2 = pd.read_csv("test.csv")


# @jit
def data_pre_proccess(data):
    temp = pd.get_dummies(data.Pclass)
    data = pd.concat((data, temp), axis=1)
    data.drop("Pclass", axis=1, inplace=True)
    data = data.drop('PassengerId', axis=1)
    data.drop("Ticket", axis=1, inplace=True)

    title = {'Ms': 'Mrs',
             'Countess': "Royalty",
             'Mrs': "Mrs",
             'Master': "Master",
             'Mr': "Mr",
             'Miss': "Miss",
             'Dr': "Officer",
             'Rev': "Officer",
             'Cap': "Officer",
             'Don': "Royalty",
             'Mme': "Miss",
             'Major': "Officer",
             'Col': "Officer",
             'Jonkheer': "Royalty",
             'Mlle': "Miss"}
    for key, value in title.items():
        data.loc[data.Name.str.contains(key), "Name"] = value
        pass

    temp2 = pd.get_dummies(data.Name)
    data = pd.concat([temp2, data], axis=1)
    data.drop("Name", axis=1, inplace=True)
    # data.loc[data.Sex.str.contains("female") ,"Sex"] = 1
    # data.loc[data.Sex.str.contains("male",na = False) ,"Sex"] = 0
    temp = pd.get_dummies(data.Sex)
    data = pd.concat((temp, data), axis=1)
    data.drop("Sex", axis=1, inplace=True)

    data.loc[data.Embarked.str.contains("na", na=False), "Embarked"] = "MISSING"
    temp = pd.get_dummies(data.Embarked)
    data = pd.concat([data, temp], axis=1)
    data.drop(labels="Embarked", axis=1, inplace=True)

    data.Age.fillna(data.Age.mean(), inplace=True)
    data.Fare.fillna(data.Fare.mean(), inplace=True)

    # data.drop("Age",axis = 1,inplace = True)

    data = do_sth_with_cabi(data)

    data.loc[data.Age < 18, "Age"] = 2
    data.loc[(data.Age >= 18) & (data.Age < 40), "Age"] = 3
    data.loc[(data.Age >= 40) & (data.Age < 55), "Age"] = 4
    data.loc[(data.Age >= 55), "Age"] = 5

    temp = pd.get_dummies(data.Age)
    data = pd.concat([temp, data], axis=1)
    data.drop("Age", inplace=True, axis=1)

    return data


# @jit
def do_sth_with_cabi(data):
    data.loc[data.Cabin.str.contains("na", na=True), "Cabin"] = "H"
    cabin_title = {'A': "A",
                   'B': 'B',
                   'C': 'C',
                   'D': 'D',
                   'E': 'E',
                   'F': 'F',
                   'G': 'G',
                   'H': 'H',
                   'T': 'A'
                   }
    for key, val in cabin_title.items():
        data.loc[data['Cabin'].str.contains(key, na=True), "Cabin"] = val
        pass

    temp = pd.get_dummies(data.Cabin)
    data = pd.concat((temp, data), axis=1)
    data.loc[data.Cabin.str.len() > 2, "Cabin"] = "DDD"
    data.drop("Cabin", axis=1, inplace=True)
    return data

    pass


def predict(data, data2):
    y = data["Survived"]
    X = data.drop("Survived", axis=1)

    # np.random.seed(1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=True)

    model = Sequential()
    model.add(Dense(kernel_initializer="uniform", units=19, activation="relu", input_dim=28))
    model.add(Dense(kernel_initializer="uniform", units=14, activation="relu"))
    model.add(Dense(kernel_initializer="uniform", units=1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, batch_size=5, nb_epoch=200)
    predict = model.predict(X_test)
    predict = predict > .5
    accuracy = accuracy_score(y_test, predict)
    acc4 = confusion_matrix(y_test, predict)
    a = pd.DataFrame()
    a["True"] = acc4[0, :]
    a["False"] = acc4[1, :]

    sn.heatmap(a, annot=True, xticklabels=True)
    plt.show()
    print(f"\n\nthe cross_val    :{accuracy} \nthe accuracy on test :{accuracy}\n\n{a}")

    frame = pd.DataFrame()
    frame['PassengerId'] = pd.Series(range(892, 1310, 1))
    pred2 = model.predict(data2)
    pred2 = pd.Series(pred2[:, 0])
    pred2 = pred2 > .5
    pred2 = list(pred2)
    for i,v in enumerate(pred2):
        if pred2[i] == True:
            pred2[i] = 1
        else:
            pred2[i] = 0
    frame['Survived'] = pred2
    frame.to_csv("result.csv", index=False)

    pass


#
# @jit


data = data_pre_proccess(data)
data2 = data_pre_proccess(data2)

y = data["Survived"]
X = data.drop("Survived", axis=1)

predict(data, data2)
print(f"time:{time.time() - s_time}s")
