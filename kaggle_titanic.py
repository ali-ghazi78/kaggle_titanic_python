# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm


#data["Indexes"]= data["Name"].str.find(sub) 

data = pd.read_csv("train.csv")
 

def data_pre_proccess(data):
    data = data.drop('PassengerId' , axis= 1 )
    data.drop("Ticket" , axis =1 ,inplace=True)
    data.drop("Cabin" , axis =1 ,inplace=True)
    
    #title = ['Ms','Countess','Mrs','Master','Mr','Miss','Dr','Rev','Cap','Don','Mme','Major','Col','Jonkheer','Mlle']
    #for t in title:
    #    data.loc[data.Name.str.contains(t),"Name"] = t
   
    #temp = pd.get_dummies(data.Name)
    data.drop("Name", axis = 1,inplace = True)
    #data = pd.concat([temp , data] , axis  = 1)
    data.loc[data.Sex.str.contains("female") ,"Sex"] = 1
    data.loc[data.Sex.str.contains("male",na = False) ,"Sex"] = 0
    data.loc[data.Embarked.str.contains("na",na = False),"Embarked"] = "MISSING"
    temp = pd.get_dummies(data.Embarked)
    data = pd.concat([data , temp],axis = 1)    
    data.drop(labels = "Embarked",axis=1,inplace = True)
    data.Age.fillna(data.Age.mean(),inplace = True)
    data.Fare.fillna(data.Fare.mean(),inplace = True)
    
        
    return data        


    
    
                  
data = data_pre_proccess(data)
y = data["Survived"]
X = data.drop("Survived" , axis= 1)
X_train , X_test , y_train, y_test = train_test_split(X,y,test_size = .2,shuffle = True)

model = svm.SVC(kernel='linear')
model.fit(X_train,y_train)
predict  = model.predict(X_test)
accuracy = accuracy_score(y_test, predict)
print(accuracy)
 
data2 = pd.read_csv("test.csv")
data2 = data_pre_proccess(data2)
predict = model.predict(data2)

#pd.save_csv("result.csv")
 


 

