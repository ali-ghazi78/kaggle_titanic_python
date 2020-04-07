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
data2 = pd.read_csv("test.csv")


def data_pre_proccess(data):
    data = data.drop('PassengerId' , axis= 1 )
    data.drop("Ticket" , axis =1 ,inplace=True)
    data.loc[data.Cabin.str.contains("na",na = True),"Cabin"] = "??"
    temp = pd.get_dummies(data.Name)
    
    #data.drop("Cabin" , axis =1 ,inplace=True)
    
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
     
temp=0

def do_sth_with_cabi(data):
    cabin_title = ['A','B','C','D','E','F','G','T','??']
    global temp    
    for c_t in cabin_title:
        data[c_t] = (np.zeros(len(data)))
        data['Cabin'] = data['Cabin'].str.replace('A',' ')
        temp =  data.Cabin.str.isnumeric()
                        
    
    
    return data    
    
    
    
    
def predict(data,data2):                  
    data = data_pre_proccess(data)
    y = data["Survived"]
    X = data.drop("Survived" , axis= 1)
    X_train , X_test , y_train, y_test = train_test_split(X,y,test_size = .2,shuffle = True)
    
    np.random.seed(43)
    model = RandomForestClassifier(n_estimators=70)
    model.fit(X_train,y_train)
    predict  = model.predict(X_test)
    accuracy = accuracy_score(y_test, predict)
    print(accuracy)
     
    data2 = data_pre_proccess(data2)
    
    frame = pd.DataFrame()
    frame['PassengerId']  =   pd.Series(range(892,1310,1))
    frame['Survived']  = pd.Series(model.predict(data2))
    frame.to_csv("result.csv",index = False)
 

#predict(data)
data = data_pre_proccess(data)
data2 = data_pre_proccess(data2)
data = do_sth_with_cabi(data )
