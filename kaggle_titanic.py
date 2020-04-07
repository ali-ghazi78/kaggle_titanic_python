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

 
temp=2

def data_pre_proccess(data):
    global temp2
    dummies = []
    temp = pd.get_dummies(data.Pclass)
    data = pd.concat((data,temp),axis = 1)
    data.drop("Pclass" , axis = 1)
    data = data.drop('PassengerId' , axis= 1 )
    data.drop("Ticket" , axis =1 ,inplace=True)
    data.loc[data.Cabin.str.contains("na",na = True),"Cabin"] = "??"
    data.drop("Cabin" , axis =1 ,inplace=True)
    
    title = {'Ms':'Miss',
                'Countess':"Royalty",
                'Mrs':"Miss",
                'Master':"Master",
                'Mr':"Mr",
                'Miss':"Miss",
                'Dr':"Officer",
                'Rev':"Officer",
                'Cap':"Officer",
                'Don':"Royalty",
                'Mme':"Miss",
                'Major':"Officer",
                'Col':"Officer",
                'Jonkheer':"Royalty",
                'Mlle':"Miss"  }
    for key,value in title.items():
        data.loc[data.Name.str.contains(key),"Name"] = value
        pass
    
    temp2 = pd.get_dummies(data.Name)
    data = pd.concat([temp2 , data] , axis  = 1)
    data.drop("Name",axis =1 ,inplace =True )
    
    data.loc[data.Sex.str.contains("female") ,"Sex"] = 1
    data.loc[data.Sex.str.contains("male",na = False) ,"Sex"] = 0
    
    data.loc[data.Embarked.str.contains("na",na = False),"Embarked"] = "MISSING"
    temp = pd.get_dummies(data.Embarked)
    data = pd.concat([data , temp],axis = 1)    
    data.drop(labels = "Embarked",axis=1,inplace = True)
    
    data.Age.fillna(data.Age.mean(),inplace = True)
    data.Fare.fillna(data.Fare.mean(),inplace = True)
    
        
    return data   
    
def do_sth_with_cabi(data):
    cabin_title = ['A','B','C','D','E','F','G','T','??']
    global temp    
    for c_t in cabin_title:
        data[c_t] = (np.zeros(len(data)))
        data[data['Cabin'].str.contains(c_t)] =   data.Cabin.str.extract('(\d+)')
        temp = data.Cabin.str.extract('(\d+)')
        
        pass
    
    return data    
    
    
    pass


def predict(data , data2):                  
    y = data["Survived"]
    X = data.drop("Survived" , axis= 1)
    X_train , X_test , y_train, y_test = train_test_split(X,y,test_size = .2,shuffle = True)
    
    np.random.seed(43)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train,y_train)
    predict  = model.predict(X_test)
    accuracy = accuracy_score(y_test, predict)
    print(accuracy)
     
    
    frame = pd.DataFrame()
    frame['PassengerId']  =   pd.Series(range(892,1310,1))
    frame['Survived']  = pd.Series(model.predict(data2))
    frame.to_csv("result.csv",index = False)
 
    pass



data = data_pre_proccess(data)
data2 = data_pre_proccess(data2)
predict(data,data2)

#data = data_pre_proccess(data)
#data2 = data_pre_proccess(data2)
#data = do_sth_with_cabi(data )
