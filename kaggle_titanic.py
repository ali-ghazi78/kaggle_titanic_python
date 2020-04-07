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
from sklearn.model_selection import GridSearchCV
#data["Indexes"]= data["Name"].str.find(sub) 


s_time = time.time()
data = pd.read_csv("train.csv")
data2 = pd.read_csv("test.csv")

 

#@jit
def data_pre_proccess(data):
    temp = pd.get_dummies(data.Pclass)
    data = pd.concat((data,temp),axis = 1)
    data.drop("Pclass" , axis = 1,inplace = True)
    data = data.drop('PassengerId' , axis= 1 )
    data.drop("Ticket" , axis =1 ,inplace=True)
    
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
    
        
    return do_sth_with_cabi(data)   
#@jit 
def do_sth_with_cabi(data):
    data.loc[data.Cabin.str.contains("na",na = True),"Cabin"] = "H"
    cabin_title = {'A':"A",
                   'B':'B',
                   'C':'C',
                   'D':'D',
                   'E':'E',
                   'F':'F',
                   'G':'G',
                   'H':'H',
                   'T':'F'
                   }
    for key,val in cabin_title.items():
        data.loc[data['Cabin'].str.contains(key,na = True),"Cabin"] =  val        
        pass

    temp = pd.get_dummies(data.Cabin)
    data = pd.concat((temp,data),axis = 1)
    data.loc[data.Cabin.str.len()>2,"Cabin"] =  "DDD"
    data.drop("Cabin" , axis =1 ,inplace=True)
    return data    
    
    pass


#@jit
def predict(data , data2,param):                  
    y = data["Survived"]
    X = data.drop("Survived" , axis= 1)

    np.random.seed(1)
    X_train , X_test , y_train, y_test = train_test_split(X,y,test_size = .2,shuffle = True)
    
    model = RandomForestClassifier(random_state = 1 , **param)

    model.fit(X_train,y_train)
    predict  = model.predict(X_test)
    accuracy = accuracy_score(y_test, predict)
    print(accuracy)
     
    
    frame = pd.DataFrame()
    frame['PassengerId']  =   pd.Series(range(892,1310,1))
    frame['Survived']  = pd.Series(model.predict(data2))
    frame.to_csv("result.csv",index = False)
 
    pass
@jit
def hp_optim(model,X,y):
    # Number of trees in random forest
    n_estimators = range(100,300,10)
    # Number of features to consider at every split
    #max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = (range(4,100,1))
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    #train_scoreNum, test_scoreNum = validation_curve(model, scoring="accuracy" ,cv=5,X=X, y=y, param_name = 'max_features', param_range=max_features)
    #train_scoreNum = train_scoreNum.mean(axis = 1)
    #test_scoreNum = test_scoreNum.mean(axis = 1)
    
    gridF = GridSearchCV(model, random_grid, cv = 5, verbose = 1, 
                          n_jobs = -1)
    X_train , X_test , y_train , y_test = train_test_split(X,y,train_size = .2)
    gridF.fit(X_train,y_train)
    print(gridF.best_params_) 

    #plt.plot(n_estimators,train_scoreNum,label = 'train' ) 
    #plt.plot(n_estimators,test_scoreNum,label = 'test' ) 
    #plt.legend()
    return gridF.best_params_
    pass    



best_param = {"n_estimators" : 190}
data = data_pre_proccess(data)
data2 = data_pre_proccess(data2)
model = RandomForestClassifier(random_state=10,n_estimators = 190)

y = data["Survived"]
X = data.drop("Survived" , axis= 1)

#param = {'bootstrap': True, 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 190}
param = hp_optim(model , X,y)
predict(data,data2,param)
print(f"time:{time.time() - s_time}s")

              
              
           

#data = data_pre_proccess(data)
#data2 = data_pre_proccess(data2)
#data = do_sth_with_cabi(data )



