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
    #data.loc[data.Sex.str.contains("female") ,"Sex"] = 1
    #data.loc[data.Sex.str.contains("male",na = False) ,"Sex"] = 0
    temp = pd.get_dummies(data.Sex)
    data = pd.concat((temp,data),axis = 1)
    data.drop("Sex",axis = 1,inplace = True)
    
    data.loc[data.Embarked.str.contains("na",na = False),"Embarked"] = "MISSING"
    temp = pd.get_dummies(data.Embarked)
    data = pd.concat([data , temp],axis = 1)    
    data.drop(labels = "Embarked",axis=1,inplace = True)
    
    data.Age.fillna(data.Age.mean(),inplace = True)
    data.Fare.fillna(data.Fare.mean(),inplace = True)
    

   # data.drop("Age",axis = 1,inplace = True)
    
  
    data = do_sth_with_cabi(data)   
    #data.drop("A",axis = 1 , inplace = True)
    #data.drop("G",axis = 1 , inplace = True)
    #data.drop("Q",axis = 1 , inplace = True)
    #data.drop("Royalty",axis = 1 , inplace = True)
    #data.drop("Officer",axis = 1 , inplace = True)
    #data.drop("F",axis = 1 , inplace = True)
    #data.drop("Age",axis = 1 , inplace = True)
    #data.drop("SibSp",axis = 1 , inplace = True)
    #data.drop("Parch",axis = 1 , inplace = True)
    
    return data
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

def predict(data , data2,param={}):                  
    y = data["Survived"]
    X = data.drop("Survived" , axis= 1)

    np.random.seed(1)
    X_train , X_test , y_train, y_test = train_test_split(X,y,test_size = .2,shuffle = True)
    
    model = svm.SVC(**param)
    
    model.fit(X_train,y_train)
    predict  = model.predict(X_test)
    accuracy = accuracy_score(y_test, predict)
    acc2 = cross_val_score(model, X,y).mean()
    print(f"\n\nthe cross_val    :{acc2} \nthe accuracy on test :{accuracy}\n\n")
     
    
    frame = pd.DataFrame()
    frame['PassengerId']  =   pd.Series(range(892,1310,1))
    frame['Survived']  = pd.Series(model.predict(data2))
    frame.to_csv("result.csv",index = False)
 
    pass
#
#@jit
def hp_optim(model,X,y):
    n_estimators = range(200,240,10)
    max_features = ['auto']#, 'sqrt']
    max_depth = ([80,90,100,110,120,70])
    min_samples_split = ([100,50,10,8,25])
    min_samples_leaf = ([2])
    bootstrap = [True]#, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'max_features': max_features , 
                   'bootstrap': bootstrap}
    random_grid_svm = {'gamma':np.arange(0.5,2,.1),
                       'C' : np.arange(0.01,2,.1),
                       'kernel': ['rbf' , "linear"]  
                    }
    gridF = GridSearchCV(model, random_grid_svm, cv = 5, verbose = 1, 
                          n_jobs = -1)
    X_train , X_test , y_train , y_test = train_test_split(X,y,train_size = .2,shuffle = True)
    gridF.fit(X_train ,y_train )
    print(gridF.best_params_) 
    predict = gridF.predict(X_test)
    acc = accuracy_score(y_test, predict)
    print(acc) 

    
    return gridF.best_params_
    pass    


data = data_pre_proccess(data)
data2 = data_pre_proccess(data2)
model = SVC("poly")


y = data["Survived"]
X = data.drop("Survived" , axis= 1)



fig , ax = plt.subplots( )
corr_mat = pd.concat((X,y),axis = 1 )
corr_mat = data.corr("pearson" )
sn.heatmap( corr_mat,ax = ax,annot = False ,xticklabels=True ,yticklabels=True)

fig2 , ax2 = plt.subplots()


labels = pd.DataFrame()
labels["label"] =   (corr_mat.columns)
labels = labels["label"].apply(str)
corr_mat = corr_mat.Survived

plt.xticks(rotation='vertical')
ax2.bar(labels,corr_mat    )


param = hp_optim(model , X,y)
predict(data,data2,param)
print(f"time:{time.time() - s_time}s")


