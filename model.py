# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:54:40 2020

@author: prate
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
from  sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

train["avg_training_score"].value_counts()
train.head(10)
train.describe()

#imputation
train['education']=train['education'].fillna("Bachelor's")
train['previous_year_rating']=train['previous_year_rating'].fillna(3)

train1=train

#dummy variables
"""
p1=pd.get_dummies(train1["department"])    
p2=pd.get_dummies(train1["education"])
p3=pd.get_dummies(train1["recruitment_channel"])
"""
labelencoder_X = LabelEncoder()
train1['gender'] = labelencoder_X.fit_transform(train1['gender'])
train1['department'] = labelencoder_X.fit_transform(train1['department'])
train1['education'] = labelencoder_X.fit_transform(train1['education'])
train1['recruitment_channel'] = labelencoder_X.fit_transform(train1['recruitment_channel'])

#train1=pd.concat([train1,p1,p2,p3],axis=1)

#removing columns
train1.drop(["employee_id","region"], axis=1,inplace =True)

y=train["is_promoted"]
x=train1
x.drop("is_promoted",axis=1,inplace=True)
#y=train["is_promoted"]

#splitting dataset
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size = 0.20,random_state =205)


sc_X = StandardScaler()
#x_train = sc_X.fit_transform(x_train)
#x_val = sc_X.transform(x_val)




#####Random Forest Classification
################################

from sklearn.ensemble import RandomForestClassifier

classifier3=RandomForestClassifier(n_estimators=1000, criterion='gini', random_state=0, max_depth=100,max_features=11, min_samples_leaf=4, min_samples_split=5,max_leaf_nodes=100)
classifier3.fit(x_train,y_train)

pred3=classifier3.predict(x_val)
trainpred3=classifier3.predict(x_train)

cm3=confusion_matrix(y_val,pred3)
print(cm3)

from sklearn.metrics import f1_score

f1_score(y_val,pred3, average='binary')
#f1 score=
f1_score(y_train,trainpred3, average='binary')




import pickle

pickle.dump(classifier3,open("model.pkl","wb"))
model=pickle.load(open("model.pkl","rb"))
