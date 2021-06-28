from flask import Flask, render_template, request, redirect, url_for, session, jsonify

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import tree 
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, recall_score, precision_score, auc, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

import pandas as pd 
import numpy as np
import pickle

filename = "model_lr.sav"
lr_model = pickle.load(open(filename, 'rb'))

print("Model loaded successfully")

def preprocess(X):
    X = X[3: ]
    print(X)

    if(X[1] == "Germany"):
        X[1] = 1
    else:
        X[1] = 0
    
    if(X[2] == "Male"):
        X[2] = 1
    else:
        X[2] = 0

    X = np.array(X,dtype="float")
    mean1 = 650.529	
    sd1 = 96.653

    mean3 = 38.922	
    sd3 = 10.488

    mean4 = 5.013	
    sd4 = 2.892

    mean5 = 76485.889	
    sd5 = 62397.405		

    mean9 = 100090.240	
    sd9 = 57510.493

    X[0] = (X[0] - mean1) / sd1
    X[3] = (X[3] - mean3) / sd3
    X[4] = (X[4] - mean4) / sd4
    X[5] = (X[5] - mean5) / sd5
    X[9] = (X[9] - mean9) / sd9
    return X

X_data = np.array(["1", "15634602","Hargrave",619,"France", "Female", 42, 2 ,0.00 ,1 ,1 ,1 ,101348.8])
X_data = np.array(["2","15647311",	"Hill",	608,	"Spain"	, "Female", 41,	1,	83807.86,1	,0	,1	,112542.58])


X_data = preprocess(X_data)

X_data = X_data.reshape(1,-1)
#X_data = np.array(X_data, dtype="float")
y_data = lr_model.predict(X_data)
print(y_data)