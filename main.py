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

from sklearn.neighbors import KNeighborsClassifier # K neighbors classification model
from sklearn.naive_bayes import GaussianNB # Gaussian Naive bayes classification model

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

import pandas as pd 
import numpy as np
import pickle
import json
import requests
import os


app = Flask(__name__)
port = int(os.environ.get('PORT', 5000))


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



def pred_lr(X):
    filename = "model_lr.sav"
    lr_model = pickle.load(open(filename, 'rb'))
    print("Model loaded successfully")
    y = lr_model.predict(X)
    return y

def pred_svc(X):
    filename = "model_svc.sav"
    svc_model = pickle.load(open(filename, 'rb'))
    print("Model loaded successfully")
    y = svc_model.predict(X)
    return y

def pred_rf(X):
    filename = "model_rf.sav"
    rf_model = pickle.load(open(filename, 'rb'))
    print("Model loaded successfully")
    y = rf_model.predict(X)
    return y

def pred_knn(X):
    filename = "model_knn.sav"
    knn_model = pickle.load(open(filename, 'rb'))
    print("Model loaded successfully")
    y = knn_model.predict(X)
    return y

def pred_gbc(X):
    filename = "model_gbc.sav"
    gbc_model = pickle.load(open(filename, 'rb'))
    print("Model loaded successfully")
    y = gbc_model.predict(X)
    return y

def pred_gnb(X):
    filename = "model_gnb.sav"
    gnb_model = pickle.load(open(filename, 'rb'))
    print("Model loaded successfully")
    y = gnb_model.predict(X)
    return y

def pred_ann(X):
    ann_model = load_model("model_ann2.h5")
    print("Model loaded successfully")
    y = ann_model.predict(X)
    return y

@app.route('/')
def index():
	return render_template('index2.html')


@app.route('/home')
def home():
    return render_template('index2.html')

@app.route('/predict', methods = ['GET', 'POST'])
def prediction():
    custid = request.form['custid']
    custname = request.form['custname']
    credscore = request.form['credscore']
    geog = request.form['geog']
    gender = request.form['gender']
    age = request.form['age']
    tenure = request.form['tenure']
    balance = request.form['balance']
    numprod = request.form['numprod']
    hascard = request.form['hascard']
    isactive = request.form['isactive']
    estdsal = request.form['estdsal']
    modelname = request.form['model']

    X = [" ", custid, custname, credscore, geog, gender, age, tenure, balance, numprod, hascard, isactive, estdsal]
    X = np.array(X)
    X = preprocess(X)
    X = X.reshape(1, -1)

    if(modelname == "LR"):
        y_pred = pred_lr(X)
        print(y_pred)
    
    elif(modelname == "SVC"):   
        y_pred = pred_svc(X)
        print(y_pred)
    
    elif(modelname == "RF"):
        y_pred = pred_rf(X)
        print(y_pred)
    
    elif(modelname == "KNN"):
        y_pred = pred_knn(X)
        print(y_pred)
       
    elif(modelname == "GNB"):
        y_pred = pred_gnb(X)
        print(y_pred)
    
    elif(modelname == "GBC"):
        y_pred = pred_gbc(X)
        print(y_pred)

    elif(modelname == "ANN"):
        y_pred = pred_ann(X)
        print(y_pred)

    if(y_pred > 0.5):
        pred = "Yes"
    else:
        pred = "No"

    return render_template("result.html", custid = custid, custname = custname, pred = pred)


@app.route('/predict2', methods = ['GET', 'POST'])
def prediction2():
    custid = request.form['custid']
    custname = request.form['custname']
    credscore = request.form['credscore']
    geog = request.form['geog']
    gender = request.form['gender']
    age = request.form['age']
    tenure = request.form['tenure']
    balance = request.form['balance']
    numprod = request.form['numprod']
    hascard = request.form['hascard']
    isactive = request.form['isactive']
    estdsal = request.form['estdsal']
    modelname = request.form['model']

    X = [" ", custid, custname, credscore, geog, gender, age, tenure, balance, numprod, hascard, isactive, estdsal]
    X = np.array(X)
    X = preprocess(X)
    X = X.reshape(1, -1)

    
    y_pred_lr = "Yes" if pred_lr(X) > 0.5 else "No"
    y_pred_svc = "Yes" if pred_svc(X) > 0.5 else "No"
    y_pred_rf = "Yes" if pred_rf(X) > 0.5 else "No"
    y_pred_knn = "Yes" if pred_knn(X) > 0.5 else "No"
    y_pred_gnb = "Yes" if pred_gnb(X) > 0.5 else "No"
    y_pred_gbc = "Yes" if pred_gbc(X) > 0.5 else "No"
    y_pred_ann = "Yes" if pred_ann(X) > 0.5 else "No"

    return render_template("result2.html", custid = custid, custname = custname, y_pred_ann = y_pred_ann, y_pred_gbc = y_pred_gbc, y_pred_gnb=y_pred_gnb, y_pred_knn = y_pred_knn, y_pred_lr = y_pred_lr, y_pred_rf = y_pred_rf, y_pred_svc = y_pred_svc)



if __name__ == '__main__':
	app.run(host='0.0.0.0', port=port, debug=True)