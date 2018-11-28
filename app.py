import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug import secure_filename
import uuid
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn.metrics import *
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle

# Constants
UPLOAD_FOLDER = '/tmp/'
ALLOWED_EXTENSIONS = set(['xlsx', 'xls'])
dataFilename = '/tmp/data.pkl'
trainFilename = '/tmp/train.pkl'
testFilename = '/tmp/test.pkl'
modelFilename = '/tmp/model.pkl'
xTestDataFilename = '/tmp/xTest.pkl'
yTestDataFilename = '/tmp/yTest.pkl'

# Initialize flask app
app = Flask(__name__)
app.secret_key = uuid.uuid4().bytes
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

"""
    ROUTES
"""

@app.route('/', methods=['GET'])
def home():
    return """
    <html>
        <h1>Hello World</h1>
        <form action="/train/onehot" method="post">
            <input type="submit" name="oneHotTrain" value="Train One Hot" />
        </form>
        <form action="/train/w2v" method="post">
            <input type="submit" name="w2vTrain" value="Train Word2Vec" />
        </form>
        <form action="/test" method="post">
            <input type="submit" name="modelTest" value="Test Model" />
        </form>
    </html>
    """

@app.route('/train/onehot', methods=['POST', 'GET'])
def trainOneHot():
    data, train, test = loadDataTrain()
    wm = getOneHotModel(data, train)
    X, y = getXYOneHot(test, data)
    xTestDataFile = file.open(xTestDataFilename, 'wb')
    yTestDataFile = file.open(yTestDataFilename, 'wb')
    modelFile = file.open(modelFielname, 'wb')
    pickle.dump(X, xTestDataFile)
    pickle.dump(y, yTestDataFile)
    pickle.dump(wm, modelFile)
    return redirect('/')

@app.route('/train/w2v', methods=['POST', 'GET'])
def trainW2V():
    return redirect('/')

@app.route('/test', methods=['GET'])
def testModel():
    model, testX, testY = loadDataTest()
    return """
        <html>
            <div>
                <h1>Test Results</h1>
            </div>
            <div>
                <h4>Accuracy</h4><p>""" + str(accuracy_score(model.predict(testX), testY)) + """</p>
            </div>
        </html>
    """


"""
    Training Helper Functions
"""
def getXYW2V(d):
    X = list(d.w2v.values)
    y = list(d.polarity.values)
    return X, y
def getXYOneHot(d, data):
    cvX = CountVectorizer(token_pattern="\\w+", lowercase=True)
    cvX.fit(data.tweet)
    X = cvX.transform(d.tweet)
    y = d.polarity.values
    return X, y
def getOneHotModel(data, train):
    X, y = getXYOneHot(train, data)
    oneModel = linear_model.LogisticRegression(penalty='l2')
    oneModel.fit(X, y)
    return oneModel
def getW2VModel(data, train):
    X, y = getXYW2V(train)
    wModel = linear_model.LogisticRegression(penalty='l2')
    wModel.fit(X, y)
    return wModel

"""
    Data functions
"""
def dataSetup():
    data = pd.read_pickle('sentiment.pkl')
    data.polarity = data.polarity.apply(lambda x : 1 if x == 4 else 0)
    data = data[data.w2v.map(type) != np.float64]
    train = data[:4000]
    test = data[5000:]
    dataFile = open(dataFilename, 'wb')
    trainFile = open(trainFilename, 'wb')
    testFile = open(testFilename, 'wb')
    pickle.dump(data, dataFile)
    pickle.dump(train, trainFile)
    pickle.dump(test, testFile)

def loadDataTrain():
    data = pickle.load(dataFilename)
    train = pickle.load(trainFilename)
    test = pickle.load(testFilename)
    return data, train, test

def loadDataTest():
    model = pickle.load(modelFilename)
    testX = pickle.load(xTestDataFilename)
    testY = pickle.load(yTestDataFilename)
    return model, testX, testY

# Main
if __name__ == '__main__':
    dataSetup()
    app.run(debug=True, use_reloader=True)
