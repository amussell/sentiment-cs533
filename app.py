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

global data, train, test, xTest, yTest, model

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
    global data, train, test, xTest, yTest
    wm = getOneHotModel(data, train)
    xTestData, yTestData = getXYOneHot(test, data)
    return redirect('/')

@app.route('/train/w2v', methods=['POST', 'GET'])
def trainW2V():
    global data, train, test
    return redirect('/')

@app.route('/test', methods=['GET', 'POST'])
def testModel():
    global model, testX, testY
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

glob = "FIRST"

@app.route('/')

# Main
if __name__ == '__main__':
    global data, train, test
    data = pd.read_pickle('sentiment.pkl')
    data.polarity = data.polarity.apply(lambda x : 1 if x == 4 else 0)
    data = data[data.w2v.map(type) != np.float64]
    train = data[:4000]
    test = data[5000:]
    app.run(debug=True, use_reloader=True)
