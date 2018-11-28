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
UPLOAD_FOLDER = 'tmp/'

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

def plot_roc(gold, guess):
    x, y, _ = metrics.roc_curve(gold, guess)
    auc = metrics.auc(x, y)

    plt.figure()
    lw = 2
    plt.plot(x, y, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    return plt

global data, train, test, testX, testX, model

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
    global data, train, test, testX, testY, model
    model = getOneHotModel(data, train)
    testX, testY = getXYOneHot(test, data)
    return redirect('/')

@app.route('/train/w2v', methods=['POST', 'GET'])
def trainW2V():
    global data, train, test
    return redirect('/')

@app.route('/test', methods=['GET', 'POST'])
def testModel():
    global model, testX, testY
    plt = plot_roc(model.predict(testX), testY)
    plt.savefig('roc.png')
    return render_template('test.html', name="roc.png", accuracy=accuracy_score(model.predict(testX), testY))

# Main
if __name__ == '__main__':
    global data, train, test
    data = pd.read_pickle('sentiment.pkl')
    data.polarity = data.polarity.apply(lambda x : 1 if x == 4 else 0)
    data = data[data.w2v.map(type) != np.float64]
    train = data[:4000]
    test = data[5000:]
    app.run(debug=True, use_reloader=True)
