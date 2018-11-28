import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug import secure_filename
import uuid
import pandas as pd
import numpy as np

# Constants
UPLOAD_FOLDER = '/tmp/'
ALLOWED_EXTENSIONS = set(['xlsx', 'xls'])

# Initialize flask app
app = Flask(__name__)
app.secret_key = uuid.uuid4().bytes
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET'])
def home():
    return """
    <html>
        <h1>Hello World</h1>
        <form action="/train" method="post">
            <input type="submit" name="upvote" value="Train" />
        </form>
    </html>
    """

@app.route('/train', methods=['POST'])
def train():
    return ""

# Main
if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
