#!flask/bin/python

from flask import Flask
from flask import request
import pickle
import numpy as np
import json

app = Flask(__name__)

@app.route('/isAlive')
def index():
    return "true"

@app.route('/prediction/', methods=['GET'])
def get_prediction():
    feature = __process_inp(request.args.get('inputs'))
    model = pickle.load(open("best_estimator.pickle", "rb"))
    pred = model.predict(feature)
    return {'prediction': str(pred[0])}

def __process_inp(inputs):
    # model expects a 2d np array
    inputs = np.array(json.loads(inputs))
    inputs = np.array([inputs])
    return inputs

if __name__ == '__main__':
    # if os.environ['ENVIRONMENT'] == 'production':
    app.run(port=80,host='0.0.0.0')
