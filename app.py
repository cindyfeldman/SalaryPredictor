import json
from distutils.log import debug
import numpy as np
import pandas as pd
from flask import Flask, request
import pickle, os

from sklearn import preprocessing

app = Flask(__name__)

salary_model = pickle.load(open('model.pickle', 'rb'))
prediction = None


@app.route('/predict', methods=['POST'])
def predict_conditions():
    global prediction
    server_input = request.get_json()
    input_list = list(server_input.values())
    print(input_list)
    label_encoder = preprocessing.LabelEncoder()
    input_list = label_encoder.fit_transform(input_list)
    print("prediction: ", salary_model.predict([input_list]))
    prediction = salary_model.predict([input_list]).tolist()
    return prediction, 200


if __name__ == "__main__":
    print("running")
    app.run(port=int(os.getenv('http://127.0.0.1', 8080)), debug=True)
