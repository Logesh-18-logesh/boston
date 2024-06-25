import json
import pickle
import os
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
app_dir = os.path.dirname(os.path.abspath(__file__))
regmodel_path = os.path.join(app_dir, 'models', 'regmodel.pkl')
scalar_path = os.path.join(app_dir, 'models', 'scaling.pkl')

regmodel = pickle.load(open(regmodel_path, 'rb'))
scalar = pickle.load(open(scalar_path, 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text="The House price prediction is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)
