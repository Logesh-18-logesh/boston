import json
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from azureml.core.model import Model

app = Flask(__name__)

def init():
    global model
    global scalar

    # Load the model and scalar
    model_path = Model.get_model_path('regmodel')
    scalar_path = Model.get_model_path('scaling')
    model = pickle.load(open(model_path, 'rb'))
    scalar = pickle.load(open(scalar_path, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()['data']
        data = np.array(list(data.values())).reshape(1, -1)
        data = scalar.transform(data)
        prediction = model.predict(data)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    init()
    app.run(host='0.0.0.0', port=31311)
