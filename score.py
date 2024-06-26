import json
import pickle
import os
import numpy as np
from azureml.core.model import Model

def init():
    global regmodel
    global scalar

    model_path = Model.get_model_path('regmodel')
    scalar_path = Model.get_model_path('scaling')
    regmodel = pickle.load(open(model_path, 'rb'))
    scalar = pickle.load(open(scalar_path, 'rb'))

def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = np.array(list(data.values())).reshape(1, -1)
        data = scalar.transform(data)
        prediction = regmodel.predict(data)
        return json.dumps({"prediction": prediction.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
