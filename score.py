import json
import pickle
import numpy as np
from azureml.core.model import Model

def init():
    global model
    global scalar

    # Load the model and scalar
    model_path = Model.get_model_path('regmodel')
    scalar_path = "scaling.pkl"
    model = pickle.load(open(model_path, 'rb'))
    scalar = pickle.load(open(scalar_path, 'rb'))

def run(raw_data):
    data = json.loads(raw_data)['data']
    data = np.array(list(data.values())).reshape(1, -1)
    data = scalar.transform(data)
    prediction = model.predict(data)
    return json.dumps({"prediction": prediction.tolist()})
