from flask import Flask, jsonify, request
from sklearn.linear_model import ElasticNet
import json
import pandas
import pickle

app = Flask(__name__)

model = pickle.load(open('/Users/dannysievers/go/src/github.com/dannysievers/data-capture-sidecar/service/model/diabetes-progression.pkl', 'rb'))

@app.route("/predict", methods=["POST"])
def predict():
  features = json.loads(request.data)["features"]
  values = json.loads(request.data)["values"]
  df = pandas.DataFrame(values, columns=features)
  return jsonify(str(model.predict(df)))

if __name__ == '__main__':
  app.run(host='0.0.0.0')