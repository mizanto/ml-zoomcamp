import pickle
from flask import Flask, request, jsonify


with open('data/dv.bin', 'rb') as f_dv:
    dv = pickle.load(f_dv)

with open('data/model.bin', 'rb') as f_model:
    model = pickle.load(f_model)


def predict_single(client, dv, model):
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


app = Flask('bank')


@app.route('/', methods=['GET'])
def home():
    return 'Welcome to the Bank Scoring Page'


@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    prediction = predict_single(client, dv, model)
    decision = prediction >= 0.5

    result = {
        'prediction': float(prediction),
        'decision': bool(decision)
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
