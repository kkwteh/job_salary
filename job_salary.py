import pickle
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import job_salary_model_training

resources = {}
def create_app():
    app = Flask(__name__)
    with open('model.pkl', 'r') as f:
        resources['model'] = pickle.load(f)

    return app

app = create_app()

@app.route('/predict_author', methods=['POST', 'OPTIONS'])
def predict_author():
    json_request = request.get_json(force=True)

    # Construct design matrix
    df = pd.DataFrame(columns=json_request.keys(),
                      data=[json_request.values()])
    feature_fns = resources['model'].description['model_config']['feature_fns']
    X = job_salary_model_training.design_matrix(df, feature_fns)
    prediction = resources['model'].predict(X)[0]
    return jsonify({'prediction': prediction})


@app.route('/model_description')
def model_description():
    return jsonify(resources['model'].description)

if __name__ == "__main__":
    app.run(debug=True)
