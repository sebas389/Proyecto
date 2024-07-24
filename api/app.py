from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Cargar el modelo y el scaler
model = load('random_forest_model.joblib')
scaler = StandardScaler()

def preprocess_input(data):
    df = pd.DataFrame(data, index=[0])

    # Aplica las mismas transformaciones que usaste en tu modelo
    df["Bathrooms"] = np.log(df["Bathrooms"] + 1)
    df["Bedrooms"] = np.log(df["Bedrooms"]+ 1)
    df["Area"] = np.log(df["Area"] + 1)
    df["Floor"] = np.log(df["Floor"] + 1 )

    # Dummy variables para District
    districts = ['list_of_districts_used_in_training']
    for district in districts:
        df[district] = 1 if df['District'] == district else 0

    df.drop("District", axis=1, inplace=True)

    # Escalar los datos
    df_scaled = scaler.transform(df)
    return df_scaled

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = preprocess_input(data)
    prediction = model.predict(input_data)

    predicted_price = prediction[0]

    return jsonify({'price': predicted_price})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
