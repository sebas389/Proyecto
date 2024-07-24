from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Cargar el modelo previamente entrenado usando pickle
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Cargar el escalador previamente entrenado
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

def preprocess_input(data):
    # Transforma los datos manualmente sin pandas ni numpy
    df = {
        'Bathrooms': [data['bathrooms']],
        'Bedrooms': [data['rooms']],
        'Area': [data['area']],
        'Floor': [data['floor']],
        'District': [data['district']]
    }
    
    # Aplica transformaciones
    for key in ['Bathrooms', 'Bedrooms', 'Area', 'Floor']:
        df[key] = [val + 1 for val in df[key]]  # Example transformation

    # Dummy variables para District
    districts = ['list_of_districts_used_in_training']
    for district in districts:
        df[district] = [1 if df['District'][0] == district else 0]

    # Escalar los datos (Ejemplo, ajustar según sea necesario)
    df_scaled = scaler.transform([df[k] for k in df.keys() if k != 'District'])

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
