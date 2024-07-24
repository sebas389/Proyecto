from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Simulación de carga de modelo
# En tu proyecto, reemplaza esto con la carga de tu modelo real
def dummy_predict(data):
    return [sum(data) * 1000]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    department = data['department']
    province = data['province']
    district = data['district']
    area = data['area']
    rooms = data['rooms']
    bathrooms = data['bathrooms']

    # Preprocesamiento de datos
    input_data = [area, rooms, bathrooms]  # Modifica según tu modelo
    # Realizar la predicción
    # prediction = modelo.predict(input_data)  # Descomentar para usar el modelo real
    prediction = dummy_predict(input_data)  # Simulación para este ejemplo

    predicted_price = prediction[0]

    return jsonify({'price': predicted_price})

if __name__ == '__main__':
    app.run(debug=True)
