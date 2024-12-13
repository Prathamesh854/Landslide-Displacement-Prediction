from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd

app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open('bilstm.pickle', 'rb'))
landslide = pd.read_csv('landslidedata.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    max_temperature = sorted(landslide['TEMP_MAX'].unique())
    min_temperature = sorted(landslide['TEMP_MIN'].unique())
    precipitation = sorted(landslide['PRECIPITATION'].unique(), reverse=True)
    specific_humidity = landslide['SPECIFIC_HUMIDITY'].unique()
    relative_humidity = landslide['RELATIVE_HUMIDITY'].unique()
    min_wind = landslide['WIND_MIN'].unique()
    max_wind = landslide['WIND_MAX'].unique()
    earthquake_depth = landslide['EARTHQUAKE_DEPTH'].unique()
    earthquake_magnitude = landslide['EARTHQUAKE_MAGNITUDE'].unique()

    return render_template('index.html', max_temperature=max_temperature, min_temperature=min_temperature,
                           precipitation=precipitation, specific_humidity=specific_humidity,
                           relative_humidity=relative_humidity, min_wind=min_wind, max_wind=max_wind,
                           earthquake_depth=earthquake_depth, earthquake_magnitude=earthquake_magnitude)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    max_temperature = float(request.form.get('max_temperature'))
    min_temperature = float(request.form.get('min_temperature'))
    precipitation = float(request.form.get('precipitation'))
    specific_humidity = float(request.form.get('specific_humidity'))
    relative_humidity = float(request.form.get('relative_humidity'))
    min_wind = float(request.form.get('min_wind'))
    max_wind = float(request.form.get('max_wind'))
    earthquake_depth = float(request.form.get('earthquake_depth'))
    earthquake_magnitude = float(request.form.get('earthquake_magnitude'))

    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'TEMP_MAX': [max_temperature],
        'TEMP_MIN': [min_temperature],
        'PRECIPITATION': [precipitation],
        'SPECIFIC_HUMIDITY': [specific_humidity],
        'RELATIVE_HUMIDITY': [relative_humidity],
        'WIND_MIN': [min_wind],
        'WIND_MAX': [max_wind],
        'EARTHQUAKE_DEPTH': [earthquake_depth],
        'EARTHQUAKE_MAGNITUDE': [earthquake_magnitude]
    })

    # Perform prediction
    prediction = model.predict(input_data)
    
    # Convert the prediction to a scalar and then round it
    rounded_prediction = round(prediction.item(), 2)

    return jsonify({'prediction': rounded_prediction})

if __name__ == '__main__':
    app.run()

