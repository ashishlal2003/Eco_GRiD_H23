import pickle
import pandas as pd
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

lgbmModel = pickle.load(open('Eco_GRiD_H23\\ml-part\\best_lgbm_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html', prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        latitude = float(request.form['Latitude'])
        humidity = float(request.form['Humidity'])
        ambient_temp = float(request.form['AmbientTemp'])
        wind_speed = float(request.form['WindSpeed'])
        visibility = float(request.form['Visibility'])
        pressure = float(request.form['Pressure'])
        cloud_ceiling = float(request.form['CloudCeiling'])
        hour = int(request.form['Hour'])
        time = request.form['Time']

        season_spring = int(request.form.get('Season_Spring', 0))
        season_summer = int(request.form.get('Season_Summer', 0))
        season_winter = int(request.form.get('Season_Winter', 0))
        season_fall = int(request.form.get('Season_Fall', 0))

        location_usafa = int(request.form.get('Location_USAFA', 0))
        location_travis = int(request.form.get('Location_Travis', 0))
        location_peterson = int(request.form.get('Location_Peterson', 0))
        location_offutt = int(request.form.get('Location_Offutt', 0))
        location_marchafb = int(request.form.get('Location_MarchAFB', 0))
        location_hillweber = int(request.form.get('Location_HillWeber', 0))
        location_grissom = int(request.form.get('Location_Grissom', 0))
        location_campmurray = int(request.form.get('Location_CampMurray', 0))

        data = [
            [latitude, humidity, ambient_temp, wind_speed, visibility, pressure, cloud_ceiling,
             season_spring, season_summer, season_winter, season_fall,
             location_usafa, location_travis, location_peterson, location_offutt, location_marchafb,
             location_hillweber, location_grissom, location_campmurray,
             hour, time]
        ]

        input_data = pd.DataFrame(data)
        output = lgbmModel.predict(input_data)

        response = {'predictions': output}
        return jsonify(response)
    except Exception as e:
        # Handle errors and return an error response
        error_message = str(e)
        return jsonify({'error': error_message}), 400
    
if __name__ == '__main__':
    app.run(debug=True)