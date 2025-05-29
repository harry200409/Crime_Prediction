from flask import Flask, render_template, request
import pickle
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    model = None

# Initialize LabelEncoders for categorical features
encoders = {
    "city": LabelEncoder(),
    "crime_description": LabelEncoder(),
    "victim_gender": LabelEncoder(),
    "weapon_used": LabelEncoder(),
    "crime_domain": LabelEncoder()
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        city = request.form['City'].strip()
        crime_code = int(request.form['Crime Code'])
        crime_description = request.form['Crime Description'].strip()
        victim_age = int(request.form['Victim Age'])
        victim_gender = request.form['Victim Gender'].strip()
        weapon_used = request.form['Weapon Used'].strip()
        crime_domain = request.form['Crime Domain'].strip()
        police_deployed = int(request.form['Police Deployed'])

        city = city.encode("utf-8", "ignore").decode("utf-8")
        crime_description = crime_description.encode("utf-8", "ignore").decode("utf-8")
        victim_gender = victim_gender.encode("utf-8", "ignore").decode("utf-8")
        weapon_used = weapon_used.encode("utf-8", "ignore").decode("utf-8")
        crime_domain = crime_domain.encode("utf-8", "ignore").decode("utf-8")

        city_encoded = encoders["city"].fit_transform([city])[0]
        crime_desc_encoded = encoders["crime_description"].fit_transform([crime_description])[0]
        gender_encoded = encoders["victim_gender"].fit_transform([victim_gender])[0]
        weapon_encoded = encoders["weapon_used"].fit_transform([weapon_used])[0]
        crime_domain_encoded = encoders["crime_domain"].fit_transform([crime_domain])[0]

        features = np.array([[crime_code, victim_age, police_deployed, city_encoded, crime_desc_encoded, gender_encoded, weapon_encoded, crime_domain_encoded]])
        features = features.astype(np.float32)

        prediction = model.predict(features)
        output = "Case Not Closed" if prediction == 1 else "Case Closed"
        return render_template('index.html', prediction_text=output)

        
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)