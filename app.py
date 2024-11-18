import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the Random Forest model
model = pickle.load(open('glucose_guru_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data for the 8 features
    pregnancies = float(request.form['pregnancies'])
    bmi = float(request.form['bmi'])
    skin_thickness = float(request.form['skin_thickness'])
    insulin = float(request.form['insulin'])
    glucose = float(request.form['glucose'])
    blood_pressure = float(request.form['blood_pressure'])
    age = float(request.form['age'])
    diabetes_pedigree = float(request.form['diabetes_pedigree'])

    # Prepare features for prediction
    features = [pregnancies, bmi, skin_thickness, insulin, glucose, blood_pressure, age, diabetes_pedigree]

    # Predict using the loaded Random Forest model
    prediction = model.predict([features])

    # Interpret the prediction (assuming 1 = diabetic, 0 = non-diabetic)
    if prediction[0] == 1:
        message = "The person is predicted to be diabetic."
    else:
        message = "The person is predicted to be non-diabetic."

    # Return prediction result
    return render_template('index.html', prediction_text=message)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000,use_reloader=False)
