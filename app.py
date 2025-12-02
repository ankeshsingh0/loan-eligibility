from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import pickle

app = Flask(__name__)
CORS(app)

# Load model and scaler
model = pickle.load(open('loan_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    input_data = pd.DataFrame({
        'Gender': [1 if data['gender'] == "Male" else 0],
        'Married': [1 if data['married'] == "Yes" else 0],
        'Dependents': [int(data['dependents'].replace("3+", "3"))],
        'Education': [1 if data['education'] == "Graduate" else 0],
        'Self_Employed': [1 if data['self_employed'] == "Yes" else 0],
        'ApplicantIncome': [float(data['applicant_income'])],
        'CoapplicantIncome': [float(data['coapplicant_income'])],
        'LoanAmount': [float(data['loan_amount'])],
        'Loan_Amount_Term': [float(data['loan_term'])],
        'Credit_History': [int(data['credit_history'])],
        'Property_Area': [
            0 if data['property_area'] == "Rural" 
            else 1 if data['property_area'] == "Semiurban" 
            else 2
        ]
    })

    # Scale numeric values
    input_data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']] = (
        scaler.transform(input_data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']])
    )

    # Predict
    prediction = model.predict(input_data)[0]
    result = "Eligible" if prediction == 1 else "Not Eligible"

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
