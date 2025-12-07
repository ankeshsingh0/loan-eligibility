import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon="💰",
    layout="centered"
)

# ------------------ Load model & scaler ------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("loan_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_model()

# ------------------ Preprocessing ------------------
def preprocess_input(data, scaler):
    gender = 1 if data["Gender"] == "Male" else 0
    married = 1 if data["Married"] == "Yes" else 0
    dependents = 3 if data["Dependents"] == "3+" else int(data["Dependents"])
    education = 1 if data["Education"] == "Graduate" else 0
    self_emp = 1 if data["Self_Employed"] == "Yes" else 0

    property_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}
    property_area = property_map[data["Property_Area"]]

    df = pd.DataFrame({
        "Gender": [gender],
        "Married": [married],
        "Dependents": [dependents],
        "Education": [education],
        "Self_Employed": [self_emp],
        "ApplicantIncome": [data["ApplicantIncome"]],
        "CoapplicantIncome": [data["CoapplicantIncome"]],
        "LoanAmount": [data["LoanAmount"]],
        "Loan_Amount_Term": [data["Loan_Amount_Term"]],
        "Credit_History": [data["Credit_History"]],
        "Property_Area": [property_area],
    })

    num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
    df[num_cols] = scaler.transform(df[num_cols])

    return df

# ------------------ UI ------------------
st.title("💰 Loan Eligibility Predictor")
st.write("Fill the details below to check loan eligibility")

col1, col2 = st.columns(2)

with col1:
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
    LoanAmount = st.number_input("Loan Amount", min_value=1)
    Loan_Amount_Term = st.number_input("Loan Term (days)", min_value=1)
    Credit_History = st.selectbox("Credit History", [1, 0])
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# ------------------ Prediction ------------------
if st.button("Predict Loan Eligibility"):
    data = {
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": Property_Area
    }

    input_df = preprocess_input(data, scaler)
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("✅ Congratulations! You are Eligible for the Loan")
    else:
        st.error("❌ Sorry! You are NOT Eligible for the Loan")
