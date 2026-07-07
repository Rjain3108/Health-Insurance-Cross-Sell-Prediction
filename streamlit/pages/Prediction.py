import streamlit as st
import requests
from pathlib import Path

predictionMethod = st.selectbox("Prediction Method", ["--Select--","Single", "Batch"])

if predictionMethod == "Single":
    st.write("You selected Single Prediction")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    id = st.number_input("id", min_value=0, max_value=120, value=30)
    Driving_License = st.checkbox("Driving License")
    Region_Code = st.number_input("Region Code", value=28.0)
    Previously_Insured = st.number_input("Previously Insured", min_value=0, max_value=1, value=0)
    Vehicle_Age = st.selectbox("Vehicle Age", ["< 1 Year", "1-2 Year", "> 2 Years"])
    Vehicle_Damage = st.selectbox("Vehicle Damage", ["No", "Yes"])
    Annual_Premium = st.number_input("Annual Premium", value=40454.0)
    Policy_Sales_Channel = st.number_input("Policy Sales Channel", value=26.0)
    Vintage = st.number_input("Vintage", min_value=0, max_value=1000, value=217)

    if st.button("Predict"):
        payload = {
        "Age": age,
        "Gender": gender,
        "id": id,
        "Driving_License": Driving_License,
        "Region_Code": Region_Code,
        "Previously_Insured": Previously_Insured,
        "Vehicle_Age": Vehicle_Age,
        "Vehicle_Damage": Vehicle_Damage,
        "Annual_Premium": Annual_Premium,
        "Policy_Sales_Channel": Policy_Sales_Channel,
        "Vintage": Vintage
    }
        url = "https://health-insurance-cross-sell-prediction.onrender.com/predict"
        response = requests.post( url, json=payload)
        result = response.json()
        print(result)
        st.success(result["label"])
        st.metric(
            "Probability",
            f"{result['probability']:.2%}"
        )

if predictionMethod == "Batch":
    st.write("You selected Batch Prediction")
    sample_csv = Path(__file__).resolve().parents[2]/"dataset"/"test.csv"
    with open(sample_csv, "rb") as file:
        st.download_button(
            label="📥 Download Sample CSV",
            data=file,
            file_name="sample_input.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    uploaded = st.file_uploader(
        "Upload CSV",
        type="csv"
    )

    if uploaded:
        print("Uploaded file:", uploaded.name)
        url = "https://health-insurance-cross-sell-prediction.onrender.com/batch-predict"
        files = {
            "file": (uploaded.name, uploaded.getvalue(), "text/csv")
        }
        headers = {
            "accept": "application/json"
        }
        response = requests.post(url, headers=headers, files=files)
        print("Status Code:", response.status_code)
        try:
            result = response.json()    
        except Exception:
            print(response.text)
        st.dataframe(result)