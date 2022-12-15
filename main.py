import streamlit as st
import joblib
import numpy as np


def load_model(model_file):
    loaded_model = joblib.load(open(model_file, "rb"))
    return loaded_model


def process_input(input_features):

    # process the input features Yes/No to 1/0
    input_features = [1 if x == "Yes" else x for x in input_features]
    # process the input features Yes/No to 1/0
    input_features = [0 if x == "No" else x for x in input_features]
    # process the patient type
    input_features = [0 if x == "Hospitalized" else x for x in input_features]
    input_features = [1 if x == "Returned Home" else x for x in input_features]

    # process the sex type
    input_features = [1 if x == 'Female' else x for x in input_features]
    input_features = [0 if x == 'Male' else x for x in input_features]

    return np.array(input_features).astype(np.float64)


def build_input():
    usmer = st.selectbox("Medical History", ("Yes", "No"))
    sex = st.selectbox("What is your gender?", ("Female", "Male"))
    patient_type = st.selectbox(
        "What is your patient type?", ("Returned Home", "Hospitalized"))
    pneumonia = st.selectbox("Do you have Pneumonia?", ("Yes", "No"))
    age = st.select_slider("What is your Age?", options=range(0, 100))
    pregnant = st.selectbox("Are you pregnant?", ("Yes", "No"))
    diabetes = st.selectbox("Do you have Diabetes?", ("Yes", "No"))
    copd = st.selectbox("Do you have COPD?", ("Yes", "No"))
    ashma = st.selectbox("Do you have Asthma?", ("Yes", "No"))
    insupr = st.selectbox("Do you have Inmsupr?", ("Yes", "No"))
    hipertension = st.selectbox("Do you have Hipertension?", ("Yes", "No"))
    other_disease = st.selectbox("Do you have Other Disease?", ("Yes", "No"))
    cardiovascular = st.selectbox("Do you have Cardiovascular?", ("Yes", "No"))
    obesity = st.selectbox("Do you have Obesity?", ("Yes", "No"))
    renal_chronic = st.selectbox("Do you have Renal Chronic?", ("Yes", "No"))
    tobacco = st.selectbox("Do you have Tobacco?", ("Yes", "No"))

    input_features = [
        usmer,
        sex,
        patient_type,
        pneumonia,
        age,
        pregnant,
        diabetes,
        copd,
        ashma,
        insupr,
        hipertension,
        other_disease,
        cardiovascular,
        obesity,
        renal_chronic,
        tobacco,
    ]

    return process_input(input_features)


model = load_model("random_forest.pkl")


def main():

    st.title("Covid-19 Prediction")
    st.write("Please fill the following information to predict the Covid 19")
    st.write("## Input Features")

    input_data = build_input()

    prediction = model.predict_proba([input_data])

    if prediction[0][0] > 0.5:
        prediction = "Negative"
    else:
        prediction = "Positive"

    st.write("High Risk of Covid-19: ", prediction)


if __name__ == "__main__":
    main()
