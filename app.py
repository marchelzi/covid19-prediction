import streamlit as st
import joblib

# write a function that will load the saved model and make predictions


def load_model(model_file):
    loaded_model = joblib.load(open(model_file, "rb"))
    return loaded_model


def predict(model, input_df):
    predictions = model.predict(input_df)
    return predictions


# create a title Covid-19 Prediction
st.title("Covid-19 Prediction")

# Take input from the user
"""
## Input Features
- Sex
- Age
- Pneumonia
- Diabetes
- COPD
- Asthma
- Inmsupr
- Hipertension
- Other Disease
- Cardiovascular
- Obesity
- Renal Chronic
- Tobacco
- Other Disease
"""
st.write("Please fill the following information to predict the Covid-19")
