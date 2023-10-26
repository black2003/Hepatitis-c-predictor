import pandas as pd
import streamlit as st
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pickle
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Hepatitis C prediction"
)

LightGBM = pickle.load("LGBMCClassy.pkl", 'rb')
XGBoost = pickle.load("Xgb.pkl", 'rb')
analysis_option = st.sidebar.selectbox(
    "Select analysis option:",
    ("Predictor", "About Model Used", "About Us")
)
if analysis_option == "Predictor":
    st.title("Hepatitis C Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="text-align:center;">Hepatitis C Predictor using LFT parameters(App) </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    age = st.text_input("Age","Type Here")
    sex = st.selectbox("Sex:",("Male","Female"))
    ALB = st.text_input("ALB","Type Here(Albumin level)")
    ALP = st.text_input("ALP","Type Here(ALP Level)")
    ALT = st.text_input("ALT","Type Here(ALT Level)")
    AST = st.text_input("AST","Type Here(AST Level)")
    BIL = st.text_input("BIL","Type Here(BIL Level)")
    CHE = st.text_input("ALP","Type Here(CHE Level)")
    CHOL = st.text_input("CHOL","Type Here(CHOL Level)")
    CREA = st.text_input("CREA","Type Here(CREA Level)")
    GGT = st.text_input("GGT","Type Here(GGT Level)")
    PROT = st.text_input("PROT","Type Here(Protein Level)")
    analysis_method = st.selectbox(
        "Select Analysis Method:",
        ("LightGBM", "XGBoost")
    )
    features = np.array([[age,sex,ALB,ALP,ALT,AST,BIL,CHE,CHOL,CREA,GGT,PROT]])
    if analysis_method == "LightGBM":
        if st.button("Predict"):
            result=LightGBM.predict(features)
    elif analysis_method == "XGBoost":
        if st.button("Predict"):
            result=XGBoost.predict(features)
    if result == 0:
        st.success("Congrats! You are free of Hepatitis")
    elif result == 1:
        st.warning("You might have Cirrosis")
    elif result == 2:
        st.warning("You Might have Fibrosis")
    elif result == 3:
        st.error("You might have Hepatitis")
elif analysis_option == "About Model Used":
    st.write()
    about_option = st.selectbox("Select an option:", ("About LightGBM", "About XGBoost"))
    if about_option == "About LightGBM":
        with open('metrics1.txt', 'r') as metrics_file:
            metrics_data = metrics_file.read()
        st.download_button(label="Download LightGBM Metrics", data=metrics_data, key="model1_metrics")

    elif about_option == "About XGBoost":
        with open('metrics2.txt', 'r') as metrics_file:
            metrics_data = metrics_file.read()
        st.download_button(label="Download XGBoost Metrics", data=metrics_data, key="model2_metrics")
elif analysis_option == "About Us":
    st.write("We are the students!!!")
