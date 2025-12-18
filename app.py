import streamlit as st
import pandas as pd
import pickle

def load_model():
    with open("model_gbs.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def get_prediction_from_csv(data:pd.DataFrame):
    model = load_model()
    class_prediction = model.predict(data)
    prob_prediction = model.predict_proba(data)
    prob_prediction = prob_prediction[:,1]
    return {"class": class_prediction, "prob":prob_prediction}


st.title("HR Analytic Program")

st.write("Upload the Data")
upload_file = st.file_uploader(label="Upload your data (csv)", type="csv")

if upload_file is not None:
    df = pd.read_csv(upload_file)
    st.write(df)
    start_predict = st.button("Get Prediction!", use_container_width=True)

    if start_predict:
        st.write("Prediksi dimulai!")
        prediction = get_prediction_from_csv(df)
        df["prediction"] = prediction["class"]
        df["prediction_prob"] = prediction["prob"]
        st.write(df)