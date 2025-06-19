import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image
import subprocess

# Load the trained model and label encoders
def load_model_and_encoders():
    model = joblib.load("party_prediction_model.pkl")
    state_encoder = joblib.load("state_label_encoder.pkl")
    party_encoder = joblib.load("party_label_encoder.pkl")
    return model, state_encoder, party_encoder

def get_unique_states():
    df = pd.read_csv("cleaned_loksabha_data.csv")
    return sorted(df['state'].unique())

def show_chart(chart_path):
    if os.path.exists(chart_path):
        st.image(chart_path, use_column_width=True)
    else:
        st.error(f"Chart not found: {chart_path}")

def predict_party(state, year, turnout, margin, model, state_encoder, party_encoder):
    try:
        encoded_state = state_encoder.transform([state])[0]
        input_data = [[encoded_state, year, turnout, margin]]
        training_columns = list(pd.read_csv("cleaned_loksabha_data.csv").drop(columns=['party']).columns)
        input_df = pd.DataFrame(input_data, columns=['state', 'year', 'Turnout', 'margin'])
        for col in training_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[training_columns]
        prediction = model.predict(input_df)
        predicted_party = party_encoder.inverse_transform(prediction)[0]
        return predicted_party
    except Exception as e:
        return f"Prediction failed: {e}"

def update_dataset(uploaded_file):
    if uploaded_file is not None:
        with open("cleaned_loksabha_data.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Dataset updated successfully!")
    else:
        st.warning("Please upload a CSV file.")

def retrain_model():
    try:
        result = subprocess.run(["python", "4_model_party_prediction.py"], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("Model re-trained successfully!")
        else:
            st.error(f"Model re-training failed: {result.stderr}")
    except Exception as e:
        st.error(f"Failed to re-train the model: {e}")

# Streamlit UI
st.set_page_config(page_title="Online Voting Result Analysis System", layout="wide")
st.title("Online Voting Result Analysis System")

# Tabs
tabs = st.tabs(["EDA Analysis", "Prediction", "Settings"])

# EDA Tab
with tabs[0]:
    st.header("EDA Analysis")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Top 10 Parties by Wins"):
            show_chart("charts/top_parties_wins.png")
        if st.button("Victory Margin Distribution"):
            show_chart("charts/margin_distribution.png")
    with col2:
        if st.button("Voter Turnout Trend"):
            show_chart("charts/turnout_trend.png")
        if st.button("State-wise Election Participation"):
            show_chart("charts/statewise_elections.png")

# Prediction Tab
with tabs[1]:
    st.header("Predict Winning Party")
    model, state_encoder, party_encoder = load_model_and_encoders()
    unique_states = get_unique_states()
    state = st.selectbox("State", unique_states)
    year = st.selectbox("Year", list(range(1962, 2024, 5)))
    turnout = st.slider("Turnout (%)", 40, 100, 60)
    margin = st.slider("Margin of Victory (%)", 1, 50, 10)
    if st.button("Predict Winning Party"):
        result = predict_party(state, year, turnout, margin, model, state_encoder, party_encoder)
        st.info(f"The predicted winning party is: {result}")

# Settings Tab
with tabs[2]:
    st.header("Settings and Customization")
    st.subheader("Update Dataset")
    uploaded_file = st.file_uploader("Upload new cleaned_loksabha_data.csv", type=["csv"])
    if st.button("Update Dataset"):
        update_dataset(uploaded_file)
    st.subheader("Re-train Model")
    if st.button("Re-train Model"):
        retrain_model()
