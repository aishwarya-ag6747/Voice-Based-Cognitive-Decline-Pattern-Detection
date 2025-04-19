# app.py

import streamlit as st
import tempfile
from model import transcribe, extract_features, predict_risk
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pandas as pd

# Load scaler + model trained earlier
df = pd.read_csv("features.csv")
scaler = StandardScaler().fit(df.drop(columns=["file"]))
isof = IsolationForest(contamination=0.2, random_state=42).fit(scaler.transform(df.drop(columns=["file"])))

st.title("🧠 Voice-Based Cognitive Decline Detector")
st.write("Upload a `.wav` audio sample to predict cognitive risk.")

audio_file = st.file_uploader("Upload your `.wav` file", type=['wav'])

if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        path = tmp_file.name

    st.audio(audio_file, format='audio/wav')

    st.info("Transcribing and analyzing...")

    transcript = transcribe(path)
    st.write("**Transcript:**", transcript)

    features = extract_features(transcript, path)
    st.write("**Extracted Features:**", features)

    risk = predict_risk(features, scaler, isof)
    st.success(f"🧠 Cognitive Risk: **{risk}**")
