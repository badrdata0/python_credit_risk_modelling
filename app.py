"""
Credit Risk Prediction App
Application Streamlit qui prédit si un crédit est à risque faible (good) ou élevé (bad).
Lancement : streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

# Colonnes texte qui ont été encodées en nombres pendant l'entraînement
CAT_COLS = ["Sex", "Housing", "Saving accounts", "Checking account"]


# --- Chargement du modèle et des encodeurs (une seule fois grâce au cache) ---
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/credit_model.pickle")
    encoders = {col: joblib.load(f"models/{col}_encoder.pickle") for col in CAT_COLS}
    return model, encoders


model, encoders = load_artifacts()

# --- Interface ---
st.title("💳 Credit Risk Prediction App")
st.write("Renseigne les informations du demandeur pour estimer le risque du crédit.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Âge", min_value=18, max_value=80, value=30)
    sex = st.selectbox("Sexe", ["male", "female"])
    job = st.number_input("Niveau d'emploi (0 = non qualifié, 3 = très qualifié)",
                          min_value=0, max_value=3, value=1)
    housing = st.selectbox("Logement", ["own", "rent", "free"])

with col2:
    saving = st.selectbox("Compte épargne", ["little", "moderate", "rich", "quite rich"])
    checking = st.selectbox("Compte courant", ["little", "moderate", "rich"])
    amount = st.number_input("Montant du crédit (DM)", min_value=0, value=1000, step=100)
    duration = st.number_input("Durée (mois)", min_value=1, value=12)

# --- Préparation de l'entrée ---
# Mêmes colonnes, même ordre et même encodage que pendant l'entraînement
input_df = pd.DataFrame({
    "Age": [age],
    "Sex": encoders["Sex"].transform([sex]),
    "Job": [job],
    "Housing": encoders["Housing"].transform([housing]),
    "Saving accounts": encoders["Saving accounts"].transform([saving]),
    "Checking account": encoders["Checking account"].transform([checking]),
    "Credit amount": [amount],
    "Duration": [duration],
})

# --- Prédiction ---
if st.button("Prédire le risque"):
    pred = model.predict(input_df)[0]               # 1 = good, 0 = bad
    proba_good = model.predict_proba(input_df)[0][1]  # probabilité d'être "good"

    if pred == 1:
        st.success(f"Risque **faible** (good) — confiance : {proba_good:.0%}")
    else:
        st.error(f"Risque **élevé** (bad) — confiance : {1 - proba_good:.0%}")
