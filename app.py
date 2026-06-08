import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("skin_disease_model.pkl")

st.title("Skin Disease Prediction")

st.write("Enter Patient Details")

features = {}

feature_names = [
    'erythema',
    'scaling',
    'definite_borders',
    'itching',
    'koebner_phenomenon',
    'polygonal_papules',
    'follicular_papules',
    'oral_mucosal_involvement',
    'knee_and_elbow_involvement',
    'scalp_involvement',
    'family_history',
    'melanin_incontinence',
    'eosinophils_in_the_infiltrate',
    'PNL_infiltrate',
    'fibrosis_of_the_papillary_dermis',
    'exocytosis',
    'acanthosis',
    'hyperkeratosis',
    'parakeratosis',
    'clubbing_of_the_rete_ridges',
    'elongation_of_the_rete_ridges',
    'thinning_of_the_suprapapillary_epidermis',
    'spongiform_pustule',
    'munro_microabcess',
    'focal_hypergranulosis',
    'disappearance_of_the_granular_layer',
    'vacuolisation_and_damage_of_basal_layer',
    'spongiosis',
    'saw-tooth_appearance_of_retes',
    'follicular_horn_plug',
    'perifollicular_parakeratosis',
    'inflammatory_monoluclear_inflitrate',
    'band-like_infiltrate',
    'Age'
]

for col in feature_names:
    features[col] = st.number_input(
        col,
        value=0.0
    )

if st.button("Predict"):

    input_df = pd.DataFrame(
        [features]
    )

    prediction = model.predict(input_df)

    st.success(
        f"Predicted Disease Class: {prediction[0]}"
    )