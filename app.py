import streamlit as st
import pandas as pd
import joblib

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Skin Disease Predictor",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>
    @keyframes fadeSlideUp {
        0% {
            opacity: 0;
            transform: translateY(20px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .main-header {
        background: linear-gradient(135deg,#0b2b3b,#0a1a2a);
        padding: 1.5rem;
        border-radius: 2rem;
        text-align: center;
        animation: fadeSlideUp 0.6s ease-out;
    }

    .main-header h1 {
        background: linear-gradient(
            120deg,
            #a0e9ff,
            #6ee7b7,
            #fde68a
        );
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-size: 3rem;
        font-weight: 800;
    }

    .prediction-card {
        background: linear-gradient(
            145deg,
            #0f172a,
            #1e293b
        );
        border-radius: 2rem;
        padding: 1.5rem;
        text-align: center;
        border-left: 6px solid #10b981;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load("skin_disease_model.pkl")

model = load_model()

# ==========================================================
# FEATURE LIST
# ==========================================================

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

# ==========================================================
# DISEASE MAPPING
# ==========================================================

disease_names = {
    1: "Psoriasis",
    2: "Seboreic Dermatitis",
    3: "Lichen Planus",
    4: "Pityriasis Rosea",
    5: "Chronic Dermatitis",
    6: "Pityriasis Rubra Pilaris"
}

disease_descriptions = {
    1: "Autoimmune disorder with red plaques and silvery scales.",
    2: "Inflammatory condition with greasy yellow scales.",
    3: "Pruritic purple polygonal papules often affecting oral mucosa.",
    4: "Acute self-limiting skin rash with herald patch.",
    5: "Chronic itchy dermatitis with thickened skin.",
    6: "Rare papulosquamous disorder with follicular plugging."
}

# ==========================================================
# SAMPLE PATIENTS
# ==========================================================

sample_profiles = {

    "Psoriasis": {
        'erythema':3,
        'scaling':3,
        'koebner_phenomenon':3,
        'knee_and_elbow_involvement':3,
        'family_history':1,
        'parakeratosis':3,
        'munro_microabcess':3,
        'Age':35
    },

    "Seboreic Dermatitis": {
        'scaling':3,
        'scalp_involvement':3,
        'itching':2,
        'follicular_papules':1,
        'Age':32
    },

    "Lichen Planus": {
        'itching':3,
        'polygonal_papules':3,
        'oral_mucosal_involvement':2,
        'saw-tooth_appearance_of_retes':3,
        'band-like_infiltrate':2,
        'Age':45
    },

    "Pityriasis Rosea": {
        'erythema':2,
        'scaling':2,
        'spongiosis':1,
        'Age':27
    },

    "Chronic Dermatitis": {
        'erythema':3,
        'itching':3,
        'exocytosis':3,
        'acanthosis':3,
        'spongiosis':2,
        'Age':52
    },

    "Pityriasis Rubra Pilaris": {
        'erythema':3,
        'scaling':3,
        'follicular_papules':3,
        'follicular_horn_plug':3,
        'perifollicular_parakeratosis':3,
        'family_history':1,
        'Age':28
    }
}

# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🩺 SKIN DISEASE PREDICTOR</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("## 📊 Model Performance")

    st.metric("Accuracy", "98.65%")
    st.metric("CV Score", "98.63%")

    st.markdown("---")

    st.markdown("## 🧪 Sample Patients")

    for name, profile in sample_profiles.items():

        if st.button(name, use_container_width=True):

            for feat in feature_names:
                st.session_state[feat] = profile.get(feat, 0)

            st.rerun()

    st.markdown("---")

    st.markdown("""
    ### 📖 How To Use

    1. Adjust sliders
    2. Or select a sample patient
    3. Click Predict Disease
    """)

# ==========================================================
# SESSION STATE DEFAULTS
# ==========================================================

for feat in feature_names:

    if feat not in st.session_state:

        if feat == "Age":
            st.session_state[feat] = 35
        else:
            st.session_state[feat] = 0

# ==========================================================
# INPUT SECTION
# ==========================================================

st.markdown("### 📋 Patient Features")

cols_per_row = 3

feature_groups = [
    feature_names[i:i+cols_per_row]
    for i in range(0, len(feature_names), cols_per_row)
]

for group in feature_groups:

    cols = st.columns(cols_per_row)

    for idx, feat in enumerate(group):

        with cols[idx]:

            if feat == "family_history":

                st.selectbox(
                    feat.replace('_',' ').title(),
                    [0,1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    key=feat
                )

            elif feat == "Age":

                st.slider(
                    "Age",
                    min_value=0,
                    max_value=75,
                    key=feat
                )

            else:

                st.slider(
                    feat.replace('_',' ').title(),
                    min_value=0,
                    max_value=3,
                    key=feat,
                    help="0=Absent | 1=Mild | 2=Moderate | 3=Severe"
                )

# ==========================================================
# PREDICTION
# ==========================================================

if st.button(
    "🔮 PREDICT DISEASE",
    use_container_width=True,
    type="primary"
):

    input_dict = {
        feat: st.session_state[feat]
        for feat in feature_names
    }

    input_df = pd.DataFrame([input_dict])

    with st.spinner("Analyzing Patient Data..."):

        prediction = model.predict(input_df)[0]

        disease = disease_names.get(
            prediction,
            f"Class {prediction}"
        )

        description = disease_descriptions.get(
            prediction,
            ""
        )

    st.balloons()

    st.markdown(
        f"""
        <div class="prediction-card">
            <h2>Predicted Disease</h2>
            <p style="font-size:2rem;font-weight:800;color:#fde68a;">
                {disease}
            </p>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.info(
        "👈 Adjust sliders or choose a sample patient and click Predict Disease."
    )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
    """
    ---
    🧬 Project PRCP-1027 |
    Developed by **Kalyana Sundar — AI Engineer**
    """
)