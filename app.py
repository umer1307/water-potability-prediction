
import streamlit as st
import joblib
import pandas as pd

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="AquaCheck | Water Predictor",
    page_icon="💧",
    layout="wide"
)

# --------------------------------
# LOAD MODEL
# --------------------------------
@st.cache_resource
def load_model():
    return joblib.load("water_model.pkl")

model = load_model()

FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]

# --------------------------------
# CUSTOM CSS
# --------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #F4F8FC;
    color: #172B4D;
}

header[data-testid="stHeader"] {
    background-color: transparent;
}

.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: 800;
    color: #075985;
}

.subtitle {
    text-align: center;
    color: #52657A;
    font-size: 16px;
    margin-bottom: 25px;
}

/* Make all input labels visible */
.stApp label,
.stApp [data-testid="stWidgetLabel"] p {
    color: #172B4D !important;
    font-weight: 600 !important;
}

/* Input field styling */
.stApp div[data-baseweb="input"] {
    background-color: #FFFFFF !important;
    border: 1px solid #B8C9D9 !important;
    border-radius: 8px !important;
}

.stApp div[data-baseweb="input"] input {
    color: #172B4D !important;
    background-color: #FFFFFF !important;
    -webkit-text-fill-color: #172B4D !important;
}

/* Input focus */
.stApp div[data-baseweb="input"]:focus-within {
    border: 1px solid #0284C7 !important;
}

/* Buttons */
.stButton > button {
    background-color: #0284C7;
    color: white !important;
    border-radius: 10px;
    border: none;
    height: 48px;
    font-weight: 700;
}

.stButton > button:hover {
    background-color: #0369A1;
}

/* Cards */
div[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #DCE6EF;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# HEADER
# --------------------------------
st.markdown(
    '<div class="main-title">💧 AquaCheck</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Water Quality & Potability Prediction System'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter water quality measurements to predict "
    "whether a water sample belongs to the potable "
    "or non-potable category."
)

st.divider()

# --------------------------------
# INPUT SECTION
# --------------------------------
st.subheader("🔬 Water Quality Parameters")

st.caption(
    "Suggested starting values are provided below. "
    "You can edit every field according to your sample."
)

col1, col2, col3 = st.columns(3)

with col1:

    ph = st.number_input(
        "pH Value",
        min_value=0.0,
        max_value=14.0,
        value=7.0,
        step=0.1,
        help="Acidity or alkalinity of water."
    )

    hardness = st.number_input(
        "Hardness",
        min_value=0.0,
        value=200.0,
        help="Calcium and magnesium content."
    )

    solids = st.number_input(
        "Total Dissolved Solids",
        min_value=0.0,
        value=20000.0,
        help="Dissolved minerals and salts."
    )

with col2:

    chloramines = st.number_input(
        "Chloramines",
        min_value=0.0,
        value=7.0,
        help="Disinfectant compounds in water."
    )

    sulfate = st.number_input(
        "Sulfate",
        min_value=0.0,
        value=300.0,
        help="Sulfate concentration."
    )

    conductivity = st.number_input(
        "Conductivity",
        min_value=0.0,
        value=400.0,
        help="Ability of water to conduct electricity."
    )

with col3:

    organic_carbon = st.number_input(
        "Organic Carbon",
        min_value=0.0,
        value=10.0,
        help="Organic carbon content."
    )

    trihalomethanes = st.number_input(
        "Trihalomethanes",
        min_value=0.0,
        value=60.0,
        help="Disinfection by-products."
    )

    turbidity = st.number_input(
        "Turbidity",
        min_value=0.0,
        value=4.0,
        help="Cloudiness of water."
    )

st.divider()

# --------------------------------
# PREDICTION BUTTON
# --------------------------------
if st.button(
    "💧 Predict Water Quality",
    type="primary",
    use_container_width=True
):

    # Create fresh input on every click
    input_data = pd.DataFrame([{
        "ph": ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": sulfate,
        "Conductivity": conductivity,
        "Organic_carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes,
        "Turbidity": turbidity
    }], columns=FEATURES)

    try:

        # Match model's training feature order
        if hasattr(model, "feature_names_in_"):
            input_data = input_data.loc[
                :, model.feature_names_in_
            ]

        # Make a fresh prediction
        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("📊 Prediction Result")

        if prediction == 1:

            st.success("💧 Water is Potable!")

            st.write(
                "The model predicts that this sample "
                "belongs to the potable class."
            )

        else:

            st.error("⚠️ Water is Not Potable!")

            st.write(
                "The model predicts that this sample "
                "belongs to the non-potable class."
            )

        # Show latest input values
        with st.expander("View Current Input Values"):

            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "This is a machine learning prediction, "
            "not a laboratory water safety test."
        )

    except Exception as e:

        st.error(f"Prediction Error: {e}")

# --------------------------------
# ABOUT PROJECT
# --------------------------------
st.divider()

with st.expander("ℹ️ About AquaCheck"):

    st.write("""
    AquaCheck uses a Decision Tree Classifier
    to predict water potability.

    Dataset: Water Potability Dataset

    Dataset Samples: 3,276

    Input Features: 9 water quality parameters

    Target:
    - 1 = Potable
    - 0 = Not Potable

    Technology:
    Python, Pandas, Scikit-learn and Streamlit.
    """)
