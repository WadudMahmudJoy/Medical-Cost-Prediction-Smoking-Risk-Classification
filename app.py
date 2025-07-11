import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# --- Modern, Elegant Style ---
st.set_page_config(page_title="Medical Cost Prediction", layout="wide", page_icon="💡")
st.markdown("""
    <style>
    body {
        background: linear-gradient(120deg, #e0eafc 0%, #cfdef3 100%) !important;
    }
    .stApp {
        background: rgba(255,255,255,0.85);
        border-radius: 18px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.10);
    }
    .block-container {
        padding-top: 2rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #36d1c4 0%, #5b86e5 100%);
        color: #fff;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 0.5rem 2rem;
        box-shadow: 0 2px 8px rgba(91,134,229,0.10);
        transition: 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #5b86e5 0%, #36d1c4 100%);
        color: #fff;
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>div>div {
        background: #f7fafc;
        color: #222;
        border-radius: 8px;
        border: 1px solid #e0eafc;
    }
    .stTable {
        background: #f7fafc;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)


st.title("💡 Medical Cost Prediction & Smoking Risk Classification")
st.markdown("""
<div style='font-size:1.2rem; color:#444; margin-bottom:1.5rem;'>
Enter your details below. All fields are required. Errors will be shown inline. <br>
<b>Click <span style='color:#36d1c4;'>Submit</span> to see your prediction and charts.</b>
</div>
""", unsafe_allow_html=True)


# --- Input Columns ---
with st.form("input_form", clear_on_submit=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30, help="Enter your age (0-120)")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], help="Select your gender")
        n_medical_services = st.number_input("# Medical Services", min_value=0, max_value=50, value=2, help="Number of medical services used")
    with col2:
        premium = st.number_input("Premium ($)", min_value=0.0, max_value=100000.0, value=5000.0, help="Annual insurance premium paid")
        distribution_channel = st.selectbox("Distribution Channel", ["Online", "Agent", "Branch", "Other"], help="How you bought insurance")
        type_policy = st.selectbox("Type of Policy", ["Basic", "Comprehensive", "Premium"], help="Select policy type")
    with col3:
        type_product = st.selectbox("Type of Product", ["Health", "Life", "Accident", "Other"], help="Select product type")
        cost_claims_year = st.number_input("Cost of Claims (Year)", min_value=0.0, max_value=100000.0, value=2000.0, help="Total claims cost in a year")

    # --- Validation ---
    error = False
    if age < 0 or age > 120:
        st.error("Age must be between 0 and 120.")
        error = True
    if n_medical_services < 0:
        st.error("Number of medical services cannot be negative.")
        error = True
    if premium < 0:
        st.error("Premium cannot be negative.")
        error = True
    if cost_claims_year < 0:
        st.error("Cost of claims cannot be negative.")
        error = True

    submitted = st.form_submit_button("Submit", use_container_width=True)


# --- Only process and show results after submit and if no error ---
if 'submitted' in locals() and submitted and not error:
    # --- Data Preparation ---
    input_dict = {
        'age': age,
        'gender': gender,
        'premium': premium,
        'n_medical_services': n_medical_services,
        'distribution_channel': distribution_channel,
        'type_policy': type_policy,
        'type_product': type_product,
        'cost_claims_year': cost_claims_year
    }
    input_df = pd.DataFrame([input_dict])

    # --- Encoding ---
    le_gender = LabelEncoder().fit(["Male", "Female", "Other"])
    le_channel = LabelEncoder().fit(["Online", "Agent", "Branch", "Other"])
    le_policy = LabelEncoder().fit(["Basic", "Comprehensive", "Premium"])
    le_product = LabelEncoder().fit(["Health", "Life", "Accident", "Other"])

    input_df['gender'] = le_gender.transform(input_df['gender'])
    input_df['distribution_channel'] = le_channel.transform(input_df['distribution_channel'])
    input_df['type_policy'] = le_policy.transform(input_df['type_policy'])
    input_df['type_product'] = le_product.transform(input_df['type_product'])

    # --- Feature Engineering ---
    input_df['age_group'] = pd.cut(input_df['age'], bins=[0, 30, 50, 120], labels=[0, 1, 2]).astype(int)
    input_df['premium_per_service'] = input_df['premium'] / (input_df['n_medical_services'] + 1)

    # --- Scaling ---
    scaler = StandardScaler()
    scaled_cols = ['age', 'premium', 'n_medical_services', 'premium_per_service']
    input_df[scaled_cols] = scaler.fit_transform(input_df[scaled_cols])

    # --- Mock Model Prediction (Replace with your model) ---
    def mock_predict(df):
        # This is a placeholder. Replace with your trained model's prediction.
        base = 1000 + df['age'].values[0]*100 + df['premium'].values[0]*0.5 + df['n_medical_services'].values[0]*200
        if df['gender'].values[0] == 1:  # Female
            base *= 0.95
        if df['type_policy'].values[0] == 2:  # Premium
            base *= 1.2
        return np.round(base, 2)

    predicted_cost = mock_predict(input_df)

    # --- Results & Visualization ---
    colA, colB = st.columns([2,1])

    with colA:
        st.subheader("Prediction Result")
        st.markdown(f"<div style='font-size:2.2rem; color:#5b86e5; font-weight:bold; background:rgba(91,134,229,0.08); border-radius:12px; padding:1rem 2rem; margin-bottom:1rem;'>Predicted Medical Cost: <span style='color:#222;'>{predicted_cost} $</span></div>", unsafe_allow_html=True)
        st.markdown("#### Your Input Summary:")
        st.table(input_df.drop(['age_group','premium_per_service'], axis=1))

    with colB:
        st.subheader("Cost Breakdown")
        fig, ax = plt.subplots(figsize=(4,4))
        labels = ['Base', 'Age', 'Premium', 'Services']
        values = [1000, input_df['age'].values[0]*100, input_df['premium'].values[0]*0.5, input_df['n_medical_services'].values[0]*200]
        ax.pie(values, labels=labels, autopct='%1.1f%%', colors=['#5b86e5','#36d1c4','#f7b42c','#f7797d'])
        ax.set_title('Cost Contribution')
        st.pyplot(fig)

    st.markdown("""
    ---
    <div style='text-align:center; color:#888; font-size:1rem;'>
        <b>Modern Medical Cost Predictor</b> | No scrolling, all info at a glance. <br>
        <i>Design by GitHub Copilot</i>
    </div>
    """, unsafe_allow_html=True)
