import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# === Load models ===
rf_model = joblib.load('rf_model.joblib')
svd_data = joblib.load('svd_model (1).joblib')
U = svd_data['U']
Vt = svd_data['Vt']
user_to_idx = svd_data['user_to_idx']
item_to_idx = svd_data['item_to_idx']
global_mean = svd_data['global_mean']

def predict_rating(user_id, property_name):
    if user_id in user_to_idx and property_name in item_to_idx:
        u = user_to_idx[user_id]
        i = item_to_idx[property_name]
        score = float(U[u] @ Vt[:, i])
        return float(np.clip(score, 1, 5))
    return float(global_mean)

# Helper: predict rating
# === Load dataset ===
df_rent = pd.read_csv('bukit-vista.csv')

# === Load logo ===
logo = Image.open('logo_bukitvista.png')

# === Page config ===
st.set_page_config(
    page_title="Bukit Vista Rental Recommendation",
    page_icon=logo,
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CSS ===
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(120deg, #2196F3 0%, #FFA726 100%);
        background-attachment: fixed;
    }
    h1 { color: #0D47A1; text-align: center; font-size: 2.5rem; font-weight: bold; }
    h2 { color: #1E88E5; text-align: center; }
    .stButton>button {
        background-color: #FFA726; color: white;
        border-radius: 8px; height: 3em; width: 100%;
        font-weight: bold; border: none;
    }
    .stButton>button:hover { background-color: #fb8c00; }
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        text-align: center; padding: 15px;
        border-top: 1px solid #e0e0e0; z-index: 100;
    }
    </style>
""", unsafe_allow_html=True)

# === Header ===
col1, col2, col3 = st.columns([2, 6, 2])
with col1:
    st.image(logo, width=100)
    st.markdown("🌴 Empowering Dreams, Enriching Lives")
with col2:
    st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; height: 120px;">
            <h1 style="margin: 0; color: #0D47A1; font-size: 2.5rem; font-weight: bold;">
                Bukit Vista Rental Recommendation System
            </h1>
        </div>
    """, unsafe_allow_html=True)

# === Filter ===
st.header("🔎 Personalized Rental Recommendations")
amenities = ['Amazing View', 'Pool view', 'Ocean view', 'Amazing pool', 'Golfing', 'Beachfront', 'Jungle View', 'Island life']
selected_amenities = st.multiselect('Select desired amenities:', amenities)
user_id = st.number_input("Enter your User ID", min_value=1, value=1)

if st.button("Get Recommendations"):
    # Filter berdasarkan amenities
    filtered_df = df_rent.copy()
    for amenity in selected_amenities:
        if amenity in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[amenity] == 1]

    is_new_user = user_id not in df_rent['user_id'].values

    if is_new_user:
        st.warning("⚠️ New user detected! Showing popular recommendations:")
        unrated_properties = filtered_df['name'].unique()
    else:
        rated_properties = df_rent[df_rent['user_id'] == user_id]['name'].unique()
        all_properties = filtered_df['name'].unique()
        unrated_properties = [p for p in all_properties if p not in rated_properties]

    # Predict
    predictions = []
    for property_name in unrated_properties:
        score = predict_rating(user_id, property_name)
        predictions.append((property_name, score))

    if not predictions:
        st.warning("No available properties to recommend.")
        st.stop()

    st.subheader(f"🏖️ Top 10 Recommendations for User {user_id}")
    st.success("✅ Here are your personalized recommendations!")

    for name, score in sorted(predictions, key=lambda x: x[1], reverse=True)[:10]:
        stars = "⭐" * min(5, int(round(score)))
        score_text = f"{min(score, 5.0):.1f}/5"
        st.markdown(f"""
            <div style="padding:16px; border-radius:8px; margin:8px 0;
                background: linear-gradient(120deg, #BBDEFB 0%, #FFE0B2 100%);
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <div style="display:flex; justify-content:space-between; align-items:center">
                    <h4 style="margin:0; color:#1a237e;">{name}</h4>
                    <div style="background:#ffffff; padding:4px 8px; border-radius:20px;">
                        {stars} <strong style="color:black;">{score_text}</strong>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# === Footer ===
st.markdown("---")
st.markdown("""
    <div class="footer">
        Made with ❤️ by Argana © 2025 | Bukit Vista Recommendation System
    </div>
""", unsafe_allow_html=True)