import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# ==============================
# LOAD MODEL & DATA
# ==============================
rf_model = joblib.load('rf_model.joblib')  # kalau masih dipakai
model = joblib.load('svd_model_fixed.joblib')

svd_data = model["predictions"]
user_item_matrix = model["user_item"]

df_rent = pd.read_csv('bukit-vista.csv')
logo = Image.open('logo_bukitvista.png')

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Bukit Vista Rental Recommendation",
    page_icon=logo,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# CSS
# ==============================
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

# ==============================
# HEADER
# ==============================
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

# ==============================
# FILTER
# ==============================
st.header("🔎 Personalized Rental Recommendations")

amenities = [
    'Amazing View', 'Pool view', 'Ocean view',
    'Amazing pool', 'Golfing', 'Beachfront',
    'Jungle View', 'Island life'
]

selected_amenities = st.multiselect('Select desired amenities:', amenities)
user_id = st.number_input("Enter your User ID", min_value=1, value=1)

# ==============================
# BUTTON ACTION
# ==============================
if st.button("Get Recommendations"):

    # Filter amenities
    filtered_df = df_rent.copy()
    for amenity in selected_amenities:
        if amenity in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[amenity] == 1]

    # Check new user
    is_new_user = user_id not in svd_data.index

    # ==========================
    # NEW USER → POPULAR ITEMS
    # ==========================
    if is_new_user:
        st.warning("⚠️ New user detected! Showing popular recommendations:")

        recommendations = (
            filtered_df.groupby('name')['rating']
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

    # ==========================
    # EXISTING USER → SVD
    # ==========================
    else:
        rated_properties = df_rent[df_rent['user_id'] == user_id]['name'].unique()
        all_properties = filtered_df['name'].unique()

        unrated_properties = [p for p in all_properties if p not in rated_properties]

        # Filter item yang ada di model
        valid_items = [p for p in unrated_properties if p in svd_data.columns]

        if not valid_items:
            st.warning("No available properties to recommend.")
            st.stop()

        recommendations = (
            svd_data.loc[user_id][valid_items]
            .sort_values(ascending=False)
            .head(10)
        )

    # ==========================
    # DISPLAY
    # ==========================
    st.subheader(f"🏖️ Top Recommendations for User {user_id}")
    st.success("✅ Here are your personalized recommendations!")

    for i, (name, score) in enumerate(recommendations.items(), 1):
        stars = "⭐" * min(5, int(round(score)))
        score_text = f"{min(score, 5.0):.1f}/5"

        st.markdown(f"""
            <div style="padding:16px; border-radius:8px; margin:8px 0;
                background: linear-gradient(120deg, #BBDEFB 0%, #FFE0B2 100%);
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <div style="display:flex; justify-content:space-between; align-items:center">
                    <h4 style="margin:0; color:#1a237e;">
                        {i}. {name}
                    </h4>
                    <div style="background:#ffffff; padding:4px 8px; border-radius:20px;">
                        {stars} <strong style="color:black;">{score_text}</strong>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown("""
    <div class="footer">
        Made with ❤️ by Argana © 2025 | Bukit Vista Recommendation System
    </div>
""", unsafe_allow_html=True)