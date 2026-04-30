import streamlit as st
import pandas as pd
import joblib
from PIL import Image
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

# === Load models ===
rf_model = joblib.load('rf_model.joblib')  # Model Random Forest
svd_model = joblib.load('svd_model.joblib')  # Model SVD yang disimpan dengan joblib

# === Load dataset ===
df_rent = pd.read_csv('bukit-vista.csv')  # Dataset yang berisi properti, rating, dan user_id

# === Load logo BukitVista ===
logo = Image.open('logo_bukitvista.png')

# === Page config ===
st.set_page_config(
    page_title="Bukit Vista Rental Recommendation",
    page_icon=logo,
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Custom Bukit Vista Theme (CSS Injection) ===
st.markdown(
    """
    <style>
    body {
        background-color: #f5f8fd;
    }
    .stApp {
        background: linear-gradient(120deg, #2196F3 0%, #FFA726 100%);
        background-attachment: fixed;
    }
    .css-18e3th9, .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 20px;
    }
    h1 {
        color: #0D47A1;
        text-align: center;  /* Center align the header */
        font-size: 2.5rem;   /* Adjust font size */
        margin-top: 50px;    /* Add top margin for spacing */
        margin-bottom: 30px; /* Add bottom margin */
        font-weight: bold;
    }
    h2 {
        color: #1E88E5;
        text-align: center;
    }
    .stButton>button {
        background-color: #FFA726;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #fb8c00;
        color: white;
    }
    .css-10trblm {
        font-family: 'Arial', sans-serif;
    }
    /* Footer styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: linear-gradient(120deg, #2196F3 0%, #FFA726 100%);
        text-align: center;
        padding: 15px;
        border-top: 1px solid #e0e0e0;
        z-index: 100;
    }
    
    /* Space untuk footer */
    .main-container {
        margin-bottom: 80px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# === Header ===
col1, col2, col3 = st.columns([2, 6, 2])  # Tambah kolom kosong di kanan untuk balance
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
with col3:
    pass  # Kolom kosong untuk balance
# === Filter based on amenities ===
st.header("🔎 Personalized Rental Recommendations")

# Select amenities
amenities = ['Amazing View', 'Pool view', 'Ocean view', 'Amazing pool', 'Golfing', 'Beachfront', 'Jungle View', 'Island life']
selected_amenities = st.multiselect('Select desired amenities:', amenities)

# User ID input
user_id = st.number_input("Enter your User ID", min_value=1, value=1)

if st.button("Get Recommendations"):
    # 1. Check if user is new
    if user_id not in df_rent['user_id'].values:
        st.warning("⚠️ New user detected! Showing recommendations:")

        # Filter the dataset based on selected amenities
        filtered_df = df_rent.copy()
        for amenity in selected_amenities:
            if amenity in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[amenity] == 1]  # Assuming '1' means the amenity is available

        # Get properties that the user has not rated yet
        unrated_properties = filtered_df['name'].unique()

        # Predict ratings for unrated properties using the SVD model (simulate recommendations for a new user)
        predictions = []
        for property_name in unrated_properties:
            try:
                pred = svd_model.predict(user_id, property_name)
                predictions.append((property_name, pred.est))
            except Exception as e:
                st.error(f"Error predicting for {property_name}: {str(e)}")

        if not predictions:
            st.warning("No available properties to recommend.")
            st.stop()

        # Normalize scores to 1-5 scale
        min_score = min(p[1] for p in predictions)
        max_score = max(p[1] for p in predictions)

        # Display the top 10 recommendations for the new user
        st.subheader(f"🏖️ Top 10 Recommendations for User {user_id}")

        st.success("✅ Here are your personalized recommendations!")

        for name, score in sorted(predictions, key=lambda x: x[1], reverse=True)[:10]:
            stars = "⭐" * min(5, int(round(score)))
            score_text = f"{min(score, 5.0):.1f}/5"

            with st.container():
                st.markdown(
                    f"""
                    <div style="
                        padding: 16px;
                        border-radius: 8px;
                        margin: 8px 0;
                        background: linear-gradient(120deg, #BBDEFB 0%, #FFE0B2 100%);
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center">
                            <h4 style="margin: 0; color: #1a237e;">{name}</h4>
                            <div style="background: #ffffff; padding: 4px 8px; border-radius: 20px;">
                                {stars} <strong style="color: black; font-weight: bold;">{score_text}</strong>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # 2. For existing users, recommend based on SVD model
    else:
        # Filter the dataset based on selected amenities
        filtered_df = df_rent.copy()
        for amenity in selected_amenities:
            if amenity in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[amenity] == 1]  # Assuming '1' means the amenity is available

        # Get properties that the user has not rated yet
        rated_properties = df_rent[df_rent['user_id'] == user_id]['name'].unique()
        all_properties = filtered_df['name'].unique()
        unrated_properties = [prop for prop in all_properties if prop not in rated_properties]

        # Predict ratings for unrated properties using the SVD model
        predictions = []
        for property_name in unrated_properties:
            try:
                pred = svd_model.predict(user_id, property_name)
                predictions.append((property_name, pred.est))
            except Exception as e:
                st.error(f"Error predicting for {property_name}: {str(e)}")

        if not predictions:
            st.warning("No available properties to recommend.")
            st.stop()

        # Normalize scores to 1-5 scale
        min_score = min(p[1] for p in predictions)
        max_score = max(p[1] for p in predictions)

        # Display the top 10 recommendations for the existing user
        st.subheader(f"🏖️ Top 10 Recommendations for User {user_id}")

        st.success("✅ Here are your personalized recommendations!")

        for name, score in sorted(predictions, key=lambda x: x[1], reverse=True)[:10]:
            stars = "⭐" * min(5, int(round(score)))
            score_text = f"{min(score, 5.0):.1f}/5"

            with st.container():
                st.markdown(
                    f"""
                    <div style="
                        padding: 16px;
                        border-radius: 8px;
                        margin: 8px 0;
                        background: linear-gradient(120deg, #BBDEFB 0%, #FFE0B2 100%);
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center">
                            <h4 style="margin: 0; color: #1a237e;">{name}</h4>
                            <div style="background: #ffffff; padding: 4px 8px; border-radius: 20px;">
                                {stars} <strong style="color: black; font-weight: bold;">{score_text}</strong>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# === Footer ===
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        Made with ❤️ by Argana © 2025 | Bukit Vista Recommendation System
    </div>
    """,
    unsafe_allow_html=True
)
