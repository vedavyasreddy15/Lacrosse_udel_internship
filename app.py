import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Lacrosse xG Engine", layout="wide")
st.title("🥍 Lacrosse xG Simulator (Production)")

# Caching temporarily disabled to ensure the new model loads instantly
def load_assets():
    model = joblib.load("xg.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

ensemble_model, scaler = load_assets()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Shot Parameters")
    distance = st.slider("Shot Distance (Yards)", 1.0, 20.0, 10.0, 0.5)
    angle = st.slider("Shot Angle (Degrees)", -80, 80, 0, 1)
    
    st.markdown("### Mechanics & Pressure")
    hands_free = st.checkbox("Hands Free?", value=True)
    feet_set = st.checkbox("Feet Set?", value=True)
    challenged = st.checkbox("Defensive Challenge?", value=False)
    
    st.markdown("### Motion")
    motion = st.selectbox("Type of Motion", ["Overhand", "Sidearm", "Underhand", "Unknown"])

# --- PURE DATA PREP (No Multiplier Hacks Needed) ---
hands_val = 1 if hands_free else 0
feet_val = 1 if feet_set else 0
challenged_val = 1 if challenged else 0

spatial_danger = distance * abs(angle)
shooter_mechanics = hands_val * feet_val

is_over = 1 if motion == "Overhand" else 0
is_side = 1 if motion == "Sidearm" else 0
is_under = 1 if motion == "Underhand" else 0
is_unk = 1 if motion == "Unknown" else 0

raw_data = np.array([[
    distance, abs(angle), hands_val, feet_val, challenged_val,
    spatial_danger, shooter_mechanics, 
    is_over, is_side, is_under, is_unk
]])

scaled_data = scaler.transform(raw_data)
probability = ensemble_model.predict_proba(scaled_data)[0][1] * 100

with col2:
    st.header("Model Output")
    st.metric(label="Expected Goals (xG)", value=f"{probability:.1f}%")
    
    # Visual Field Plotting
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_facecolor('#2E7D32') 
    ax.plot([-20, 20], [0, 0], color='white', linewidth=3, zorder=1)
    ax.add_patch(plt.Circle((0, 0), 3, color='white', fill=False, linewidth=2, zorder=2))
    ax.fill([-1, 0, 1], [0, -2, 0], color='white', alpha=0.15, zorder=2)
    ax.plot([-1, 1], [0, 0], color='#FF5722', linewidth=5, zorder=4)
    
    rad_angle = np.radians(angle)
    px, py = distance * np.sin(rad_angle), distance * np.cos(rad_angle)
    
    ax.fill([px, -1, 1], [py, 0, 0], color='#FFD54F', alpha=0.3, zorder=3)
    ax.plot([px, 0], [py, 0], color='black', linestyle='--', linewidth=2, zorder=5)
    ax.scatter(px, py, color='white', edgecolors='#0D47A1', s=350, zorder=7, linewidth=2)
    if distance > 0:
        ax.plot([px, px - (px/distance)*1.8], [py, py - (py/distance)*1.8], color='#B0BEC5', linewidth=4, zorder=6)

    ax.set_xlim(-22, 22); ax.set_ylim(-4, 23); ax.set_aspect('equal'); ax.axis('off')
    st.pyplot(fig)
