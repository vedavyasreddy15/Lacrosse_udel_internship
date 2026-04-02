import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Lacrosse xG Simulator", layout="wide", initial_sidebar_state="expanded")
st.title("🥍 Lacrosse Expected Goals (xG) Simulator")
st.markdown("Adjust the player's position and mechanics to see how the mathematics of the shot change in real-time.")

# --- 2. LOAD THE ENGINE ---
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("lacrosse_xg_engine.json") 
    return model

xg_model = load_model()

# --- 3. THE DASHBOARD UI ---
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Shot Parameters")
    
    st.markdown("### Location")
    distance = st.slider("Shot Distance (Yards)", min_value=1.0, max_value=20.0, value=10.0, step=0.5)
    angle = st.slider("Shot Angle (Degrees)", min_value=-80, max_value=80, value=0, step=1)
    
    st.markdown("### Mechanics")
    hands_free = st.checkbox("Hands Free?", value=True)
    feet_set = st.checkbox("Feet Set?", value=True)
    
    st.markdown("### Motion")
    motion = st.selectbox("Type of Motion", ["Overhand", "Sidearm", "Underhand", "Unknown"])

# --- 4. PREPARE THE MATH ---
hands_val = 1 if hands_free else 0
feet_val = 1 if feet_set else 0
spatial_danger = distance * abs(angle)
shooter_mechanics = hands_val * feet_val

is_over = 1 if motion == "Overhand" else 0
is_side = 1 if motion == "Sidearm" else 0
is_under = 1 if motion == "Underhand" else 0
is_unk = 1 if motion == "Unknown" else 0

input_data = pd.DataFrame([[
    distance, abs(angle), hands_val, feet_val, 
    spatial_danger, shooter_mechanics, 
    is_over, is_side, is_under, is_unk
]], columns=[
    'Shot_Distance', 'Shot_Angle', 'Hands_Free', 'Feet_Set', 
    'Spatial_Danger', 'Shooter_Mechanics', 
    'Type_of_Motion_over', 'Type_of_Motion_side', 
    'Type_of_Motion_under', 'Type_of_Motion_unknown'
])

# --- 5. CALCULATE PROBABILITY ---
probability = xg_model.predict_proba(input_data)[0][1] * 100

with col2:
    st.header("Model Output")
    st.metric(label="Expected Goals (xG) Probability", value=f"{probability:.1f}%")
    
    # --- 6. DRAW THE HIGH-FIDELITY FIELD VISUALIZATION ---
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Deep Turf Green Background
    ax.set_facecolor('#2E7D32') 
    
    # Field Boundaries (Crisp White)
    ax.plot([-20, 20], [0, 0], color='#FFFFFF', linewidth=3, zorder=1) # Endline
    ax.plot([-20, -20], [-5, 25], color='#FFFFFF', linewidth=3, zorder=1) # Left Sideline
    ax.plot([20, 20], [-5, 25], color='#FFFFFF', linewidth=3, zorder=1) # Right Sideline
    ax.plot([-20, 20], [20, 20], color='#FFFFFF', linewidth=2, zorder=1, alpha=0.7) # Restraining Line
    
    # Crease
    crease = plt.Circle((0, 0), 3, color='#FFFFFF', fill=False, linewidth=2, zorder=2)
    ax.add_patch(crease)
    
    # Realistic Net (Mesh and Pipes)
    ax.fill([-1, 0, 1], [0, -2, 0], color='#FFFFFF', alpha=0.15, zorder=2) # Translucent mesh
    ax.plot([-1, 0, 1], [0, -2, 0], color='#FFFFFF', linestyle=':', linewidth=1.5, zorder=3)
    ax.plot([-1, 1], [0, 0], color='#FF5722', linewidth=5, zorder=4) # Bright orange pipes
    
    # Calculate player coordinates
    rad_angle = np.radians(angle)
    player_x = distance * np.sin(rad_angle)
    player_y = distance * np.cos(rad_angle)
    
    # The "Visible Net" Cone (Vibrant Gold Shading)
    ax.fill([player_x, -1, 1], [player_y, 0, 0], color='#FFD54F', alpha=0.35, zorder=3)
    ax.plot([player_x, -1], [player_y, 0], color='#FFD54F', linewidth=1.5, alpha=0.9, zorder=4)
    ax.plot([player_x, 1], [player_y, 0], color='#FFD54F', linewidth=1.5, alpha=0.9, zorder=4)
    
    # The Trajectory Line (Dashed Dark Charcoal)
    ax.plot([player_x, 0], [player_y, 0], color='#212121', linestyle='--', linewidth=2.5, zorder=5)
    
    # The Human Player (Helmet)
    ax.scatter(player_x, player_y, color='#FFFFFF', edgecolors='#0D47A1', s=350, zorder=7, linewidth=2.5)
    
    # The Lacrosse Stick (Sleek Silver pointing at goal)
    if distance > 0:
        dir_x = -player_x / distance
        dir_y = -player_y / distance
        stick_end_x = player_x + dir_x * 1.8
        stick_end_y = player_y + dir_y * 1.8
        ax.plot([player_x, stick_end_x], [player_y, stick_end_y], color='#B0BEC5', linewidth=4, zorder=6, solid_capstyle='round')
    
    # Format the graph limits to show the whole offensive zone cleanly
    ax.set_xlim(-22, 22)
    ax.set_ylim(-4, 23)
    ax.set_aspect('equal')
    ax.axis('off') # Clean borderless look
    
    # Ensure layout is tight and aesthetic
    plt.tight_layout()
    
    st.pyplot(fig)
