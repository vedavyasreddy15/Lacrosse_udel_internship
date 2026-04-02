import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Lacrosse xG Simulator", layout="wide")
st.title("🥍 Lacrosse Expected Goals (xG) Simulator")
st.markdown("Adjust the player's position and mechanics to see how the mathematics of the shot change in real-time.")

# --- 2. LOAD THE ENGINE ---
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("lacrosse_xg_engine.json") # Ensure this file is in your GitHub repo!
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
    
    # --- 6. DRAW THE ADVANCED FIELD VISUALIZATION ---
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor('#4CAF50') # Grass Green
    
    # 6a. Draw the Crease and Goal Line
    ax.plot([-15, 15], [0, 0], color='white', linewidth=2, zorder=1) # Endline
    crease = plt.Circle((0, 0), 3, color='white', fill=False, linewidth=2, zorder=2)
    ax.add_patch(crease)
    
    # 6b. Draw the Realistic Net
    # The back of the net (triangle mesh)
    ax.fill([-1, 0, 1], [0, -2, 0], color='white', alpha=0.3, zorder=2)
    ax.plot([-1, 0, 1], [0, -2, 0], color='white', linestyle=':', linewidth=2, zorder=3)
    # The front orange pipes
    ax.plot([-1, 1], [0, 0], color='orange', linewidth=4, zorder=4)
    
    # Calculate player coordinates
    rad_angle = np.radians(angle)
    player_x = distance * np.sin(rad_angle)
    player_y = distance * np.cos(rad_angle)
    
    # 6c. Draw the "Bad Angle" Shooting Cone (Visible Net)
    # Shades the area from the player to the left post (-1) and right post (1)
    ax.fill([player_x, -1, 1], [player_y, 0, 0], color='yellow', alpha=0.25, zorder=3, label='Visible Net')
    ax.plot([player_x, -1], [player_y, 0], color='yellow', linewidth=1, alpha=0.8, zorder=4)
    ax.plot([player_x, 1], [player_y, 0], color='yellow', linewidth=1, alpha=0.8, zorder=4)
    
    # 6d. Draw the Trajectory Line (Black Dashed)
    ax.plot([player_x, 0], [player_y, 0], color='black', linestyle='--', linewidth=2, zorder=5)
    
    # 6e. Draw the "Human" Top-Down Player
    # Helmet (White with a blue stripe)
    ax.scatter(player_x, player_y, color='white', edgecolors='#00539F', s=250, zorder=7, linewidth=2)
    # Lacrosse Stick (Extending from player toward goal)
    stick_end_x = player_x - 1.5 * np.sin(rad_angle)
    stick_end_y = player_y - 1.5 * np.cos(rad_angle)
    ax.plot([player_x, stick_end_x], [player_y, stick_end_y], color='silver', linewidth=3, zorder=6)
    
    # Format the graph
    ax.set_xlim(-15, 15)
    ax.set_ylim(-3, 22)
    ax.set_aspect('equal')
    ax.axis('off')
    
    st.pyplot(fig)
    
    st.caption("*Notice how the yellow 'Visible Net' cone shrinks as the angle increases.*")
