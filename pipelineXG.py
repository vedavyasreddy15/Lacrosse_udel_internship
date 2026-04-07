import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
import time
from datetime import datetime

def run_production_pipeline():
    print("==================================================")
    print(f"🥍 LACROSSE MLOPS PIPELINE INITIATED | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("==================================================\n")
    start_time = time.time()

    # --- STEP 1: DATA INGESTION & PREPROCESSING ---
    print("[1/3] Extracting and Preprocessing Data...")
    
    # ⚠️ Update this path if Tanner gives you a new file!
    data_path = r"C:\Users\Vedav\Downloads\Master_xG_Dataset_Final.csv" 
    df = pd.read_csv(data_path)
    
    df['Type_of_Motion'] = df['Type_of_Motion'].str.lower().str.strip()
    for col in ['Goal', 'Hands_Free', 'Feet_Set', 'Challenged']:
        df[col] = df[col].astype(int)

    df['Spatial_Danger'] = df['Shot_Distance'] * df['Shot_Angle'].abs()
    df['Shooter_Mechanics'] = df['Hands_Free'] * df['Feet_Set']
    df['Type_of_Motion_over'] = (df['Type_of_Motion'] == 'over').astype(int)
    df['Type_of_Motion_side'] = (df['Type_of_Motion'] == 'side').astype(int)
    df['Type_of_Motion_under'] = (df['Type_of_Motion'] == 'under').astype(int)
    df['Type_of_Motion_unknown'] = (~df['Type_of_Motion'].isin(['over', 'side', 'under'])).astype(int)

    features = ['Shot_Distance', 'Shot_Angle', 'Hands_Free', 'Feet_Set', 'Challenged',
                'Spatial_Danger', 'Shooter_Mechanics', 'Type_of_Motion_over', 
                'Type_of_Motion_side', 'Type_of_Motion_under', 'Type_of_Motion_unknown']

    X = df[features]
    y = df['Goal']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- STEP 2: MODEL TRAINING (THE LOCKED BRAIN) ---
    print("[2/3] Forging Final Production Model...")
    
    # Apply the 3.0 "Megaphone" to fix the defensive bias
    weights = np.ones(len(y))
    weights[(X['Challenged'] == 1) & (y == 0)] = 3.0

    # The locked XGBoost architecture
    constraints = (-1, 0, 1, 1, -1, 0, 0, 0, 0, 0, 0)
    calibrated_weight = 1.9 # The winning number we found earlier

    final_model = xgb.XGBClassifier(
        n_estimators=500, 
        max_depth=3, 
        learning_rate=0.01, 
        scale_pos_weight=calibrated_weight, 
        random_state=42,
        monotone_constraints=constraints
    )
    
    final_model.fit(X_scaled, y, sample_weight=weights)

    # --- STEP 3: DEPLOYMENT ---
    print("[3/3] Saving Production Assets...")
    joblib.dump(final_model, r"C:\Users\Vedav\Downloads\xg.pkl")
    joblib.dump(scaler, r"C:\Users\Vedav\Downloads\scaler.pkl")

    elapsed_time = round(time.time() - start_time, 2)
    print("\n==================================================")
    print(f"✅ PIPELINE SUCCESSFUL | Executed in {elapsed_time} seconds")
    print("   New 'xg.pkl' and 'scaler.pkl' generated.")
    print("   Upload these to GitHub to update the live app!")
    print("==================================================")

if __name__ == "__main__":
    run_production_pipeline()
