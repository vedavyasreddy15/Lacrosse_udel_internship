# 🥍 Lacrosse Expected Goals (xG) Engine & MLOps Pipeline

https://xgudelmenlacrosse.streamlit.app/

## Project Overview
This repository contains a production-grade Expected Goals (xG) machine learning model and an automated MLOps pipeline built for Division 1 Lacrosse analytics. The project transitions raw, biased shot data into a fully calibrated, interactive web application used by coaching staffs to evaluate offensive performance and shot quality.

**Author:** Vedavyas Bommineni | MS in Data Science, University of Delaware

---

## 🧠 The Business Problem
In lacrosse, predicting whether a single shot will result in a goal is highly volatile due to goalie performance, defensive pressure, and luck. Binary predictions (Goal/No Goal) fail to capture the true rhythm of an offense. 

**The Objective:** Build a probabilistic model that assigns a mathematical Expected Goal (xG) value to every shot based on spatial geometry, shooter mechanics, and defensive pressure, allowing coaches to measure *shot quality* independently of the *shot outcome*.

---

## 📊 Data Challenges & Methodological Solutions

### 1. The "Elite Bias" Problem (Class Imbalance)
During initial exploratory data analysis (EDA), the model exhibited a severe bias, assigning a near 0.0% feature importance to defensive pressure (`Challenged`). 
* **The Cause:** A "loud minority" of elite players in the dataset were consistently scoring heavily contested shots, tricking the algorithm into assuming defense was irrelevant.
* **The Solution:** Implemented **Cost-Sensitive Learning**. I applied a custom sample weight multiplier (`3.0x`) specifically to missed, challenged shots. This mathematically forced the AI to respect the physical reality of defensive pressure, raising its feature importance to an accurate 12.6%.

### 2. Algorithmic Selection: Why XGBoost?
While testing complex architectures (including a Voting Classifier Tri-Engine combining Random Forest, Logistic Regression, and XGBoost), the ensemble model suffered from a "Dilution Effect," where weaker linear models dragged down the accuracy.
* I selected a pure **XGBoost Classifier** for the production environment. It provided superior hyperparameter tuning, avoided overfitting, and, most importantly, offered high **explainability** for the coaching staff compared to a black-box ensemble.

### 3. Hyperparameter Calibration
To prevent the model from over-predicting goal totals, an automated hyperparameter tuning script was built to calibrate the `scale_pos_weight`. This ensures the sum of the predicted probabilities perfectly matches the natural baseline scoring rate of the dataset.

---

## 🛡️ Validation & Stress Testing
To prove the model's stability against chaotic real-world variance, the engine was subjected to a rigorous **10-Shuffle Cross-Validation Stress Test**. 

* **Methodology:** 10 completely random, unstratified subsets of 153 shots (representing an average game volume) were hidden from the training data. The model was trained from scratch 10 times and forced to predict the total Expected Goals for the unseen 'vaults'.
* **Results:** Despite actual goals swinging wildly from 32 to 51 across the random samples, the model's xG remained remarkably stable, calculating the underlying shot quality with an **average error of only ~1.8 to 4.9 goals** per simulated game.

---

## ⚙️ The MLOps Pipeline
This project is built for continuous integration. Rather than manually retraining the model when new game data is collected, I engineered a master `pipeline.py` script.

**Features of the Pipeline:**
1. **Automated Ingestion:** Reads and preprocesses new `.csv` data, instantly engineering features like `Spatial_Danger` and `Shooter_Mechanics`.
2. **Self-Calibrating Training:** Applies the defensive sample weights and trains a fresh XGBoost brain.
3. **Asset Deployment:** Automatically exports the updated `xg.pkl` (model) and `scaler.pkl` (standardizer) for immediate deployment to the Streamlit frontend.

---

## 💻 Tech Stack
* **Machine Learning:** `XGBoost`, `Scikit-Learn`, `NumPy`
* **Data Manipulation:** `Pandas`
* **Deployment & UI:** `Streamlit`, `Matplotlib` (for spatial field rendering)
* **Environment:** Python 3.x

---

## 🚀 How to Run the Application

**1. Run the MLOps Pipeline (To train a fresh model):**
```bash
python pipeline.py
