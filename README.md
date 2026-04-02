# 🥍 Lacrosse Expected Goals (xG) Predictive Engine

https://xgudelmenlacrosse.streamlit.app/
**Author:** Vedavyas Reddy Bommineni  
**Tech Stack:** Python, Pandas, XGBoost, SHAP, Streamlit

## 1. Project Objective
This project replaces subjective coaching "eye tests" with a mathematical probability engine. It calculates an objective **Expected Goals (xG)** metric (0% to 100%) for every shot taken, based strictly on the physical state of the field at the exact moment of release.

**Business Value for Coaching Staff:**
* Optimize offensive shot selection using hard data.
* Eliminate inefficient shooting habits.
* Mathematically prove which techniques actually result in goals.

---

## 2. Data Pipeline & Feature Engineering
The model was trained on roughly 1,500 historical shots. Strict feature selection was applied to isolate physics from subjective opinions.

### Base Features (The Raw Data)
* **`Shot_Distance`:** Distance to the goal in yards.
* **`Shot_Angle`:** Degrees off-center (maps the amount of visible net).
* **`Hands_Free`:** Did the shooter have free arms? (1 = Yes, 0 = No).
* **`Feet_Set`:** Was the shooter balanced and planted? (1 = Yes, 0 = No).

### Engineered Features (Lacrosse IQ)
Created to help the model find immediate synergies:
* **`Spatial_Danger` (`Distance` × `Angle`):** A single metric defining the geometric threat level.
* **`Shooter_Mechanics` (`Hands_Free` × `Feet_Set`):** A massive probability multiplier indicating perfect time-and-room shooting form.

### Transformed Features
* **`Type_of_Motion`:** One-Hot Encoded into binary columns (Overhand, Sidearm, Underhand) so the machine learning engine does not assume numerical hierarchy among shooting styles.

### Excluded Features
* **`Fantastic_Four`:** A subjective metric grading pre-shot passing sequences. 
  * *Why we dropped it:* **Data Leakage & Subjectivity.** xG measures the micro-physics of the shot itself. Fantastic Four measures macro-team tactics. Good passing sequences result in the player getting `Hands_Free` and `Feet_Set`. Including Fantastic Four would mathematically double-count these advantages and introduce human bias into a physics engine.

---

## 3. Model Architecture (XGBoost)
The engine utilizes **Extreme Gradient Boosting (XGBoost)**. Sports data is heavily imbalanced (missed shots vastly outnumber goals). The following hyperparameters were set to stabilize the learning curve:
* `scale_pos_weight = 2.53`: Forces the AI to study goals heavily by penalizing missed predictions.
* `max_depth = 3`: Restricts tree depth to prevent overfitting (the "Smiley Face" curve) on a smaller dataset.
* `learning_rate = 0.01`: Forces slow, highly-accurate steps during gradient descent.

---

## 4. Model Insights (SHAP Analysis)
The model was unpacked using **SHAP (SHapley Additive exPlanations)** to prove *why* the AI makes its decisions. The audit revealed three core truths:
1. **Distance is King:** `Shot_Distance` is the #1 undisputed driver of scoring probability.
2. **Mechanics Over Motion:** Securing `Hands_Free` is the #2 driver.
3. **Form is Irrelevant:** Hand motion (`Type_of_Motion`) has an average SHAP impact of 0.000. The math proves that how a player flicks their wrists is statistically irrelevant compared to their positioning and mechanics.

---

## 5. Interpreting the Output: The SHAP "Biggest Flaw" Quirk
The generated output file (`Tanner_Master_xG_Database.csv`) translates the complex SHAP math into plain-English columns for coaches: **Top Advantage** and **Biggest Flaw**.

**Note on Relative Minimums:**
Because SHAP values are relative, the script is instructed to find the highest (Advantage) and lowest (Flaw) mathematical impact for every shot. 

On highly lethal, mathematically perfect shots (e.g., 85%+ xG), almost every feature generates a massive positive score. To populate the `Biggest_Flaw` column, the engine is forced to flag the *lowest relative number*. Therefore, it will frequently flag a microscopic mathematical penalty—such as a `-0.001` deduction for a sidearm release—as the "Biggest Flaw." 

This is not a bug; it indicates the shot was executed so flawlessly that the engine had to scrape the bottom of the mathematical barrel to find a critique. If all SHAP values for a shot are strictly `0.0` or higher, the engine accurately outputs *"Nothing hurt."*
