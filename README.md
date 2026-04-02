# Lacrosse_udel_internship

Lacrosse Expected Goals (xG) Predictive Engine
Author: Vedavyas Reddy Bommineni
Tech Stack: Python, Pandas, XGBoost, SHAP, Streamlit

Project Objective
In professional and collegiate lacrosse, shot quality is traditionally evaluated using subjective "eye tests" and coaching biases. The objective of this project was to build a machine learning engine that calculates an objective Expected Goals (xG) metric—a mathematical probability (0-100%) of a shot scoring based strictly on the physical state of the field at the exact moment of release.

By translating raw gameplay data into actionable percentages, this engine allows coaching staffs to optimize offensive shot selection, eliminate inefficient habits, and mathematically prove which techniques actually win games.

Data Pipeline & Feature Engineering
The dataset consisted of roughly 1,500 historical shots. To optimize the model's accuracy, strict feature selection and engineering protocols were applied:

1. Base Features (Included)
Shot_Distance (Yards): The raw distance to the goal.

Shot_Angle (Degrees): The angle from the center of the goal, mapping the visible net.

Hands_Free (Binary): Did the shooter have their arms free from defenders? (1 = Yes, 0 = No)

Feet_Set (Binary): Was the shooter planted and balanced? (1 = Yes, 0 = No)

2. Excluded Features
Fantastic_Four: A subjective team metric tracking the number of successful criteria (passes/movement) completed prior to the shot. Reason for exclusion: Data leakage and scope mismatch. "Fantastic Four" measures macro-team tactics and passing, while xG measures micro-shot physics. Furthermore, a great passing sequence ultimately results in the player getting their hands free and feet set. Including it alongside the base features would result in mathematical double-counting.

3. Engineered Features (Lacrosse IQ)
To help the shallow decision trees identify synergies, we engineered two interaction terms:

Spatial_Danger (Distance × Angle): Combines distance and visible net to create a single threat metric.

Shooter_Mechanics (Hands_Free × Feet_Set): A lethal multiplier indicating a player was completely unimpeded with perfect form (e.g., a time-and-room step-down shot).

4. Transformed Features
Type_of_Motion (Overhand, Sidearm, Underhand): Machine learning models cannot process English text. This column was transformed using One-Hot Encoding into distinct binary columns (Type_of_Motion_over, Type_of_Motion_side, etc.) to prevent the model from assuming numerical hierarchy among shooting styles.

Model Architecture & Training
The engine was built using XGBoost (Extreme Gradient Boosting). Because sports data is inherently chaotic and heavily imbalanced (missed shots vastly outnumber goals at a ratio of 2.53 to 1), strict hyperparameters were implemented to stabilize the learning curve and prevent extreme variance (overfitting):

scale_pos_weight = 2.53: Forced the engine to study the minority class (Goals) by penalizing missed predictions.

max_depth = 3: Prevented the AI from building hyper-specific, illogical decision trees on a small dataset.

learning_rate = 0.01 & n_estimators = 1000: Forced the engine to take slow, careful steps during gradient descent.

subsample = 1.0 & colsample_bytree = 1.0: Disabled random data dropping, which acts as a stabilizer for smaller sports datasets.

Explainability (SHAP Analysis)
To ensure coaching buy-in, the model was unpacked using SHAP (SHapley Additive exPlanations). The SHAP audit definitively proved:

Distance is King: Shot_Distance was the #1 mathematical driver of scoring probability.

Mechanics Over Motion: Securing Hands_Free was the #2 driver.

Form is Irrelevant: The specific Type_of_Motion (Overhand vs. Underhand) had an average SHAP impact of 0.000. The math proved that hand motion is statistically irrelevant to scoring success compared to field positioning.
