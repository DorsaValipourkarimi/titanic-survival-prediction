# titanic-survival-prediction
COMP 379 - HW1
Titanic Survival Prediction – Homework 1
This project is for Homework 1 in COMP 379. The goal was to practice working with data in Python and to try simple ways of predicting survival on the Titanic dataset. The assignment was less about getting the best score and more about experimenting and reflecting on what works and what doesn’t.
##Approach
Loaded the dataset (train.csv) into a pandas DataFrame and converted it into NumPy arrays for some checks.
Looked at dataset structure, correlations, and group survival rates by Sex, Class, and Embarked.
Built simple one-factor models:
Sex-only (female = survived)
Class-only (1st class = survived)
Age-only (child under 16 = survived)
Fare-only (fare above mean = survived)
Built two-factor weighted models:
Sex + Class (normalized class values)
Sex + Fare (normalized fare values with outlier clipping)
Wrote a function to combine features into a weighted score, with a cutoff of 0.5 for predicting survival.
Added a basic training loop that adjusted weights whenever the model made a mistake.
##Results
Sex-only was the strongest predictor (~0.787 accuracy).
Class (~0.679), Fare (~0.66), and Age (~0.63) were weaker on their own.
Combining Sex with Class or Fare usually gave results very close to Sex-only. If the second feature was given too much weight, accuracy dropped.
The mistake-driven training loop raised accuracy slightly over epochs (~0.72 → ~0.74–0.75), but still did not beat Sex-only.
##Key Takeaways
Sex is the most important factor in this dataset.
Adding Class, Age, or Fare in a simple linear weighting scheme did not improve accuracy much.
Scaling and thresholds make a big difference, and one strong feature can dominate the model.
This project was good practice for using pandas, NumPy, simple weighting rules, and training loops.
Files
titanic.py – main script with all models (one-factor, two-factor, and training loop).
train.csv – training dataset (from Kaggle Titanic competition).
correlation_heatmap.png – saved plot of correlations between numeric features.
