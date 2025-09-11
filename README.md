# Titanic Survival Prediction – Homework 1

This project was done for Homework 1. The goal was to practice working with data in Python and to test simple rules for predicting survival on the Titanic dataset. The focus was on trying different approaches and reflecting on what worked, not on getting the best possible accuracy.

## Approach
- Loaded the dataset (`train.csv`) with pandas and checked the structure, size, and missing values.
- Looked at correlations with a heat map and group survival rates by Sex, Class, and Embarked.
- Built simple **one-factor models**:
  - Sex-only (female = survived)
  - Class-only (1st class = survived)
  - Age-only (child under 16 = survived)
  - Fare-only (fare above mean = survived)
- Built **two-factor weighted models**:
  - Sex + Class (normalized class values)
  - Sex + Fare (normalized fare values with outlier clipping)
- Wrote a function to combine features into a weighted score with a cutoff of 0.5.
- Added a very simple training loop that adjusted weights whenever the model predicted incorrectly.

## Results
- **Sex-only** was the strongest predictor (~0.787 accuracy).
- Class (~0.679), Fare (~0.66), and Age (~0.63) were weaker on their own.
- Sex + Class and Sex + Fare mostly behaved like the Sex-only model. If the second factor had too much weight, accuracy dropped.
- The training loop improved slightly over several passes (~0.72 → ~0.74–0.75) but still did not beat the Sex-only baseline.

## Key Takeaways
- Sex is the strongest feature in this dataset.
- Adding Class, Age, or Fare in a simple weighted model did not improve accuracy much.
- Scaling and thresholds are important, and a single strong variable can dominate.
- The project was good practice with pandas, NumPy, simple rules, and basic training loops.

## Files
- `titanic.py` – main script with all models.
- `train.csv` – dataset (from Kaggle Titanic competition).
- `correlation_heatmap.png` – correlation heat map of numeric features.
