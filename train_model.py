import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib


df = pd.read_csv("diet_recommendations_female_only.csv")


X = df[[
    "Age",
    "BMI",
    "Cholesterol_mg/dL",
    "Blood_Pressure_mmHg",
    "Glucose_mg/dL",
    "Weekly_Exercise_Hours",
    "Dietary_Nutrient_Imbalance_Score"
]]


y = df["Diet_Recommendation"]


model = RandomForestClassifier()
model.fit(X, y)


joblib.dump(model, "diet_model.pkl")

print("Diet model trained successfully.")