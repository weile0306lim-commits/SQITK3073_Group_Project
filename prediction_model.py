import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================
# O3: Fuel Consumption Prediction
# Predict fuel consumption using controllable trip data
# =====================================================

df = pd.read_csv("cleaned_fuel_consumption_dataset.csv")

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)
print()

# =====================================================
# 1. Select Features and Target
# =====================================================

features = [
    "vehicle_type",
    "fuel_type",
    "distance_km",
    "payload_kg"
]

target = "fuel_l"

X = df[features]
y = df[target]

# =====================================================
# 2. Split Data
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# 3. Preprocessing
# =====================================================

categorical_features = [
    "vehicle_type",
    "fuel_type"
]

numeric_features = [
    "distance_km",
    "payload_kg"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# =====================================================
# 4. Build Model
# =====================================================

fuel_prediction_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

# =====================================================
# 5. Train Model
# =====================================================

fuel_prediction_model.fit(X_train, y_train)

# =====================================================
# 6. Predict
# =====================================================

y_pred = fuel_prediction_model.predict(X_test)

# =====================================================
# 7. Model Evaluation
# =====================================================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

o3_model_metrics = {
    "MAE": mae,
    "RMSE": rmse,
    "R2 Score": r2
}

print("===== O3 Model Performance =====")
for metric, value in o3_model_metrics.items():
    print(metric, ":", round(value, 4))
print()

# =====================================================
# 8. Prediction Results
# =====================================================

o3_prediction_results = X_test.copy()
o3_prediction_results["actual_fuel_l"] = y_test.values
o3_prediction_results["predicted_fuel_l"] = y_pred
o3_prediction_results["prediction_error_l"] = (
    o3_prediction_results["actual_fuel_l"]
    - o3_prediction_results["predicted_fuel_l"]
)

print("===== Prediction Results Sample =====")
print(o3_prediction_results.head())
print()

# =====================================================
# 9. Save Model and Results
# =====================================================

joblib.dump(fuel_prediction_model, "fuel_prediction_model.pkl")
joblib.dump(o3_model_metrics, "fuel_prediction_metrics.pkl")
o3_prediction_results.to_pickle("prediction_results.pkl")

print("Objective 3 completed successfully.")
print("Saved files:")
print("- fuel_prediction_model.pkl")
print("- fuel_prediction_metrics.pkl")
print("- prediction_results.pkl")