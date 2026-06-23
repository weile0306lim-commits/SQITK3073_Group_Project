import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


# Read the fleet value dataset
df = pd.read_csv("fleet_value_dataset.csv")
print("\nOriginal Dataset shape: ", df.shape)

# Data Preprocessing
print("\nData is Cleaning ......")

target = "fuel_l"
features = ["vehicle_type", "distance_km", "payload_kg"]
df_new = df[features + [target]]
rows_before = df_new.shape[0]
df_new = df_new.dropna()
rows_after = df_new.shape[0]

print("Cleaned Dataset Shape: ", df_new.shape)
print("Rows removed: ", rows_before - rows_after, "rows")
print("Data cleaning completed.\n")

X = df_new[features]
y = df_new[target]

# preprocessing (one-hot encode vehicle_type, pass through numeric columns)
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["vehicle_type"]),
    ("num", "passthrough", ["distance_km", "payload_kg"])
    ]
)

# train-test split sizes
test_size_list = [0.1, 0.2, 0.3, 0.4]

# Store all model results
results = []

# Data Mining
for test_size in test_size_list:
    train_size = 1 - test_size
    split_ratio = str(int(train_size * 100)) + ":" + str(int(test_size * 100))

    print("Train-Test Split:", split_ratio)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Linear Regression (lr)
    print("Training Linear Regression......")

    lr_pipeline = Pipeline(steps=[("preprocessor", preprocessor),("model", LinearRegression())])
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)

    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    lr_r2 = r2_score(y_test, lr_pred)
    lr_mape = mean_absolute_percentage_error(y_test, lr_pred) * 100

    # Decision Tree (dt)
    print("Training Decision Tree......")

    dt_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", DecisionTreeRegressor(random_state=42))])
    dt_pipeline.fit(X_train, y_train)
    dt_pred = dt_pipeline.predict(X_test)

    dt_mae = mean_absolute_error(y_test, dt_pred)
    dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))
    dt_r2 = r2_score(y_test, dt_pred)
    dt_mape = mean_absolute_percentage_error(y_test, dt_pred) * 100

    # Random Forest (rf)
    print("Training Random Forest......")

    rf_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
    rf_pipeline.fit(X_train, y_train)
    rf_pred = rf_pipeline.predict(X_test)

    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)
    rf_mape = mean_absolute_percentage_error(y_test, rf_pred) * 100

    # Gradient Boosting (gb)
    print("Training Gradient Boosting......\n")

    gb_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", GradientBoostingRegressor(random_state=42))])
    gb_pipeline.fit(X_train, y_train)
    gb_pred = gb_pipeline.predict(X_test)

    gb_mae = mean_absolute_error(y_test, gb_pred)
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
    gb_r2 = r2_score(y_test, gb_pred)
    gb_mape = mean_absolute_percentage_error(y_test, gb_pred) * 100

    # Store results
    results.append(["Linear Regression", split_ratio, lr_mae, lr_rmse, lr_r2, lr_mape])
    results.append(["Decision Tree", split_ratio, dt_mae, dt_rmse, dt_r2, dt_mape])
    results.append(["Random Forest", split_ratio, rf_mae, rf_rmse, rf_r2, rf_mape])
    results.append(["Gradient Boosting", split_ratio, gb_mae, gb_rmse, gb_r2, gb_mape])

print("Model training completed.\n")


# Results Comparison
results_df = pd.DataFrame(results, columns=["Model", "Train-Test Split", "MAE", "RMSE", "R2 Score", "MAPE (%)"])
print(results_df)

results_df.to_csv("fuel_model_comparison_results.csv", index=False)

# Cross Validation
print("\nRunning Cross Validation......\n")

cv_fold_list = [2, 3, 5, 10]

cv_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

cv_scoring = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2"
}

cv_results = []

for k in cv_fold_list:
    print(f"Cross Validation with {k}-Fold......")

    for model_name, model in cv_models.items():
        cv_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        cv_scores = cross_validate(cv_pipeline, X, y, cv=k, scoring=cv_scoring, n_jobs=-1)

        mae_mean = -cv_scores["test_MAE"].mean()
        mae_std = cv_scores["test_MAE"].std()
        rmse_mean = -cv_scores["test_RMSE"].mean()
        rmse_std = cv_scores["test_RMSE"].std()
        r2_mean = cv_scores["test_R2"].mean()
        r2_std = cv_scores["test_R2"].std()

        cv_results.append([model_name, k, mae_mean, mae_std, rmse_mean, rmse_std, r2_mean, r2_std])

print("\nCross Validation completed.\n")

cv_results_df = pd.DataFrame(cv_results, columns=[
    "Model", "K-Fold",
    "MAE Mean", "MAE Std",
    "RMSE Mean", "RMSE Std",
    "R2 Mean", "R2 Std"
    ]
)

print("===== Cross Validation Results =====")
print(cv_results_df)
print()

cv_results_df.to_csv("fuel_cv_comparison_results.csv", index=False)

# Select Best Model (highest average R2 Mean across all K-fold settings tested)
cv_model_avg = (cv_results_df.groupby("Model")[["MAE Mean", "RMSE Mean", "R2 Mean"]].mean().sort_values(by="R2 Mean", ascending=False))

print("===== Average CV Performance per Model (across folds 2,3,5,10) =====")
print(cv_model_avg)
print()

best_model_name = cv_model_avg.index[0]

print("===== Best Model Selected (based on Cross Validation) =====")
print("Model         :", best_model_name)
print("Avg MAE       :", round(cv_model_avg.loc[best_model_name, "MAE Mean"], 4))
print("Avg RMSE      :", round(cv_model_avg.loc[best_model_name, "RMSE Mean"], 4))
print("Avg R2 Score  :", round(cv_model_avg.loc[best_model_name, "R2 Mean"], 4))
print()

# Map model name -> estimator (same hyperparameters used above)
model_lookup = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# Fuel Consumption Forecast Model (Final)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

fuel_forecast_model = Pipeline(steps=[("preprocessor", preprocessor), ("model", model_lookup[best_model_name])])
fuel_forecast_model.fit(X_train, y_train)
y_pred = fuel_forecast_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

forecast_metrics = {
    "Best Model": best_model_name,
    "MAE": round(mae, 4),
    "RMSE": round(rmse, 4),
    "R2 Score": round(r2, 4)
}

print("===== Fuel Forecast Model Evaluation (Final, 80:20) =====")
for key, value in forecast_metrics.items():
    print(key, ":", value)
print()

# Save Prediction Results
forecast_results = X_test.copy()
forecast_results["actual_fuel_l"] = y_test.values
forecast_results["forecasted_fuel_l"] = y_pred
forecast_results["forecast_error_l"] = (forecast_results["actual_fuel_l"] - forecast_results["forecasted_fuel_l"])
forecast_results["forecasted_fuel_l"] = forecast_results["forecasted_fuel_l"].round(2)
forecast_results["forecast_error_l"] = forecast_results["forecast_error_l"].round(2)

print("===== Forecast Results Sample =====")
print(forecast_results.head())
print()

# Save Outputs
joblib.dump(fuel_forecast_model, "fuel_forecast_model.pkl")
joblib.dump(forecast_metrics, "fuel_forecast_metrics.pkl")

forecast_results.to_csv("fuel_forecast_results.csv", index=False)
forecast_results.to_pickle("fuel_forecast_results.pkl")

print("Fuel forecast model completed successfully.")
print("Best model used:", best_model_name)
print("Saved files:")
print("- fuel_forecast_model.pkl")
print("- fuel_forecast_metrics.pkl")
print("- fuel_forecast_results.csv")
print("- fuel_forecast_results.pkl")
print("- fuel_model_comparison_results.csv  (all models x all splits)")