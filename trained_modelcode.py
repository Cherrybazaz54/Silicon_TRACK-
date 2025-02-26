
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
df = pd.read_csv('updated_logic_depth_dataset.csv')

# Check data
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData description:")
print(df.describe())

# Split data
X = df.drop('logic_depth', axis=1)
y = df['logic_depth']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest with hyperparameter tuning
rf = RandomForestRegressor(random_state=42)
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='r2', n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

# XGBoost with hyperparameter tuning
xgb = XGBRegressor(random_state=42, objective='reg:squarederror')
xgb_params = {
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0]
}

xgb_grid = GridSearchCV(xgb, xgb_params, cv=5, scoring='r2', n_jobs=-1)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_

# Evaluation function
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\n{name} Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    return y_pred, r2, rmse

# Evaluate both models
rf_pred, rf_r2, rf_rmse = evaluate_model(best_rf, X_test, y_test, "Random Forest")
xgb_pred, xgb_r2, xgb_rmse = evaluate_model(best_xgb, X_test, y_test, "XGBoost")

# Feature importance visualization
def plot_feature_importance(importances, features, title):
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), features[indices], rotation=45)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()

# Plot importances
rf_importances = best_rf.feature_importances_
xgb_importances = best_xgb.feature_importances_
features = X.columns

plot_feature_importance(rf_importances, features, "Random Forest Feature Importance")
plot_feature_importance(xgb_importances, features, "XGBoost Feature Importance")

# Comparison plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_pred, alpha=0.3, label='Random Forest')
plt.scatter(y_test, xgb_pred, alpha=0.3, label='XGBoost')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Model Predictions vs True Values')
plt.legend()
plt.show()

print("\nBest Random Forest Parameters:", rf_grid.best_params_)
print("Best XGBoost Parameters:", xgb_grid.best_params_)
