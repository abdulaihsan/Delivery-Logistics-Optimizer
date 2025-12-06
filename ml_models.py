import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def get_top_categories(df, col, n=10):
    """Helper function to get the top N most frequent categories from a column."""
    return df[col].value_counts().head(n).index.tolist()

df = pd.read_csv("UberDataset (1).csv")

df = df.dropna(subset=['START_DATE', 'END_DATE', 'MILES'])

df['PURPOSE'] = df['PURPOSE'].fillna('Unknown')

df['START_DATE'] = pd.to_datetime(df['START_DATE'], errors='coerce')
df['END_DATE'] = pd.to_datetime(df['END_DATE'], errors='coerce')

df = df.dropna(subset=['START_DATE', 'END_DATE'])

print("Data cleaning and date conversion complete.")

df['TRAVEL_TIME_MIN'] = (df['END_DATE'] - df['START_DATE']).dt.total_seconds() / 60.0

df['START_HOUR'] = df['START_DATE'].dt.hour
df['DAY_OF_WEEK'] = df['START_DATE'].dt.dayofweek # 0=Monday, 6=Sunday

df = df[df['TRAVEL_TIME_MIN'] > 0]
df = df[df['MILES'] > 0]

print(f"Feature engineering complete. Target 'TRAVEL_TIME_MIN' created.")
print(f"Dataset shape after cleaning: {df.shape}")

numeric_features = ['MILES', 'START_HOUR', 'DAY_OF_WEEK']

top_starts = get_top_categories(df, 'START', n=20)
top_stops = get_top_categories(df, 'STOP', n=20)

df['START_LOC'] = df['START'].apply(lambda x: x if x in top_starts else 'Other')
df['STOP_LOC'] = df['STOP'].apply(lambda x: x if x in top_stops else 'Other')

categorical_features = ['CATEGORY', 'PURPOSE', 'START_LOC', 'STOP_LOC']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

encoded_cats = encoder.fit_transform(df[categorical_features])

encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out())

df.reset_index(drop=True, inplace=True)
encoded_df.reset_index(drop=True, inplace=True)

X_numeric = df[numeric_features]
X_final = pd.concat([X_numeric, encoded_df], axis=1)

y = df['TRAVEL_TIME_MIN']
X = X_final

print(f"Encoding complete. Final features shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

# --- Train Model 1: Linear Regression (Baseline) ---
print("\nTraining Linear Regression model...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression training complete.")

# --- Train Model 2: Random Forest (Proposed) ---
print("\nTraining Random Forest Regressor model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest training complete.")

print("\n\n--- Progress Report 2: Model Evaluation ---")
print("---------------------------------------------")
print(f"Baseline Model: Linear Regression")
print(f"  RMSE: {rmse_lr:.2f} minutes")
print(f"  R2 Score: {r2_lr:.4f}")
print("\n")
print(f"Proposed Model: Random Forest Regressor")
print(f"  RMSE: {rmse_rf:.2f} minutes")
print(f"  R2 Score: {r2_rf:.4f}")
print("---------------------------------------------")

print("\n--- Sample Predictions vs. Actual (Linear Regression) ---")
lr_predictions_df = pd.DataFrame({
    'Actual Travel Time (min)': y_test.head(),
    'Predicted Travel Time (min)': y_pred_lr[:5]
})
print(lr_predictions_df.round(2).to_string())

print("\n--- Sample Predictions vs. Actual (Random Forest) ---")
rf_predictions_df = pd.DataFrame({
    'Actual Travel Time (min)': y_test.head(),
    'Predicted Travel Time (min)': y_pred_rf[:5]
})
print(rf_predictions_df.round(2).to_string())

def generate_interpretability_report(model, X_train, X_test):
    """
    Generates SHAP (SHapley Additive exPlanations) plots to explain
    the Random Forest model's predictions.
    """
    print("\n--- Generating Interpretability Report (SHAP) ---")
    
    # 1. Create a Tree Explainer for the Random Forest
    explainer = shap.TreeExplainer(model)
    
    # 2. Calculate SHAP values (using a sample of test data for speed)
    sample_X = X_test.iloc[:100]
    shap_values = explainer.shap_values(sample_X)
    
    print("SHAP values calculated. Generating plots...")

    # 3. Summary Plot (Global Interpretability)
    plt.figure()
    shap.summary_plot(shap_values, sample_X, show=False)
    plt.title("Feature Importance (SHAP Summary)")
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png')
    print("Saved: shap_summary_plot.png")
    plt.close()

    # 4. Dependence Plot (Feature Interaction)
    if 'MILES' in sample_X.columns and 'START_HOUR' in sample_X.columns:
        plt.figure()
        shap.dependence_plot(
            "MILES", 
            shap_values, 
            sample_X, 
            interaction_index="START_HOUR", 
            show=False
        )
        plt.title("Dependence Plot: Miles vs. Start Hour")
        plt.tight_layout()
        plt.savefig('shap_dependence_miles.png')
        print("Saved: shap_dependence_miles.png")
        plt.close()

    print("Interpretability report generated.")


#Model 3 (additional) Logarithimic Regression
X_train_log = X_train.copy()
X_test_log = X_test.copy()
X_train_log['MILES'] = np.log(X_train['MILES'] + 1)
X_test_log['MILES'] = np.log(X_test['MILES'] + 1)
log_model = LinearRegression()
log_model.fit(X_train_log, y_train)
y_pred_log = log_model.predict(X_test_log)


if __name__ == "__main__":
    try:
        generate_interpretability_report(rf_model, X_train, X_test)
    except NameError:
        print("Skipping SHAP report (library not installed or import failed).")
    except Exception as e:
        print(f"An error occurred generating SHAP report: {e}")