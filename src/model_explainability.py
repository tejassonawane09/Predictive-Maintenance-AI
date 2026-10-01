import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split


# =========================
# Load data and model
# =========================

DATA_PATH = "data/raw/ai4i2020.csv"
MODEL_PATH = "models/predictive_maintenance_model.pkl"

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)


# =========================
# Feature engineering
# =========================

df["Temperature Difference [K]"] = (
    df["Process temperature [K]"] - df["Air temperature [K]"]
)

df["Power Proxy"] = (
    df["Torque [Nm]"] * df["Rotational speed [rpm]"]
)


# =========================
# Select model features
# =========================

features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temperature Difference [K]",
    "Power Proxy"
]

X = df[features]


# =========================
# Get feature names
# =========================

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()

importances = classifier.feature_importances_


# =========================
# Create feature importance
# =========================

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


# =========================
# Display results
# =========================

print("\n===== Feature Importance =====\n")
print(importance_df.to_string(index=False))


# =========================
# Save feature importance
# =========================

importance_df.to_csv(
    "artifacts/feature_importance.csv",
    index=False
)


# =========================
# Plot top features
# =========================

top_features = importance_df.head(10).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Feature Importance - Random Forest")

plt.tight_layout()

plt.savefig(
    "artifacts/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFeature importance chart saved successfully.")