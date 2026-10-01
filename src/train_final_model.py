import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# 1. Load Dataset
# -----------------------------

df = pd.read_csv("data/raw/ai4i2020.csv")


# -----------------------------
# 2. Feature Engineering
# -----------------------------

df["Temperature Difference [K]"] = (
    df["Process temperature [K]"]
    - df["Air temperature [K]"]
)

df["Power Proxy"] = (
    df["Torque [Nm]"]
    * df["Rotational speed [rpm]"]
)


# -----------------------------
# 3. Features
# -----------------------------

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
y = df["Machine failure"]


# -----------------------------
# 4. Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------
# 5. Preprocessing
# -----------------------------

categorical_features = ["Type"]

numerical_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temperature Difference [K]",
    "Power Proxy"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# -----------------------------
# 6. Final Random Forest
# -----------------------------

classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)


# -----------------------------
# 7. Pipeline
# -----------------------------

final_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ]
)


# -----------------------------
# 8. Train Final Model
# -----------------------------

final_model.fit(X_train, y_train)


# -----------------------------
# 9. Save Model
# -----------------------------

joblib.dump(
    final_model,
    "models/predictive_maintenance_model.pkl"
)

print("Final model trained successfully.")
print("Model saved to:")
print("models/predictive_maintenance_model.pkl")

print("\nApplication threshold: 0.40")