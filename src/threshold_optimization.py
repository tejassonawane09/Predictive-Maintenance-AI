import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# --------------------------------
# 1. Load Dataset
# --------------------------------

df = pd.read_csv("data/raw/ai4i2020.csv")


# --------------------------------
# 2. Feature Engineering
# --------------------------------

df["Temperature Difference [K]"] = (
    df["Process temperature [K]"]
    - df["Air temperature [K]"]
)

df["Power Proxy"] = (
    df["Torque [Nm]"]
    * df["Rotational speed [rpm]"]
)


# --------------------------------
# 3. Features and Target
# --------------------------------

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


# --------------------------------
# 4. Train-Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------
# 5. Preprocessing
# --------------------------------

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


# --------------------------------
# 6. Random Forest
# --------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# --------------------------------
# 7. Train Model
# --------------------------------

model.fit(X_train, y_train)


# --------------------------------
# 8. Failure Probabilities
# --------------------------------

y_probability = model.predict_proba(X_test)[:, 1]


print("\n===== THRESHOLD ANALYSIS =====")


# --------------------------------
# 9. Test Different Thresholds
# --------------------------------

thresholds = [
    0.50,
    0.45,
    0.40,
    0.35,
    0.30,
    0.25,
    0.20,
    0.15,
    0.10
]

results = []


for threshold in thresholds:

    y_pred_threshold = (
        y_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred_threshold
    ).ravel()

    results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp
    })


# --------------------------------
# 10. Display Results
# --------------------------------

results_df = pd.DataFrame(results)

print(
    results_df.to_string(index=False)
)


# --------------------------------
# 11. Best F1 Threshold
# --------------------------------

best_row = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\n===== BEST F1 THRESHOLD =====")

print(
    f"Threshold : {best_row['Threshold']:.2f}"
)

print(
    f"Precision : {best_row['Precision']:.4f}"
)

print(
    f"Recall    : {best_row['Recall']:.4f}"
)

print(
    f"F1 Score  : {best_row['F1 Score']:.4f}"
)

print(
    f"FP        : {int(best_row['False Positives'])}"
)

print(
    f"FN        : {int(best_row['False Negatives'])}"
)


# --------------------------------
# 12. ROC-AUC
# --------------------------------

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print(
    f"\nROC-AUC   : {roc_auc:.4f}"
)