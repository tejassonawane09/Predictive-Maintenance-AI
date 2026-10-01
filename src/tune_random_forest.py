import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
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
# 3. Select Features
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
# 6. Pipeline
# --------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# --------------------------------
# 7. Hyperparameter Search Space
# --------------------------------

param_distributions = {
    "classifier__n_estimators": [
        100, 200, 300, 500
    ],

    "classifier__max_depth": [
        None, 5, 10, 15, 20
    ],

    "classifier__min_samples_split": [
        2, 5, 10
    ],

    "classifier__min_samples_leaf": [
        1, 2, 4
    ],

    "classifier__max_features": [
        "sqrt",
        "log2",
        None
    ],

    "classifier__class_weight": [
        None,
        "balanced"
    ]
}


# --------------------------------
# 8. Cross Validation
# --------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# --------------------------------
# 9. Randomized Search
# --------------------------------

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=30,
    scoring="f1",
    cv=cv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)


# --------------------------------
# 10. Train Search
# --------------------------------

search.fit(X_train, y_train)


# --------------------------------
# 11. Best Parameters
# --------------------------------

print("\n===== BEST PARAMETERS =====")

for parameter, value in search.best_params_.items():
    print(f"{parameter}: {value}")


print("\n===== BEST CROSS-VALIDATION F1 =====")
print(f"{search.best_score_:.4f}")


# --------------------------------
# 12. Final Test Evaluation
# --------------------------------

best_model = search.best_estimator_

y_pred = best_model.predict(X_test)

y_probability = best_model.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# --------------------------------
# 13. Results
# --------------------------------

print("\n===== TUNED RANDOM FOREST =====")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))


print("\n===== CLASSIFICATION REPORT =====")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)