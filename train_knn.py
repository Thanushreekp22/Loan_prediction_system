"""
Train an improved K-Nearest Neighbors model for loan approval prediction.

Improvements over the original model:
1. Feature scaling with StandardScaler (KNN is distance-based).
2. Loan_ID excluded from input features (identifier only, not predictive).
3. OneHotEncoder for nominal categorical features (no artificial ordering).
4. K value tuned via cross-validation (K in {3, 5, 7, 9, 11, 13, 15}).
5. Complete preprocessing + model saved as a single scikit-learn Pipeline
   (model_knn.pkl) for Flask compatibility — no manual scaling/encoding needed.

The original model is backed up as model_knn_original.pkl for comparison.
"""
from pathlib import Path
import pickle
import warnings

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent

ALL_COLUMNS = [
    "Loan_ID",
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]

KNN_FEATURE_COLS = [c for c in ALL_COLUMNS if c != "Loan_ID"]

NUMERIC_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

K_VALUES = [3, 5, 7, 9, 11, 13, 15]


def load_and_clean_data() -> pd.DataFrame:
    """Load train.csv and fill missing values using the same strategy as the
    original project (mean for numeric, mode for categorical)."""
    data = pd.read_csv(PROJECT_ROOT / "train.csv")

    data["LoanAmount"] = data["LoanAmount"].fillna(data["LoanAmount"].mean())
    data["Loan_Amount_Term"] = data["Loan_Amount_Term"].fillna(
        data["Loan_Amount_Term"].mean()
    )

    data["Gender"] = data["Gender"].fillna(data["Gender"].mode()[0])
    data["Married"] = data["Married"].fillna(data["Married"].mode()[0])
    data["Dependents"] = data["Dependents"].fillna(data["Dependents"].mode()[0])
    data["Self_Employed"] = data["Self_Employed"].fillna(
        data["Self_Employed"].mode()[0]
    )
    data["Credit_History"] = data["Credit_History"].fillna(
        data["Credit_History"].mode()[0]
    )

    return data


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer that:
    - Scales numerical features with StandardScaler
    - One-hot encodes nominal categorical features
    """
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    return preprocessor


def tune_k(X_train, y_train, preprocessor, k_values=K_VALUES, cv=5):
    """Try different K values using cross-validation on the training data.
    Returns the best K and a dict of {k: mean_cv_score}."""
    results = {}
    best_k = None
    best_score = -1.0

    for k in k_values:
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        mean_score = scores.mean()
        results[k] = mean_score
        print(f"  K={k:2d}  CV Accuracy: {mean_score * 100:.2f}%  (+/- {scores.std() * 100:.2f})")
        if mean_score > best_score:
            best_score = mean_score
            best_k = k

    return best_k, results


def evaluate_model(model, X_test, y_test, label: str):
    """Print classification metrics for a model on the test set.
    Works with both integer (0/1) and string ('Y'/'N') targets."""
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions)

    # Determine the positive-class key ('Y' for string targets, '1' for int targets)
    pos_key = "Y" if "Y" in report else "1"

    print(f"\n[MODEL METRICS] {label}")
    print(f"  Accuracy:  {accuracy_score(y_test, predictions) * 100:.2f}%")
    print(f"  Precision: {report[pos_key]['precision'] * 100:.2f}%")
    print(f"  Recall:    {report[pos_key]['recall'] * 100:.2f}%")
    print(f"  F1 Score:  {report[pos_key]['f1-score'] * 100:.2f}%")
    print(f"  Confusion Matrix [[TN, FP], [FN, TP]]: {matrix.tolist()}")
    return predictions


def main():
    print("=" * 60)
    print("🚀 TRAINING IMPROVED KNN MODEL")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load and clean data
    # ------------------------------------------------------------------
    data = load_and_clean_data()
    print(f"\n📊 Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")

    # ------------------------------------------------------------------
    # 2. Evaluate the ORIGINAL KNN model for comparison
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("📋 EVALUATING ORIGINAL KNN MODEL (model_knn_original.pkl)")
    print("-" * 60)

    original_model_path = PROJECT_ROOT / "model_knn_original.pkl"
    if original_model_path.exists():
        with open(original_model_path, "rb") as f:
            original_knn = pickle.load(f)

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        dt_features = data[ALL_COLUMNS].copy()
        for col in dt_features.columns:
            if dt_features[col].dtype == "object":
                dt_features[col] = LabelEncoder().fit_transform(dt_features[col])

        target_encoded = le.fit_transform(data["Loan_Status"])

        _, test_features_orig, _, test_target = train_test_split(
            dt_features, target_encoded,
            test_size=0.2, random_state=42, stratify=data["Loan_Status"]
        )

        evaluate_model(original_knn, test_features_orig, test_target,
                       "Original KNN (K=5, LabelEncoded, no scaling)")
    else:
        print("  ⚠️  Original model backup not found — skipping comparison.")
        _, _, _, test_target = train_test_split(
            data[ALL_COLUMNS], data["Loan_Status"],
            test_size=0.2, random_state=42, stratify=data["Loan_Status"]
        )

    # ------------------------------------------------------------------
    # 3. Prepare data for the improved KNN pipeline
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("🔧 PREPARING DATA FOR IMPROVED KNN PIPELINE")
    print("-" * 60)

    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["Loan_Status"]
    )

    X_train = train_data[KNN_FEATURE_COLS]
    y_train = train_data["Loan_Status"]
    X_test = test_data[KNN_FEATURE_COLS]
    y_test = test_data["Loan_Status"]

    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples:     {X_test.shape[0]}")
    print(f"  Feature columns (Loan_ID excluded): {KNN_FEATURE_COLS}")
    print(f"  Numerical features (StandardScaler): {NUMERIC_FEATURES}")
    print(f"  Categorical features (OneHotEncoder): {CATEGORICAL_FEATURES}")

    # ------------------------------------------------------------------
    # 4. Build preprocessor
    # ------------------------------------------------------------------
    preprocessor = build_preprocessor()

    # ------------------------------------------------------------------
    # 5. Tune K via cross-validation
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("🔍 TUNING K VALUE (CROSS-VALIDATION ON TRAINING DATA)")
    print("-" * 60)
    best_k, cv_results = tune_k(X_train, y_train, preprocessor)
    print(f"\n✅ Best K: {best_k} (CV Accuracy: {cv_results[best_k] * 100:.2f}%)")

    # ------------------------------------------------------------------
    # 6. Train final pipeline with best K
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("🏋️  TRAINING FINAL PIPELINE")
    print("-" * 60)

    final_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("knn", KNeighborsClassifier(n_neighbors=best_k)),
    ])
    final_pipeline.fit(X_train, y_train)
    print(f"  Pipeline trained with K={best_k}")

    # ------------------------------------------------------------------
    # 7. Evaluate improved model on holdout test set
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("📊 EVALUATING IMPROVED KNN MODEL (HOLDOUT TEST SET)")
    print("-" * 60)
    evaluate_model(final_pipeline, X_test, y_test,
                   f"Improved KNN (K={best_k}, Scaled, OneHotEncoded)")

    # ------------------------------------------------------------------
    # 8. Save the complete pipeline
    # ------------------------------------------------------------------
    output_path = PROJECT_ROOT / "model_knn.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(final_pipeline, f)
    print(f"\n💾 Improved model pipeline saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()