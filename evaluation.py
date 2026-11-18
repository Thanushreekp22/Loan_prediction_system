from pathlib import Path
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parent
COLUMNS = [
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

# Feature columns for the improved KNN pipeline (Loan_ID excluded)
KNN_FEATURE_COLS = [c for c in COLUMNS if c != "Loan_ID"]


def evaluate_models():
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

    # ------------------------------------------------------------------
    # Train / test split (stratified, same random state for both models)
    # ------------------------------------------------------------------
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["Loan_Status"]
    )
    test_target = test_data["Loan_Status"]

    # ------------------------------------------------------------------
    # Decision Tree — original preprocessing (LabelEncoder, all columns)
    # ------------------------------------------------------------------
    dt_features = test_data[COLUMNS].copy()
    for column in dt_features.columns:
        if dt_features[column].dtype == "object":
            dt_features[column] = LabelEncoder().fit_transform(
                dt_features[column]
            )
    # Encode target to match the model's expected classes (0/1)
    le_target = LabelEncoder()
    test_target_encoded = le_target.fit_transform(test_target)

    # ------------------------------------------------------------------
    # KNN — new pipeline (raw features, Loan_ID excluded,
    #           Pipeline handles OneHotEncoding + StandardScaler)
    # ------------------------------------------------------------------
    knn_features = test_data[KNN_FEATURE_COLS].copy()

    models = {
        "Decision Tree": pickle.load(open(PROJECT_ROOT / "model_dt.pkl", "rb")),
        "KNN": pickle.load(open(PROJECT_ROOT / "model_knn.pkl", "rb")),
    }

    print("\n[MODEL EVALUATION] 20% stratified holdout test set")
    for name, model in models.items():
        if name == "Decision Tree":
            predictions = model.predict(dt_features)
            report_target = test_target_encoded
        else:
            predictions = model.predict(knn_features)
            report_target = test_target

        report = classification_report(
            report_target, predictions, output_dict=True, zero_division=0
        )
        matrix = confusion_matrix(report_target, predictions)

        print(f"\n[MODEL METRICS] {name}")
        print(f"  Accuracy:  {accuracy_score(report_target, predictions) * 100:.2f}%")
        pos_key = "Y" if "Y" in report else "1"
        print(f"  Precision: {report[pos_key]['precision'] * 100:.2f}%")
        print(f"  Recall:    {report[pos_key]['recall'] * 100:.2f}%")
        print(f"  F1 Score:  {report[pos_key]['f1-score'] * 100:.2f}%")
        print(f"  Confusion Matrix [[TN, FP], [FN, TP]]: {matrix.tolist()}")


if __name__ == "__main__":
    evaluate_models()
