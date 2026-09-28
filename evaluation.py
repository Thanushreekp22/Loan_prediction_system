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

    for column in data.columns:
        if data[column].dtype == "object":
            data[column] = LabelEncoder().fit_transform(data[column])

    features = data[COLUMNS]
    target = data["Loan_Status"]
    _, test_features, _, test_target = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    models = {
        "Decision Tree": pickle.load(open(PROJECT_ROOT / "model_dt.pkl", "rb")),
        "KNN": pickle.load(open(PROJECT_ROOT / "model_knn.pkl", "rb")),
    }

    print("\n[MODEL EVALUATION] 20% stratified holdout test set")
    for name, model in models.items():
        predictions = model.predict(test_features)
        report = classification_report(
            test_target, predictions, output_dict=True, zero_division=0
        )
        matrix = confusion_matrix(test_target, predictions)

        print(f"\n[MODEL METRICS] {name}")
        print(f"  Accuracy:  {accuracy_score(test_target, predictions) * 100:.2f}%")
        print(f"  Precision: {report['1']['precision'] * 100:.2f}%")
        print(f"  Recall:    {report['1']['recall'] * 100:.2f}%")
        print(f"  F1 Score:  {report['1']['f1-score'] * 100:.2f}%")
        print(f"  Confusion Matrix [[TN, FP], [FN, TP]]: {matrix.tolist()}")


if __name__ == "__main__":
    evaluate_models()
