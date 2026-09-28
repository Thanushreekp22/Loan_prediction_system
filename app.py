from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import pandas as pd
from evaluation import evaluate_models

app = Flask(__name__)
app.secret_key = "loan_app_dev_secret_key"

# Load models
dt = pickle.load(open("model_dt.pkl", "rb"))
knn = pickle.load(open("model_knn.pkl", "rb"))

# Column names (same as training)
columns = ['Loan_ID','Gender','Married','Dependents','Education',
           'Self_Employed','ApplicantIncome','CoapplicantIncome',
           'LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']

default_form_data = {
    "Gender": 1.0,
    "Married": 1.0,
    "Dependents": 0.0,
    "Education": 1.0,
    "Self_Employed": 1.0,
    "ApplicantIncome": 0.0,
    "CoapplicantIncome": 0.0,
    "LoanAmount": 0.0,
    "Loan_Amount_Term": 0.0,
    "Credit_History": 1.0,
    "Property_Area": 0.0,
}

empty_form_data = {
    "Gender": "",
    "Married": "",
    "Dependents": "",
    "Education": "",
    "Self_Employed": "",
    "ApplicantIncome": "",
    "CoapplicantIncome": "",
    "LoanAmount": "",
    "Loan_Amount_Term": "",
    "Credit_History": "",
    "Property_Area": "",
}


def parse_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    return float(value)


def get_model_probability(model, sample):
    # Fallback to class prediction if probability is not available.
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(sample)[0][1]
        return float(prob)
    return float(model.predict(sample)[0])


def build_explanations(form_data):
    rejection_reasons = []
    plus_points = []

    applicant_income = form_data["ApplicantIncome"]
    coapplicant_income = form_data["CoapplicantIncome"]
    total_income = applicant_income + coapplicant_income
    loan_amount = form_data["LoanAmount"]
    loan_term_years = form_data["Loan_Amount_Term"]
    dependents = form_data["Dependents"]
    credit_history = form_data["Credit_History"]
    education = form_data["Education"]
    self_employed = form_data["Self_Employed"]
    property_area = form_data["Property_Area"]

    annual_income = max(total_income * 12, 1)
    debt_to_income = loan_amount / annual_income

    if credit_history < 1:
        rejection_reasons.append("Credit history is weak, which strongly affects approval.")
    else:
        plus_points.append("Good credit history improves repayment confidence.")

    if total_income < 36000:
        rejection_reasons.append("Total household income is relatively low.")
    elif total_income >= 84000:
        plus_points.append("Strong combined income supports repayment ability.")

    if debt_to_income > 4.5:
        rejection_reasons.append("Requested loan amount is high compared to total income.")
    elif debt_to_income <= 3:
        plus_points.append("Loan amount is well aligned with income level.")

    if loan_term_years > 30:
        rejection_reasons.append("Long repayment tenure can increase lending risk.")
    elif loan_term_years <= 20:
        plus_points.append("Shorter loan tenure generally reduces risk.")

    if dependents > 2:
        rejection_reasons.append("Higher number of dependents can reduce disposable income.")
    elif dependents <= 1:
        plus_points.append("Fewer dependents can help maintain repayment capacity.")

    if education == 1:
        plus_points.append("Graduate education profile is viewed positively in many cases.")
    else:
        rejection_reasons.append("Non-graduate profile may receive extra scrutiny.")

    if self_employed == 1:
        rejection_reasons.append("Self-employment income may be considered less stable.")
    else:
        plus_points.append("Salaried profile is often considered more stable.")

    if property_area in (1, 2):
        plus_points.append("Property location has favorable market profile.")

    return rejection_reasons, plus_points


@app.route('/')
def home():
    last_view = session.pop("last_view", None)

    if last_view:
        return render_template("index.html", **last_view)

    return render_template(
        "index.html",
        info_message="Fill all fields and click Predict Loan Status.",
        form_values=empty_form_data,
    )

@app.route('/predict', methods=['POST'])
def predict():
    raw_form_data = {
        "Gender": request.form.get("Gender", "").strip(),
        "Married": request.form.get("Married", "").strip(),
        "Dependents": request.form.get("Dependents", "").strip(),
        "Education": request.form.get("Education", "").strip(),
        "Self_Employed": request.form.get("Self_Employed", "").strip(),
        "ApplicantIncome": request.form.get("ApplicantIncome", "").strip(),
        "CoapplicantIncome": request.form.get("CoapplicantIncome", "").strip(),
        "LoanAmount": request.form.get("LoanAmount", "").strip(),
        "Loan_Amount_Term": request.form.get("Loan_Amount_Term", "").strip(),
        "Credit_History": request.form.get("Credit_History", "").strip(),
        "Property_Area": request.form.get("Property_Area", "").strip(),
    }

    form_data = {
        "Gender": parse_float(raw_form_data["Gender"]),
        "Married": parse_float(raw_form_data["Married"]),
        "Dependents": parse_float(raw_form_data["Dependents"]),
        "Education": parse_float(raw_form_data["Education"]),
        "Self_Employed": parse_float(raw_form_data["Self_Employed"]),
        "ApplicantIncome": parse_float(raw_form_data["ApplicantIncome"]),
        "CoapplicantIncome": parse_float(raw_form_data["CoapplicantIncome"]),
        "LoanAmount": parse_float(raw_form_data["LoanAmount"]),
        "Loan_Amount_Term": parse_float(raw_form_data["Loan_Amount_Term"]),
        "Credit_History": parse_float(raw_form_data["Credit_History"]),
        "Property_Area": parse_float(raw_form_data["Property_Area"]),
    }

    if any(v is None for v in form_data.values()):
        session["last_view"] = {
            "info_message": "Please complete all fields before prediction.",
            "form_values": raw_form_data,
        }
        return redirect(url_for("home"))

    if form_data == default_form_data:
        session["last_view"] = {
            "info_message": "Default values detected. Please enter your real details to get a meaningful prediction.",
            "form_values": raw_form_data,
        }
        return redirect(url_for("home"))

    model_form_data = form_data.copy()
    model_form_data["ApplicantIncome"] /= 12
    model_form_data["CoapplicantIncome"] /= 12
    model_form_data["LoanAmount"] /= 1000

    values = [0] + list(model_form_data.values())
    
    sample = pd.DataFrame([values], columns=columns)
    
    dt_result = int(dt.predict(sample)[0])
    knn_result = int(knn.predict(sample)[0])
    dt_prob = get_model_probability(dt, sample)
    knn_prob = get_model_probability(knn, sample)
    approval_chance = round(((dt_prob + knn_prob) / 2) * 100, 2)

    rejection_reasons, plus_points = build_explanations(form_data)
    
    if dt_result == 1 and knn_result == 1:
        result = "Loan is likely to be APPROVED"
    elif dt_result == 0 and knn_result == 0:
        result = "Loan is likely to be REJECTED"
    else:
        result = "Loan decision is borderline (manual review recommended)"

    if "REJECTED" in result and not rejection_reasons:
        rejection_reasons.append("Your current profile details may not meet the lender's approval criteria.")
        rejection_reasons.append("Try improving credit history, reducing loan amount, or increasing documented income.")
    if "APPROVED" in result and not plus_points:
        plus_points.append("Overall profile is acceptable for approval.")
    
    session["last_view"] = {
        "prediction": result,
        "approval_chance": approval_chance,
        "rejection_reasons": rejection_reasons,
        "plus_points": plus_points,
        "form_values": raw_form_data,
    }
    return redirect(url_for("home"))

if __name__ == "__main__":
    evaluate_models()
    app.run(debug=True)