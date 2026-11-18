# Loan Prediction System — Project Detailed Information

**Purpose:** This document captures everything needed to understand, present, and defend the Loan Prediction System project — from problem statement to code internals, ML theory, evaluation, weaknesses, and mock-interview Q&A.

**Project location:** `c:\Users\thanu\Desktop\Projects\Loan_prediction_system-main`
**Key files:** `app.py`, `evaluation.py`, `inspect_models.py`, `Untitled.ipynb`, `train.csv`, `model_dt.pkl`, `model_knn.pkl`, `templates/index.html`, `requirements.txt`, `render.yaml`, `Snapshots/`

---

## Phase 1 — Project Understanding

### 1.1 Problem Statement
Build a web-based loan approval prediction system. A bank / lending institution receives loan applications with applicant details (income, credit history, education, employment type, property area, loan amount, etc.). The system must predict whether each application is likely to be **APPROVED** or **REJECTED**, show an approval-chance percentage, and provide human-readable explanation points (plus points / rejection reasons). It is a **binary classification** problem.

### 1.2 Dataset
- File: `train.csv`
- Source: standard Loan Prediction dataset (commonly used for ML practice).
- Target column: `Loan_Status` (values `Y` = Approved / `N` = Rejected; encoded to `1` / `0`).
- Rows: each row is one loan application.
- Columns (`COLUMNS` in `evaluation.py` and `app.py`):

| # | Feature | Type | Meaning |
|---|---------|------|---------|
| 1 | `Loan_ID` | Categorical / ID | Unique application identifier (kept in feature list but carries no predictive signal) |
| 2 | `Gender` | Categorical | Male / Female |
| 3 | `Married` | Categorical | Yes / No |
| 4 | `Dependents` | Categorical / Numeric | 0, 1, 2, 3+ |
| 5 | `Education` | Categorical | Graduate / Not Graduate |
| 6 | `Self_Employed` | Categorical | Yes / No |
| 7 | `ApplicantIncome` | Numeric | Applicant's monthly income |
| 8 | `CoapplicantIncome` | Numeric | Co-applicant's monthly income |
| 9 | `LoanAmount` | Numeric | Loan amount requested |
| 10 | `Loan_Amount_Term` | Numeric | Loan term (months) |
| 11 | `Credit_History` | Numeric (binary) | 1 = meets credit criteria, 0 = does not |
| 12 | `Property_Area` | Categorical | Urban / Semiurban / Rural |

Sample row: `LP001002,Male,No,0,Graduate,No,5849,0,,360,1,Urban,Y`

### 1.3 Architecture
```
User (browser)
   │  HTTP GET/POST
   ▼
┌─────────────┐
│  Flask app  │  app.py  (routes / and /predict, model loading, prediction logic)
└─────────────┘
   │  renders / session
   ▼
┌─────────────┐
│ index.html  │  Jinja2 template (form + result display)
└─────────────┘
   │
   ▼
┌─────────────────────────────┐
│ model_dt.pkl  (DecisionTree) │──► prediction + probability
│ model_knn.pkl (KNeighbors)  │──► prediction + probability
└─────────────────────────────┘
   │
   ▼
evaluation.py  (startup holdout evaluation on train.csv)
```

- Frontend: single-page Jinja2 template (`templates/index.html`) with a form, Font Awesome icons, datalists for income/loan suggestions, and result/insights sections.
- Backend: Flask (`Flask`, `render_template`, `request`, `redirect`, `url_for`, `session`).
- Models: two pickled scikit-learn classifiers (`DecisionTreeClassifier`, `KNeighborsClassifier`) loaded with `pickle`.
- Data handling: `pandas` DataFrames for building the prediction sample.

### 1.4 End-to-End Workflow
1. User opens `http://localhost:5000/` (GET `/`) → empty form rendered with an info message.
2. User fills the form (annual income in ₹, loan amount in ₹, categorical dropdowns) and submits → POST `/predict`.
3. `app.py`:
   - Reads raw form values as strings.
   - `parse_float()` converts each to a float or `None` if blank.
   - Validates: no field may be `None` (else redirect with "Please complete all fields").
   - Validates: input must not equal the hard-coded `default_form_data` (else redirect with warning to enter real details).
   - Converts units to model-raw units: income ÷ 12 (annual → monthly), `LoanAmount` ÷ 1000.
   - Builds a `pandas.DataFrame` with one row and the exact training column order (`columns` list, with `Loan_ID` placeholder = `0` first).
   - Both models predict class (`0`/`1`) and probability.
   - `approval_chance = round(((dt_prob + knn_prob) / 2) * 100, 2)`.
   - Final decision:
     - both `1` → "Loan is likely to be APPROVED"
     - both `0` → "Loan is likely to be REJECTED"
     - mixed → "Loan decision is borderline (manual review recommended)".
   - `build_explanations(form_data)` produces rule-based `rejection_reasons` and `plus_points` (credit history, income level, debt-to-income, tenure, dependents, education, self-employment, property area).
   - Stores everything in `session["last_view"]` and redirects to `/` (POST-redirect-GET pattern).
4. GET `/` pops `session["last_view"]` and renders `index.html` with prediction, score, insights, form values, or info message.
5. At server startup (`python app.py`), `evaluate_models()` runs once and prints holdout metrics to the console.

---

## Phase 2 — Python and Flask

### 2.1 Routes
| Route | Methods | Purpose |
|-------|---------|---------|
| `/` | GET | Home page. Pops `session["last_view"]`. If present, renders result/info/form values; otherwise renders empty form with default info message. |
| `/predict` | POST | Form submission handler. Parses, validates, converts units, builds DataFrame, runs both models, builds explanations, stores result in session, redirects to `/`. |

### 2.2 Form Handling
- `index.html` uses `<form method="POST" action="/predict">`.
- Each `<input>`/`<select>` has `name=` matching the feature names (`Gender`, `Married`, …, `Property_Area`) and `required` attribute.
- Categorical dropdowns already emit numeric strings (`0`/`1`/`2`), so the frontend does the encoding that the model expects (see weakness in Phase 5).
- Datalists (`applicant-income-options`, `coapplicant-income-options`, `loan-amount-options`) give ₹-denominated suggestions.
- `form_values` are passed back to the template so the form is repopulated after redirect (sticky form).
- The template checks `form_values.Credit_History == "1" or == "1.0"` etc. to re-select the right option.

### 2.3 Sessions
- `app.secret_key = "loan_app_dev_secret_key"` — signs the session cookie.
- Flask `session` is a client-side signed cookie.
- `session.pop("last_view", None)` implements POST-redirect-GET: the prediction result lives in the session for exactly one render of `/`, preventing resubmission on refresh.
- `session["last_view"]` dictionary keys: `prediction`, `approval_chance`, `rejection_reasons`, `plus_points`, `form_values`, `info_message`.

### 2.4 Key Functions in `app.py`
| Function | Role |
|----------|------|
| `parse_float(value)` | Converts form string → float. Returns `None` for `None`/empty/whitespace. Used for every field. |
| `get_model_probability(model, sample)` | Returns `model.predict_proba(sample)[0][1]` (probability of class 1) if available, else falls back to `float(model.predict(sample)[0])`. |
| `build_explanations(form_data)` | Rule-based explainer. Computes `total_income`, `annual_income`, `debt_to_income`, then appends reasons/points based on thresholds (credit_history < 1, income < 36000 / ≥ 84000, DTI > 4.5 / ≤ 3, tenure > 30 / ≤ 20, dependents > 2 / ≤ 1, education, self_employed, property_area). Returns `(rejection_reasons, plus_points)`. |
| `home()` | GET `/` handler. |
| `predict()` | POST `/predict` handler (see workflow in Phase 1.4). |

### 2.5 Pandas DataFrames
- `columns = ['Loan_ID','Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']`
- Prediction sample built as:
  ```python
  values = [0] + list(model_form_data.values())   # 0 = Loan_ID placeholder
  sample = pd.DataFrame([values], columns=columns)
  ```
- `evaluation.py` reads `train.csv` into a DataFrame, imputes, label-encodes all object columns, splits features/target.

### 2.6 Model Loading
```python
dt = pickle.load(open("model_dt.pkl", "rb"))
knn = pickle.load(open("model_knn.pkl", "rb"))
```
- Loaded once at module import. If files are missing, Flask fails to start.
- Both are scikit-learn objects serialized with `pickle`.

### 2.7 Prediction Flow (line-by-line essence)
```python
model_form_data = form_data.copy()
model_form_data["ApplicantIncome"] /= 12          # annual → monthly
model_form_data["CoapplicantIncome"] /= 12
model_form_data["LoanAmount"] /= 1000             # ₹ → model raw units
values = [0] + list(model_form_data.values())     # Loan_ID placeholder first
sample = pd.DataFrame([values], columns=columns)
dt_result  = int(dt.predict(sample)[0])           # 0 or 1
knn_result = int(knn.predict(sample)[0])
dt_prob = get_model_probability(dt, sample)       # P(class=1)
knn_prob = get_model_probability(knn, sample)
approval_chance = round(((dt_prob + knn_prob) / 2) * 100, 2)
```


---

## Phase 3 — Machine Learning Fundamentals

### 3.1 Learning Type & Task
- **Supervised learning**: yes — labeled dataset (`Loan_Status` is the target).
- **Task type**: binary classification (`Y`/`N` → `1`/`0`).
- Algorithms: **Decision Tree Classifier** and **K-Nearest Neighbors Classifier** (`n_neighbors=5`, default `minkowski` / Euclidean metric).

### 3.2 Encoding
- All categorical/object columns are encoded with `sklearn.preprocessing.LabelEncoder` (one shared `le` instance looped over columns in the notebook; fresh `LabelEncoder().fit_transform` per column in `evaluation.py`).
- Mapping is **ordinal / arbitrary**, not one-hot:
  - `Gender`: Female→0, Male→1 (alphabetical)
  - `Married`: No→0, Yes→1
  - `Dependents`: 0,1,2,3+ → encoded as strings in order seen
  - `Education`: Graduate→0/1, Not Graduate→1/0 (alphabetical → "Graduate" < "Not Graduate", so Graduate=0, Not Graduate=1)
  - `Self_Employed`: No→0, Yes→1
  - `Property_Area`: Rural→0, Semiurban→1, Urban→2
- `Credit_History` is already numeric `0`/`1`.
- Income, loan amount, term are already numeric.
- **Frontend mismatch:** the HTML already posts numeric strings (`0`/`1`/`2`), so the live app never re-applies LabelEncoder — it relies on the frontend having the same mapping as training. This is fragile (see Phase 5).

### 3.3 Imputation (Missing Value Handling)
`evaluation.py` and the notebook fill missing values before training:
- `LoanAmount` → `.mean()`
- `Loan_Amount_Term` → `.mean()`
- `Gender`, `Married`, `Dependents`, `Self_Employed`, `Credit_History` → `.mode()[0]`
- Note: `train.csv` preview shows `Dependents` contains `"3+"` (string), so `.mode()` returns a string and LabelEncoder handles it.

### 3.4 Scaling
- **No feature scaling is applied anywhere** (no `StandardScaler` / `MinMaxScaler`).
- This is a **weakness for KNN**, which is distance-based: `ApplicantIncome` (thousands) and `LoanAmount` (hundreds) dominate the distance calculation relative to binary `Credit_History`. KNN results are therefore biased toward high-magnitude numeric features.

### 3.5 Models
| Model | Sklearn class | Key hyperparameters (as saved) |
|-------|---------------|--------------------------------|
| Decision Tree | `DecisionTreeClassifier()` | defaults (no `max_depth`, `min_samples_leaf`, etc.) → prone to overfitting |
| KNN | `KNeighborsClassifier(n_neighbors=5)` | `k=5`, default Minkowski distance, no scaling |

`inspect_models.py` prints: model type, tree depth, number of leaves, `n_features_in_`, `feature_names_in_`, `classes_`, feature importances (DT), and `n_neighbors`, `metric`, `n_samples_fit_` (KNN).


---

## Phase 4 — Model Evaluation

### 4.1 Train-Test Split
`evaluation.py`:
```python
_, test_features, _, test_target = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target
)
```
- 80% train / 20% stratified holdout test.
- `stratify=target` preserves the approval/rejection ratio in the test set.
- `random_state=42` makes it reproducible.
- Only the **test** portion is used for scoring; training portion is not re-used (the pickled models were already trained in the notebook).

Notebook split (no stratify): `train_test_split(X, y, test_size=0.2, random_state=42)` — slightly different from `evaluation.py`, so notebook accuracy and startup metrics may differ.

### 4.2 Metrics Printed (per model)
- Accuracy: `accuracy_score(test_target, predictions)`
- Precision (class 1): `report['1']['precision']`
- Recall (class 1): `report['1']['recall']`
- F1-score (class 1): `report['1']['f1-score']`
- Confusion matrix: `confusion_matrix(test_target, predictions)` printed as `[[TN, FP], [FN, TP]]`

Notebook-reported accuracies (non-stratified split):
- Decision Tree: **~69.1%**
- KNN: **~61.8%**

Startup evaluation uses the stratified split, so actual printed numbers may differ slightly.

### 4.3 Confusion Matrix Interpretation
For class `1` = Approved:
- `TN` = rejected correctly predicted as rejected
- `FP` = rejected wrongly predicted as approved (false approval — credit risk for bank)
- `FN` = approved wrongly predicted as rejected (lost business opportunity)
- `TP` = approved correctly predicted as approved

### 4.4 Overfitting
- Decision Tree uses **default, unrestricted hyperparameters** (`DecisionTreeClassifier()` with no `max_depth`, `min_samples_split`, `min_samples_leaf`). On a small dataset it can memorize training noise → high training accuracy, lower test accuracy → overfitting.
- No cross-validation, no learning curves, no train-vs-test gap analysis is performed.

### 4.5 Data Leakage Risks
1. **LabelEncoder fit on the full dataset before split** — category mappings (and any information in rare categories) are derived from data that includes the test set. Technically mild leakage for encoding; safer to fit encoders on train only and transform test.
2. **Mean/mode imputation computed on the full dataset before split** — test-set values influence the fill values used for training data (and vice versa). Proper pipeline: impute statistics from train only, apply to test.
3. **`Loan_ID` included as a feature** — it is a unique identifier with no predictive power. Including it can let the model memorize ID patterns (especially Decision Tree) → leakage-like overfitting.
4. **Evaluation on the same `train.csv` used for training** — no fully independent held-out test set; reported accuracy is optimistic.
5. **No stratification mismatch check between notebook and `evaluation.py`** — notebook split lacks `stratify`, so notebook accuracy ≠ startup accuracy.

---

## Phase 5 — Actual Code Walkthrough

### 5.1 `app.py` — Line-by-Line Summary

**Lines 1–5 (imports):**
```python
from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import pandas as pd
from evaluation import evaluate_models
```
- Flask web framework, template rendering, request/session handling, redirection.
- `pickle` for model deserialization; `pandas` for the prediction DataFrame.
- `evaluate_models` imported so metrics print at startup.

**Lines 6–7 (app setup):**
```python
app = Flask(__name__)
app.secret_key = "loan_app_dev_secret_key"
```
- Secret key signs session cookies. Hard-coded dev key (weakness: should be env var).

**Lines 9–11 (load models):**
```python
dt = pickle.load(open("model_dt.pkl", "rb"))
knn = pickle.load(open("model_knn.pkl", "rb"))
```
- Loaded once at module import. Missing files → app crashes on start.

**Lines 14–16 (column contract):**
```python
columns = ['Loan_ID','Gender','Married','Dependents','Education',
           'Self_Employed','ApplicantIncome','CoapplicantIncome',
           'LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']
```
- Must match training feature order exactly, else `predict` misaligns features.

**Lines 18–44 (default / empty form data):**
- `default_form_data` = all numeric defaults (used to detect "user submitted defaults, not real data").
- `empty_form_data` = all empty strings (initial form render).

**Lines 47–53 (`parse_float`):**
- Safe string→float conversion, returns `None` on blank/invalid.

**Lines 56–61 (`get_model_probability`):**
- Returns `model.predict_proba(sample)[0][1]` (probability of class 1) if available, else falls back to `float(model.predict(sample)[0])`.

**Lines 64–120 (`build_explanations`):**
- Rule-based explanation engine with hard-coded thresholds. See Phase 2.4.

**Lines 123–134 (`home`, GET `/`):**
- Pops `session["last_view"]`, renders template with result/info/form.

**Lines 136–217 (`predict`, POST `/predict`):**
- Reads raw form, `parse_float` each field, validates completeness, validates not-default, converts units (income÷12, loan÷1000), builds DataFrame, runs both models, averages probability, determines verdict, builds explanations, stores in session, redirects.

**Lines 219–221 (`__main__`):**
```python
if __name__ == "__main__":
    evaluate_models()
    app.run(debug=True)
```
- Prints holdout metrics then starts Flask debug server.


### 5.2 `evaluation.py` — Line-by-Line Summary
- Imports `pathlib`, `pickle`, `pandas`, sklearn metrics / `train_test_split` / `LabelEncoder`.
- `COLUMNS` mirrors `app.py` exactly.
- `evaluate_models()`:
  1. Reads `train.csv`.
  2. Imputes missing values (mean for `LoanAmount`/`Loan_Amount_Term`; mode for categoricals).
  3. Label-encodes every object-dtype column (fresh encoder per column).
  4. `features = data[COLUMNS]`, `target = data["Loan_Status"]`.
  5. Stratified 80/20 split (`random_state=42`).
  6. Loads both pickled models.
  7. For each: predicts on test, prints accuracy, precision/recall/F1 for class 1, and confusion matrix `[[TN, FP], [FN, TP]]`.
- `if __name__ == "__main__": evaluate_models()` — can run standalone via `python evaluation.py`.

### 5.3 `inspect_models.py`
- Diagnostic script (not used by the app). Prints model type, tree depth, number of leaves, `n_features_in_`, `feature_names_in_`, `classes_`, feature importances (DT), and `n_neighbors`, `metric`, `n_samples_fit_` (KNN). Useful to verify the pickled artifacts before deployment.

### 5.4 `Untitled.ipynb` — Notebook Summary
- Cell 1 (executed): loads `train.csv`, checks missing values (all 0 in this cleaned copy), imputes (`LoanAmount`/`Loan_Amount_Term` mean; categoricals mode), Label-encodes all object columns, splits 80/20 (`random_state=42`, **no stratify**), trains `DecisionTreeClassifier()` and `KNeighborsClassifier(n_neighbors=5)`, prints accuracies (**DT ~69.1%**, **KNN ~61.8%**), and a sample prediction.
- Later cell: `ipywidgets` UI with `IntText` inputs for each feature and a "Predict Loan" button that runs both models on a one-row DataFrame — a prototype of the Flask predictor.
- The notebook is where the models were trained and pickled (`model_dt.pkl`, `model_knn.pkl`), though the pickle-dump cell output was not captured in the read.

### 5.5 Saved Models (`model_dt.pkl`, `model_knn.pkl`)
- Serialized scikit-learn estimators via `pickle`.
- Loaded by `app.py` (prediction) and `evaluation.py` (metrics).
- **Risk:** `pickle` can execute arbitrary code on load — never load `.pkl` from untrusted sources. Joblib is the sklearn-recommended alternative.

### 5.6 `templates/index.html`
- Single Jinja2 template. Shows:
  - info message (when no prediction yet),
  - form with all 11 inputs + `required`,
  - repopulated `form_values` after redirect,
  - result banner (approved/rejected/neutral) with Font Awesome icon,
  - Approval Chance %,
  - Possible Rejection Reasons list,
  - Profile Plus Points list.

### 5.7 Important Weaknesses & Corrections (Phase 5 mandate)

| # | Weakness | Why it matters | Suggested correction |
|---|----------|----------------|----------------------|
| 1 | No scaling for KNN | Distance-based KNN biased by high-magnitude income/loan features | Add `StandardScaler`/`MinMaxScaler` in a pipeline; save scaler with model |
| 2 | `LabelEncoder` fit on full data before split | Mild leakage; inconsistent if categories differ | Fit encoder on train only; save encoder; transform test/live input |
| 3 | Imputation stats from full data before split | Test values influence train fill values | Compute mean/mode on train only; apply to test |
| 4 | `Loan_ID` used as a feature | Unique ID → memorization/overfitting, no signal | Drop `Loan_ID` from features |
| 5 | Decision Tree defaults (unrestricted) | Overfits small data | Set `max_depth`, `min_samples_leaf`, `class_weight`; tune via GridSearchCV |
| 6 | Evaluation on same data used for training | Optimistic accuracy, no independent test | Hold out a separate test set or use cross-validation |
| 7 | Frontend hard-codes encoding; app never re-encodes | If dropdown values/training mapping drift, predictions silently break | Apply the saved `LabelEncoder` (or one-hot) in `app.py`, not HTML |
| 8 | Hard-coded `secret_key` | Session cookies weakly signed | Use `os.environ.get("SECRET_KEY")` with fallback |
| 9 | `pickle` model loading | Arbitrary code execution on untrusted `.pkl` | Use `joblib`; verify model provenance |
| 10 | Hard-coded explanation thresholds | Not derived from model; may contradict model output | Base explanations on feature importances / SHAP values |
| 11 | Debug mode in `__main__` | `debug=True` exposes debugger | Run via `gunicorn app:app` (as in `render.yaml`), `debug=False` |
| 12 | No error handling for model load / predict failure | App crashes if `.pkl` missing or corrupt | Wrap load/predict in try/except, return graceful error |


---

## Phase 6 — Target Technical Mock Interview

Below are likely interview questions grouped by theme, with **model answers grounded in this specific project**.

### A. Project Questions

**Q1. What is your project, and what problem does it solve?**
> A loan approval prediction web app built with Flask. It takes applicant details (income, credit history, education, employment, property area, loan amount) and predicts whether the loan will be approved or rejected, using a Decision Tree and KNN model, and shows an approval-chance percentage plus rule-based explanations.

**Q2. Walk me through the end-to-end flow from form submission to result.**
> 1. User submits POST to `/predict`. 2. `app.py` parses each field with `parse_float`. 3. Validates all fields are filled and not defaults. 4. Converts annual income ÷ 12 and loan amount ÷ 1000 to model units. 5. Builds a one-row pandas DataFrame in training column order (with `Loan_ID`=0 placeholder). 6. Both models predict class and probability. 7. Averages probabilities → approval chance. 8. Verdict: both agree → approved/rejected; disagree → borderline. 9. `build_explanations` adds rule-based plus points/rejection reasons. 10. Result stored in `session` and redirected to `/` for display (POST-redirect-GET).

**Q3. Why two models, and how is the final decision made?**
> Decision Tree and KNN capture different patterns (rule-based vs. distance-based). Each votes with a class and a probability. If both predict the same class, that is the verdict. If they disagree, it is labeled "borderline — manual review recommended." Approval chance is the mean of both class-1 probabilities × 100.

**Q4. How are categorical variables handled?**
> In training (`evaluation.py`/notebook), all object columns are Label-Encoded (e.g., Rural=0, Semiurban=1, Urban=2). In the live Flask app, the HTML `<select>` already posts numeric strings (`0`/`1`/`2`), so the app does not re-encode — it relies on the frontend mapping matching training.

**Q5. How are missing values handled?**
> `LoanAmount` and `Loan_Amount_Term` filled with mean; `Gender`, `Married`, `Dependents`, `Self_Employed`, `Credit_History` filled with mode. Done on the full dataset before splitting.

**Q6. What does `evaluation.py` do, and when does it run?**
> It loads `train.csv`, imputes, encodes, does a stratified 80/20 split, loads both pickled models, and prints accuracy, precision, recall, F1 (for class 1), and confusion matrix `[[TN, FP], [FN, TP]]`. It runs automatically when `app.py` starts (`if __name__ == "__main__"` block) and can also run standalone.

**Q7. What is the dataset schema?**
> `Loan_ID, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Loan_Status`. Target `Loan_Status` is `Y`/`N` → encoded `1`/`0`.

**Q8. How is the app deployed?**
> On Render via `render.yaml`: build command `pip install -r requirements.txt`, start command `gunicorn app:app`. Locally: `python app.py`.


### B. ML Scenarios

**Q9. Classification vs. regression — where does this project fit?**
> Classification. Target `Loan_Status` is discrete (Approved/Rejected = 1/0), so it is binary classification, not regression.

**Q10. Decision Tree vs. KNN — how do they differ, and which is better here?**
> Decision Tree: splits data by feature thresholds, produces interpretable rules, non-parametric, but overfits easily on small data (defaults unrestricted here). KNN: classifies by majority vote of the 5 nearest neighbors using distance; simple but sensitive to feature scale and computationally heavier at prediction. KNN accuracy (~61.8%) is lower than DT (~69.1%) in this project, partly because no scaling is applied, so KNN is distance-biased by income/loan magnitude.

**Q11. What is overfitting, and is it present here?**
> Overfitting = model learns training noise, high train accuracy but poor generalization. Likely yes for the Decision Tree because it uses `DecisionTreeClassifier()` with no `max_depth`/`min_samples_leaf`, and accuracy is measured on data drawn from the same `train.csv` used to train.

**Q12. What is data leakage, and where could it occur here?**
> Leakage = information from outside the training data (including the test set) enters model building. Here: (a) `LabelEncoder` and mean/mode imputation are fit on the full dataset before train/test split; (b) `Loan_ID` (unique identifier) is used as a feature; (c) evaluation uses the same `train.csv` the models were trained on, with no independent test set.

**Q13. Why stratified train-test split?**
> `stratify=target` keeps the Approved/Rejected ratio identical in train and test, so metrics are representative and not skewed by class imbalance. `random_state=42` makes it reproducible.

**Q14. Precision, recall, F1 — what do they mean for a bank?**
- Precision (of approved predictions): of loans predicted approved, how many truly are. High precision reduces bad-loan risk.
- Recall: of truly approvable loans, how many we correctly approve. High recall reduces lost business.
- F1: harmonic mean, balances both. For a bank, false approvals (FP) are costly → precision matters; but missing good applicants (FN) also matters → track F1.

**Q15. Confusion matrix `[[TN, FP], [FN, TP]]` — explain each cell for this use case.**
- TN: rejected application correctly predicted rejected.
- FP: rejected application wrongly predicted approved → bank takes bad credit risk.
- FN: approvable application wrongly predicted rejected → lost customer/business.
- TP: approvable application correctly predicted approved.

**Q16. Is scaling used? Should it be?**
> No scaling is applied anywhere. It should be, especially for KNN (distance-based), because `ApplicantIncome` (~thousands) dominates `Credit_History` (0/1) in distance calculations. A `StandardScaler` in a scikit-learn `Pipeline` would fix this.

**Q17. How would you improve the models?**
> Drop `Loan_ID`; add scaling; fit encoders/imputers on train only; restrict Decision Tree depth (`max_depth`, `min_samples_leaf`); use `Pipeline` + `GridSearchCV`/`RandomizedSearchCV`; evaluate on a held-out test set with cross-validation; consider class weights if imbalanced.


### C. Python Coding

**Q18. What does `session.pop("last_view", None)` do?**
> Removes `last_view` from the Flask session and returns it, or returns `None` if absent. It implements POST-redirect-GET: the prediction result is shown once after redirect, then cleared, so refreshing the page doesn't resubmit the form.

**Q19. What does `parse_float` return for blank input, and why?**
> `None`. The `predict` route checks `if any(v is None for v in form_data.values())` and redirects with "Please complete all fields" — so blanks are rejected before model inference.

**Q20. What is `get_model_probability` doing?**
> Tries `model.predict_proba(sample)[0][1]` (probability of class 1 / Approved). If the model lacks `predict_proba`, falls back to `float(model.predict(sample)[0])`. Both DT and KNN have `predict_proba`, so the probability path is used.

**Q21. How is the sample DataFrame built?**
> `values = [0] + list(model_form_data.values())` — prepends `0` as `Loan_ID` placeholder, then all 11 feature values, in the exact order of `columns`. `pd.DataFrame([values], columns=columns)` creates one row.

**Q22. What does `approval_chance = round(((dt_prob + knn_prob) / 2) * 100, 2)` do?**
> Averages the class-1 probability from both models, converts to percentage, rounds to 2 decimals.

**Q23. How does the template show sticky form values?**
> `form_values` dict is passed from the session to `render_template`. Each `<select>` option checks `{% if form_values.Credit_History == "1" ... %}selected{% endif %}` so the user's previous choice remains selected after redirect.

**Q24. Debug a scenario: user submits the form but gets "Please complete all fields" although they filled everything.**
> `parse_float` returns `None` for empty/whitespace. If any field is blank (or the browser sent an empty string despite `required`), `any(v is None ...)` is True → redirect with info message. Also check that `<select>` has a non-empty value selected and `name` attributes match exactly.

**Q25. Debug a scenario: prediction is always borderline or wrong.**
> Possible causes: (a) unit conversion mismatch (app sends annual income ÷ 12 but model expects monthly — verify training units); (b) feature order mismatch between `columns` and training; (c) frontend encoding differs from training LabelEncoder mapping; (d) models were trained on scaled data but app sends raw data (no scaler applied).


### D. Debugging & Follow-ups

**Q26. The app crashes on start if `.pkl` files are missing. How would you make it robust?**
> Wrap `pickle.load` in try/except, check `os.path.exists` at startup, return a 503 / clear error message instead of a traceback, and document that `model_dt.pkl` and `model_knn.pkl` must be present.

**Q27. `requirements.txt` pins `scikit-learn==1.7.2`. What if the model was pickled with an older sklearn?**
> `pickle` is version-sensitive; unpickling with a mismatched sklearn can fail or behave differently. Pin the exact training version, or retrain and re-pickle in the deployment environment.

**Q28. Follow-up: How would you add a new model (e.g., Random Forest)?**
> Train + pickle it in the notebook; load it in `app.py` alongside `dt` and `knn`; add its prediction/probability to the averaging logic (or weighted voting); update `evaluation.py` to score it too.

**Q29. Follow-up: How would you move explanations from hard-coded rules to model-driven?**
> Use feature importances (Decision Tree has `feature_importances_`) or SHAP values to identify which features drove a specific prediction, then generate reasons like "Credit history was the strongest factor" instead of static threshold messages.

**Q30. Follow-up: What would you change before putting this in production?**
> Independent train/validation/test split; `Pipeline` (impute → encode → scale → model); cross-validation and hyperparameter tuning; drop `Loan_ID`; save encoder/scaler with the model; apply encoding in the backend (not frontend); environment-based `secret_key`; `gunicorn` + `debug=False`; input validation/sanitization; `joblib` instead of `pickle`; logging and error handling; model/version metadata.

---

## Appendix — File Reference

| File | Role |
|------|------|
| `app.py` | Flask app: routes, form parsing, sessions, model loading, prediction, explanations |
| `evaluation.py` | Holdout evaluation metrics printed at startup |
| `inspect_models.py` | Diagnostic script: model metadata, depth, leaves, feature importances |
| `Untitled.ipynb` | Training notebook: imputation, LabelEncoder, train/test split, model training, widget prototype |
| `train.csv` | Training/evaluation dataset (13 columns incl. target) |
| `model_dt.pkl` | Pickled Decision Tree classifier |
| `model_knn.pkl` | Pickled KNN classifier (`k=5`) |
| `templates/index.html` | Jinja2 frontend: form + results + insights |
| `requirements.txt` | Flask, pandas, scikit-learn==1.7.2, gunicorn |
| `render.yaml` | Render deployment config (build + start commands) |
| `Snapshots/` | Screenshots of empty form, borderline, and approved states |

