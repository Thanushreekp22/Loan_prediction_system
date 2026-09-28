# Loan Approval Prediction System

This project is a Flask-based machine learning web app that predicts whether a loan application is likely to be approved. The prediction is made from applicant details such as income, credit history, education, employment type, and property area.

The app uses two trained classifiers, a Decision Tree and a K-Nearest Neighbors model, and combines their outputs to show a final decision, an approval chance, and short explanation points.

**Live Demo:** https://loan-prediction-system-66yw.onrender.com/

## Project Overview

The workflow is simple:

1. The user enters loan details in the web form.
2. The form values are converted into the numeric format used during training.
3. The app creates a sample in the same feature order as the training data.
4. Both saved models evaluate the sample.
5. The result page shows approval, rejection, or manual review, along with reasons and strengths.

## Main Files

- `app.py` - Flask app, prediction logic, result explanation, and model loading
- `evaluation.py` - Holdout evaluation metrics printed at server startup
- `templates/index.html` - Frontend form and result display page
- `inspect_models.py` - Script for printing model details such as depth and feature importances
- `train.csv` - Dataset used for training and local accuracy checks
- `Untitled.ipynb` - Notebook used for experimentation and model building
- `Snapshots/` - Saved screenshots of the running application

## Screenshots

The `Snapshots/` folder contains example outputs from the app:

- [Empty form state](Snapshots/loan-prediction-empty-form.png)
- [Borderline decision with explanations](Snapshots/loan-prediction-borderline-form.png)
- [Borderline result details](Snapshots/loan-prediction-borderline-details.png)
- [Approved result example 1](Snapshots/loan-prediction-approved-form-1.png)
- [Approved result example 2](Snapshots/loan-prediction-approved-form-2.png)

## Requirements

- Python 3
- Flask
- pandas
- scikit-learn
- `model_dt.pkl`
- `model_knn.pkl`

## Run Locally

1. Install the Python dependencies.
2. Make sure `model_dt.pkl` and `model_knn.pkl` are present in the project root.
3. Start the app:

```bash
python app.py
```

4. Open the local Flask address shown in the terminal.

## Model Evaluation

When the app starts, it evaluates both saved models on a reproducible 20% stratified holdout from `train.csv`. The terminal reports:

- Accuracy
- Precision for approved applications
- Recall for approved applications
- F1 score for approved applications
- Confusion matrix in `[[TN, FP], [FN, TP]]` format.

## Deploy On Render

1. Push this repository to GitHub.
2. In Render, choose **New +** and create a **Blueprint** from the repository.
3. Render reads `render.yaml` and uses the configured build and start commands.
4. Open the generated public URL after the deployment finishes.

The equivalent manual settings are:

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`

## Notes

- The app expects the training data and saved models to use the same feature encoding.
- If the model files are missing, the Flask app will fail to start.
- `inspect_models.py` can be used to inspect model metadata before or after running the app.
- `evaluation.py` can also be run directly with `python evaluation.py` to print the same metrics without starting Flask.
- The form accepts annual income in rupees and loan amounts in rupees; `app.py` converts these values to the raw units expected by the saved models.