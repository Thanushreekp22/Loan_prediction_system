# Loan Approval Prediction System

This project is a Flask-based machine learning web app that predicts whether a loan application is likely to be approved. The prediction is made from applicant details such as income, credit history, education, employment type, and property area.

The app uses two trained classifiers, a Decision Tree and a K-Nearest Neighbors model, and combines their outputs to show a final decision, an approval chance, and short explanation points.

## Project Overview

The workflow is simple:

1. The user enters loan details in the web form.
2. The form values are converted into the numeric format used during training.
3. The app creates a sample in the same feature order as the training data.
4. Both saved models evaluate the sample.
5. The result page shows approval, rejection, or manual review, along with reasons and strengths.

## Main Files

- `app.py` - Flask app, prediction logic, result explanation, and model loading
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

## Notes

- The app expects the training data and saved models to use the same feature encoding.
- If the model files are missing, the Flask app will fail to start.
- `inspect_models.py` can be used to inspect model metadata before or after running the app.