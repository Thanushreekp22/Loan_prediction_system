# Loan Approval Prediction Project

This project is a small Flask web application that predicts whether a loan is likely to be approved based on applicant details such as income, credit history, education, employment type, and property area.

The app loads two pre-trained machine learning models, a Decision Tree and a K-Nearest Neighbors model, then compares their predictions to produce a final result. It also shows an approval chance and simple explanation points for why a profile looks stronger or weaker.

## What the project does

- Collects loan applicant details through a web form
- Converts the form inputs into the numeric feature layout used during training
- Loads and evaluates two saved models: `model_dt.pkl` and `model_knn.pkl`
- Displays an approval, rejection, or manual review outcome
- Generates simple positive and negative insights for the result

## Main files

- `app.py` - Flask application, prediction logic, and explanation generation
- `templates/index.html` - Frontend form and prediction result display
- `inspect_models.py` - Helper script for inspecting trained model properties
- `train.csv` - Training dataset used for model development and accuracy checks
- `Untitled.ipynb` - Notebook workspace for experimentation and model building

## How it works

1. The user fills in the loan application form in the browser.
2. The app validates the submitted values and converts them to numeric form.
3. A sample is built in the same column order used during training.
4. The Decision Tree and KNN models make predictions on the sample.
5. The app combines both model outputs and estimated probabilities into a final loan decision.
6. The page also shows explanation points such as credit history, income strength, and debt-to-income ratio.

## Requirements

- Python 3
- Flask
- pandas
- scikit-learn
- The trained model files `model_dt.pkl` and `model_knn.pkl`

## Run locally

1. Install the Python dependencies.
2. Make sure `model_dt.pkl` and `model_knn.pkl` are present in the project root.
3. Start the app:

```bash
python app.py
```

4. Open the local Flask address shown in the terminal.

## Notes

- The app expects the training data and the saved models to use the same feature encoding.
- If the model files are missing, the Flask app will fail to start.
- `inspect_models.py` can be used to print model details such as depth, neighbors, and feature importances.