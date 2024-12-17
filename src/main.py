from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Initialize FastAPI app
app = FastAPI(
    title="Term Deposit Prediction API",
    description="API for predicting term deposit subscription",
    version="1.0"
)

# Load saved model, scaler, threshold, and X_train column names
try:
    # Get absolute paths for model files
    base_path = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base_path, 'xgboost_model.pkl'))
    scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
    optimal_threshold = joblib.load(os.path.join(base_path, 'optimal_threshold.pkl'))
    x_train_columns = joblib.load(os.path.join(base_path, 'x_train_columns.pkl'))  # Load column names
except Exception as e:
    print("Error loading model, scaler, threshold, or column names:", e)

# Define the expected input data schema using Pydantic
class InputData(BaseModel):
    age: int
    duration: int
    campaign: int
    pdays: int
    previous: int
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    poutcome: str

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the Term Deposit Prediction API!"}

# Prediction endpoint
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/predict")
def predict(data: InputData):
    if x_train_columns is None:
        return {"error": "x_train_columns is not defined. Please check the server setup."}

    try:
        # Convert input data to a DataFrame
        input_df = pd.DataFrame([data.dict()])
        logger.info("Initial Input DataFrame:")
        logger.info(input_df)

        # Rename columns to match model training (if required)
        input_df = input_df.rename(columns={
            'cons_conf_idx': 'cons.conf.idx',
            'cons_price_idx': 'cons.price.idx',
            'emp_var_rate': 'emp.var.rate',
            'nr_employed': 'nr.employed'
        })
        logger.info("After Renaming Columns:")
        logger.info(input_df.columns)

        # Explicitly ensure all numerical columns exist
        numerical_columns = ['age', 'duration', 'campaign', 'pdays', 'previous',
                             'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                             'euribor3m', 'nr.employed']
        for col in numerical_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        logger.info("After Adding Missing Numerical Columns:")
        logger.info(input_df.columns)

        # One-Hot Encoding for categorical features
        categorical_columns = [
            'job', 'marital', 'education', 'default',
            'housing', 'loan', 'contact', 'month',
            'day_of_week', 'poutcome'
        ]
        input_df = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)
        logger.info("After One-Hot Encoding:")
        logger.info(input_df.columns)

        # Add missing columns and align with x_train_columns
        missing_cols = set(x_train_columns) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        input_df = input_df.reindex(columns=x_train_columns, fill_value=0)
        logger.info("Final Input DataFrame Before Prediction:")
        logger.info(input_df.columns)

        # Scale numerical features
        input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

        # Predict probabilities
        probability = model.predict_proba(input_df)[:, 1][0]
        prediction = int(probability >= optimal_threshold)

        # Return prediction result
        return {
            "Prediction": "yes" if prediction == 1 else "no",
            "Probability": round(probability, 4)
        }
    except Exception as e:
        return {"error": str(e)}
