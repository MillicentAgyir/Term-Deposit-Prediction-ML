# Term Deposit Subscription Prediction API
This repository contains a machine learning API built using FastAPI that predicts whether a client will subscribe to a term deposit based on various input features. The prediction model uses an XGBoost classifier trained and fine-tuned on a real-world banking dataset.

Project Overview
The project includes:

FastAPI: A high-performance Python web framework for API development.
XGBoost: A powerful machine learning model for classification.
Data Preprocessing: Scaling and one-hot encoding to handle numerical and categorical features.
Cloud Deployment: Ready for deployment on platforms like Render or Heroku.
Features
Prediction Endpoint: /predict takes client information and predicts term deposit subscription.
Model Optimization: Uses an optimal threshold for the F1-score to handle class imbalance.
Interactive Documentation: Swagger UI available for easy testing.
Tech Stack
Programming Language: Python
Web Framework: FastAPI
Machine Learning: XGBoost
Data Preprocessing: Scikit-learn
Deployment: Render/Heroku
How to Use
1. Clone the Repository
bash
Copy code
git clone https://github.com/your-username/Term-Deposit-Prediction-ML.git
cd Term-Deposit-Prediction-ML
2. Install Dependencies
Create a virtual environment and install the required packages:

bash
Copy code
python -m venv venv
source venv/bin/activate      # On Linux/Mac
venv\Scripts\activate         # On Windows

pip install -r requirements.txt
3. Run the API Locally
Start the FastAPI server using Uvicorn:

bash
Copy code
uvicorn src.main:app --reload
Access the API at:

arduino
Copy code
http://127.0.0.1:8000
4. Test the API
Go to the Swagger UI for interactive testing:

arduino
Copy code
http://127.0.0.1:8000/docs
You can test the /predict endpoint with the following sample JSON input:

json
Copy code
{
    "age": 35,
    "duration": 300,
    "campaign": 2,
    "pdays": 999,
    "previous": 0,
    "emp_var_rate": 1.1,
    "cons_price_idx": 93.994,
    "cons_conf_idx": -36.4,
    "euribor3m": 4.855,
    "nr_employed": 5195.8,
    "job": "technician",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "thu",
    "poutcome": "nonexistent"
}
5. Deploy to Render/Heroku
To deploy the application:

Render:
Use the following start command:
bash
Copy code
uvicorn src.main:app --host 0.0.0.0 --port $PORT
Heroku:
bash
Copy code
heroku login
heroku create term-deposit-api
git push heroku main
Endpoints
Endpoint	Method	Description
/	GET	Returns a welcome message.
/predict	POST	Predicts term deposit subscription.
Response Example:
Input:

json
Copy code
{
    "age": 35,
    "duration": 300,
    "campaign": 2,
    "pdays": 999,
    "previous": 0,
    "emp_var_rate": 1.1,
    "cons_price_idx": 93.994,
    "cons_conf_idx": -36.4,
    "euribor3m": 4.855,
    "nr_employed": 5195.8,
    "job": "technician",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "thu",
    "poutcome": "nonexistent"
}
Response:

json
Copy code
{
    "Prediction": "yes",
    "Probability": 0.82
}
Files and Directories
bash
Copy code
Term-Deposit-Prediction-ML/
│
├── src/
│   ├── main.py                # FastAPI app
│   ├── xgboost_model.pkl      # Trained XGBoost model
│   ├── scaler.pkl             # StandardScaler object
│   ├── optimal_threshold.pkl  # Optimal threshold for predictions
│   └── x_train_columns.pkl    # Column names from training data
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
Model Performance
Best F1-Score: 0.68 with the optimal threshold.
Optimized for Recall: To address class imbalance.
Future Improvements
Add additional models for comparison.
Improve the API for batch predictions.
Include better logging and error handling.
Contributions
Contributions are welcome! Feel free to fork the repository, create a new branch, and submit a pull request.

License
This project is licensed under the MIT License.

Contact
For any questions or feedback, contact:

Email: your-email@example.com
GitHub: Your GitHub Profile
Thank You for Visiting! 🚀
