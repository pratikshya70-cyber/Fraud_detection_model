# Fraud_detection_model
Developed an end-to-end Fraud Detection System for mobile financial transactions using Machine Learning to improve fraud detection accuracy and business value. The project focused on identifying fraudulent transactions while minimizing false alarms and maximizing financial savings for institutions.

🔍 Features

✅ Fraud detection for mobile financial transactions using Machine Learning

✅ Real-time fraud prediction with high accuracy and reliability

✅ Model comparison using Logistic Regression, Random Forest, and XGBoost

✅ SHAP Explainability for identifying key fraud-driving features

✅ Financial profit-based threshold optimization instead of default 0.5 threshold

✅ Interactive Streamlit dashboard for fraud analysis and prediction

✅ Business impact analysis with projected savings of $318M


🛠️ Technologies Used:

Python

Pandas

NumPy

Scikit-learn

SHAP

Matplotlib

Seaborn

Streamlit

🚀 Getting Started

1. Clone the Repository
   
git clone https://github.com/yourusername/fraud-detection-system.gitcd fraud-detection-system


2. Install Dependencies

Make sure Python is installed, then run:

pip install -r requirements.txt

If requirements.txt is missing, manually install:

pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn streamlit joblib

📊 How to Use
1. Train the Model
Run the notebook or training script:
python fraud_detection_training.py

This will perform data preprocessing, feature engineering, model training, SHAP explainability, and save the trained models.

2. Run Streamlit Application
streamlit run app.py

This will open the interactive Fraud Detection Dashboard in your browser.

🧠 How It Works

The system preprocesses transaction data and creates important engineered features such as:


Error Balance Origin

Error Balance Destination

Balance Difference

Amount-to-Balance Ratio

Multiple ML models are trained and compared, with Random Forest selected as the best performer:

Accuracy: 99.8%,
Precision: 100%,
Recall: 100%,
AUC Score: 1.000


SHAP Explainability

It identifies the most influential fraud features and improves model transparency.

Instead of using the default threshold (0.5), the system optimizes the decision threshold based on projected financial profit, resulting in approximately $318M projected savings.

📄 License

This project is open-source and available for educational and professional use.

🙌 Acknowledgements

Scikit-learn for machine learning models

SHAP for explainable AI

Streamlit for dashboard deployment

