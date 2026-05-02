# Fraud Detection System - Final Report

## 1. Executive Summary
This project aims to detect fraudulent mobile financial transactions using advanced machine learning techniques. We developed a robust classification model to identify potential fraud in real-time, thereby minimizing financial losses.

## 2. Data Analysis (EDA)
We utilized a dataset containing transaction details such as type, amount, and account balances.
- **Key Insights**:
    - Fraudulent transactions are primarily 'TRANSFER' and 'CASH_OUT' types.
    - True fraud events are rare (highly imbalanced dataset).
    - Need to handle high cardinality in account names.

## 3. Methodology
### 3.1 Data Preprocessing
- **Cleaning**: Checked for missing values and outliers.
- **Feature Engineering**: 
    - Analyzed balance discrepancies (`oldBalance + amount != newBalance`).
    - Encoded categorical variables (One-Hot Encoding for transaction types).
- **Handling Imbalance**: Utilized class weighting and appropriate evaluation metrics (Area Under Precision-Recall Curve).

### 3.2 Models Trained
1. **Logistic Regression**: Baseline model for interpretability.
2. **Random Forest**: Ensemble method robust to overfitting.
3. **XGBoost**: Gradient boosting for high performance on tabular data.

## 4. Model Performance & Evaluation
### Metrics Used:
- **Precision**: Accuracy of positive predictions (Fraud).
- **Recall**: Ability to capture actual fraud cases.
- **F1-Score**: Harmonic mean of Precision and Recall.
- **ROC-AUC**: Trade-off between True Positive Rate and False Positive Rate.

### Results:
*(Populate this section with values from the Notebook)*
- **Best Model**: [Model Name]
- **Precision**: [Value]
- **Recall**: [Value]

## 5. Financial Impact Analysis
Assuming:
- **Cost of Fraud (False Negative)**: Total amount of the fraudulent transaction lost.
- **Cost of Prevention (False Positive)**: Administrative cost to verify a legitimate transaction (e.g., $5).

The model optimizes for **Recall** to minimize missed fraud cases while maintaining reasonable Precision to reduce administrative overhead.

## 6. Conclusion and Next Steps
The [Best Model] demonstrates strong capability in flagging fraud. Recommended next steps include:
- Deploying the model as an API (e.g., via Flask/FastAPI).
- Continuous monitoring and retraining with new data.
- Investigating 'Merchant' transactions if data becomes available.
