# OTT Customer Churn Prediction

A cloud-deployed machine learning application that predicts whether an OTT streaming customer is likely to churn based on subscription, engagement, and customer-support behavior.

## Live Demo

**Streamlit App:** https://ott-churn-prediction-yb.streamlit.app

## Project Overview

The goal of this project was to build an end-to-end churn prediction workflow for an OTT streaming platform.

The original dataset contained approximately 2,000 customer records. After removing rows with missing target values, 1,965 labeled records were used for modeling. Churn represented roughly 13% of the dataset, so the project used stratified sampling and a class-balanced Random Forest to better account for the imbalanced target.

The final deployed model is packaged together with its preprocessing pipeline and served through an interactive Streamlit web application.

## Model Inputs

The deployed model uses the following customer features:

- Gender
- Age
- Number of days subscribed
- Multi-screen usage
- Mail subscription status
- Weekly minutes watched
- Minimum daily minutes watched
- Maximum daily minutes watched
- Weekly maximum night minutes
- Videos watched
- Maximum days inactive
- Customer support calls

Identifiers such as customer ID, phone number, and year were excluded from model training.

## Data Preprocessing

A scikit-learn `ColumnTransformer` and `Pipeline` were used so preprocessing is applied consistently during both training and prediction.

### Numerical Features

- Median imputation for missing values
- Standard scaling

### Categorical Features

- Most-frequent-value imputation
- One-hot encoding with unknown-category handling

The binary `multi_screen` and `mail_subscribed` variables were converted from Yes/No values to 1/0 before modeling.

## Modeling

The project used a class-balanced `RandomForestClassifier`.

Hyperparameters were explored using both:

- `RandomizedSearchCV`
- `GridSearchCV`

The RandomizedSearchCV model generalized better on the held-out test set and was selected for deployment.

### Selected Model Performance

| Metric | Result |
|---|---:|
| Test Accuracy | 91.2% |
| Churn Precision | 67.4% |
| Churn Recall | 66.7% |
| Churn F1 Score | 67.1% |

The model was evaluated on a stratified 33% holdout test set.

## Deployment

The complete fitted scikit-learn pipeline was serialized using Joblib and loaded directly by the Streamlit application.

Deployment workflow:

```text
Customer Input
      ↓
Streamlit Web App
      ↓
Saved scikit-learn Pipeline
      ↓
Preprocessing
      ↓
Random Forest
      ↓
Churn Prediction + Probability
```

The application returns:

- Predicted customer status: **Churn** or **Stay**
- Estimated churn probability
- A visual probability indicator
- The exact input record passed to the model

## Tech Stack

- Python
- pandas
- scikit-learn
- Joblib
- Streamlit
- GitHub
- Streamlit Community Cloud

## Project Structure

```text
ott-churn-prediction/
│
├── app.py
├── churn_model.joblib
├── requirements.txt
└── README.md
```

## Run Locally

Clone the repository and move into the project folder:

```bash
git clone https://github.com/yatharthbahl7-web/ott-churn-prediction.git
cd ott-churn-prediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
streamlit run app.py
```

The app will normally open at:

```text
http://localhost:8501
```

## Example Use Case

A retention team could use a churn model like this to identify customers with elevated churn risk and prioritize them for actions such as:

- retention offers
- targeted engagement campaigns
- customer-support follow-up
- subscription experience improvements

This project is a portfolio demonstration and is not intended to represent an enterprise production system.

## Future Improvements

Potential extensions include:

- probability threshold tuning based on business costs
- SHAP-based prediction explanations
- model monitoring and drift detection
- REST API deployment using FastAPI
- Docker containerization
- AWS deployment
- automated retraining pipelines

## Author

**Yatharth Bahl**

Machine Learning / Data Science Portfolio Project
