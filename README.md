# Bank Customer Analytics Using Artificial Neural Networks (ANN)

This repository contains two Deep Learning projects built using **Artificial Neural Networks (ANN)** with Python and TensorFlow.

The projects demonstrate both Classification and Regression problems using the Bank Customer dataset.

## Projects Overview

### 1. Bank Customer Churn Prediction (ANN Classification)

**Objective:** Predict whether a bank customer is likely to leave the bank or continue using its services.

**Target Variable:** `Exited`

- `0` – Customer will not leave the bank.
- `1` – Customer is likely to leave the bank.

**Input Features:**
- Credit Score
- Geography
- Gender
- Age
- Tenure
- Account Balance
- Number of Products
- Credit Card Status
- Active Member Status

**Technologies Used:**
- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Streamlit

**Model Output:**
- Churn Probability
- Likely to Churn / Not Likely to Churn

### 2. Estimated Salary Prediction (ANN Regression)

**Objective:** Predict a customer's estimated salary based on their banking and personal information.

**Target Variable:** `EstimatedSalary`

**Input Features:**
- Credit Score
- Geography
- Gender
- Age
- Tenure
- Account Balance
- Number of Products
- Credit Card Status
- Active Member Status

**Technologies Used:**
- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Streamlit

**Model Output:**
- Predicted Estimated Salary (Continuous Numerical Value)

## Project Structure

```text
Bank-Customer-Analytics/
│
├── Classification/
│   ├── app.py
│   ├── model.h5
│   ├── preprocessor.pkl
│   └── requirements.txt
│
├── Regression/
│   ├── app.py
│   ├── regression_model.py
│   ├── regression_preprocessor.pkl
│   └── requirements.txt
│
├── README.md
└── .gitignore
```

*Note: Update the model filenames in this structure to match your actual saved model artifacts.*

## Key Concepts Implemented

- Data Cleaning and Preprocessing
- Train-Test Split
- Feature Encoding
- Feature Scaling
- ColumnTransformer
- Artificial Neural Network Architecture
- Forward Propagation
- Model Training and Evaluation
- Early Stopping
- TensorBoard
- Model Serialization using Pickle and Keras
- Streamlit Application Development
- Model Deployment

## How to Run the Projects Locally

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the Project Folder

For Classification:

```bash
cd Classification
```

For Regression:

```bash
cd Regression
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```bash
python -m streamlit run app.py
```

## Deployment

Both applications can be deployed separately using Streamlit Community Cloud.

Each application uses its own Streamlit entry point and trained model artifacts.

## Learning Outcomes

Through these projects, I learned how to implement two different types of Deep Learning problems:

- **Classification:** Predicting categorical outcomes using ANN.
- **Regression:** Predicting continuous numerical values using ANN.

I also gained practical experience in building end-to-end Machine Learning and Deep Learning applications, saving trained models, creating interactive interfaces, and deploying applications.

## Author

**Bajrang**

Deep Learning | Artificial Neural Networks | Python | TensorFlow | Streamlit