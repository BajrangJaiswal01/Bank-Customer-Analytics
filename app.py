
import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
import os

# Page configuration
st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

# Project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.pkl")


# Load model and preprocessor
@st.cache_resource
def load_artifacts():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)

        with open(PREPROCESSOR_PATH, "rb") as file:
            preprocessor = pickle.load(file)

        return model, preprocessor

    except Exception as e:
        st.error(f"Error loading model or preprocessor: {e}")
        st.stop()


model, preprocessor = load_artifacts()


# Application heading
st.title("🏦 Bank Churn Prediction")

st.write(
    "Enter customer information to predict whether "
    "the customer is likely to leave the bank."
)

st.divider()

st.subheader("Customer Information")


# Input form
with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650
        )

        geography = st.selectbox(
            "Geography",
            ["France", "Germany", "Spain"]
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )

        tenure = st.number_input(
            "Tenure",
            min_value=0,
            max_value=10,
            value=5
        )

    with col2:

        balance = st.number_input(
            "Account Balance",
            min_value=0.0,
            value=75000.0
        )

        num_of_products = st.selectbox(
            "Number of Products",
            [1, 2, 3, 4]
        )

        has_cr_card = st.selectbox(
            "Has Credit Card?",
            ["Yes", "No"]
        )

        is_active_member = st.selectbox(
            "Is Active Member?",
            ["Yes", "No"]
        )

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=60000.0
        )

    submit = st.form_submit_button(
        "Predict Churn",
        use_container_width=True
    )


# Prediction
if submit:

    try:

        new_data = {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_of_products,
            "HasCrCard": 1 if has_cr_card == "Yes" else 0,
            "IsActiveMember": 1 if is_active_member == "Yes" else 0,
            "EstimatedSalary": estimated_salary
        }

        # Convert dictionary into DataFrame
        new_customer = pd.DataFrame([new_data])

        # Apply preprocessing
        processed_data = preprocessor.transform(new_customer)

        # Model prediction
        prediction = model.predict(processed_data, verbose=0)

        # Extract probability
        probability = float(prediction[0][0])

        # Display result
        st.divider()
        st.subheader("Prediction Result")

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

        st.progress(probability)

        if probability >= 0.5:
            st.error("Customer is Likely to Churn")
        else:
            st.success("Customer is Not Likely to Churn")

       

    except Exception as e:

        st.error("An error occurred during prediction.")
        st.exception(e)


# Footer
st.divider()

