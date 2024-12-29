import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Loading the dataset to pandas DataFrame
loan_dataset = pd.read_csv(r'Loan_Data.csv')

# Preprocessing steps as per your previous code
loan_dataset.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)

# Replacing the value of 3+ to 4
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)

# Convert categorical columns to numerical values
loan_dataset.replace({'Married': {'No': 0, 'Yes': 1}, 'Gender': {'Male': 1, 'Female': 0}, 'Self_Employed': {'No': 0, 'Yes': 1},
                      'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2}, 'Education': {'Graduate': 1, 'Not Graduate': 0}}, inplace=True)

loan_dataset = loan_dataset.dropna()

# Separating the data and label
X = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
Y = loan_dataset['Loan_Status']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Accuracy on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# Accuracy on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# Streamlit app layout
st.title("Loan Prediction App")
st.write("### Predict if a loan will be approved or rejected based on the input data.")

# Input fields for the user
applicant_income = st.number_input("Enter Applicant's Income", min_value=0.0, step=1000.0)
coapplicant_income = st.number_input("Enter Coapplicant's Income", min_value=0.0, step=1000.0)
loan_amount = st.number_input("Enter Loan Amount", min_value=0.0, step=1000.0)
loan_amount_term = st.number_input("Enter Loan Amount Term (in months)", min_value=0, step=12)
credit_history = st.selectbox("Enter Credit History (0 or 1)", options=[0, 1])

# Function to predict the loan status
def predict_loan_status(applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history):
    input_data = pd.DataFrame({
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Gender': [1],  # Assuming Male for this example
        'Married': [1],  # Assuming Married for this example
        'Self_Employed': [0],  # Assuming Not Self Employed
        'Property_Area': [1],  # Assuming Semiurban for this example
        'Education': [1],  # Assuming Graduate for this example
        'Dependents': [0]  # Assuming No Dependents for this example
    })

    # Ensure the columns are in the same order as the training data
    input_data = input_data[X.columns]

    # Predicting the loan status
    prediction = classifier.predict(input_data)

    # Returning the loan status as 'Approved' or 'Rejected'
    if prediction == 1:
        return "Loan Approved"
    else:
        return "Loan Rejected"

# Button to make predictions
if st.button("Predict Loan Status"):
    loan_status = predict_loan_status(applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history)
    st.success(f"Predicted Loan Status: {loan_status}")

# Displaying model accuracy
st.write(f"### Model Accuracy:")
st.write(f"Accuracy on training data: {training_data_accuracy * 100:.2f}%")
st.write(f"Accuracy on test data: {test_data_accuracy * 100:.2f}%")
