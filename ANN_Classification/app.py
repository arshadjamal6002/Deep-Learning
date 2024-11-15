import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load encoders and scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:
    enc = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    le = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Title and subheader
st.title("Customer Churn Prediction")
st.subheader("Predict whether a customer is likely to churn based on their profile")

# Sidebar for user inputs (for better layout)
st.sidebar.header("Enter Customer Details")

# User input (using Streamlit sidebar)
geography = st.sidebar.selectbox('Geography', enc.categories_[0])
gender = st.sidebar.selectbox('Gender', le.classes_)
age = st.sidebar.slider('Age', 18, 92)
balance = st.sidebar.number_input('Balance', min_value=0.0, max_value=1e6, step=100.0)
credit_score = st.sidebar.number_input('Credit Score', min_value=0, max_value=850, step=1)
estimated_salary = st.sidebar.number_input('Estimated Salary', min_value=0.0, max_value=1e6, step=100.0)
tenure = st.sidebar.slider('Tenure (Years)', 0, 10)
num_of_products = st.sidebar.slider('Number of Products', 1, 4)
has_cr_card = st.sidebar.selectbox('Has Credit Card', [0, 1])
is_active_member = st.sidebar.selectbox('Is Active Member', [0, 1])

# Display user inputs on the main page
st.write("### Customer Input Details")
input_data_dict = {
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Balance': balance,
    'Credit Score': credit_score,
    'Estimated Salary': estimated_salary,
    'Tenure': tenure,
    'Number of Products': num_of_products,
    'Has Credit Card': has_cr_card,
    'Is Active Member': is_active_member
}

# Show the input data as a table
st.table(pd.DataFrame(input_data_dict, index=[0]))

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [le.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
encoded_values = enc.transform(np.array([geography]).reshape(-1, 1)).astype(int)
input_encoded = pd.DataFrame(encoded_values, columns=enc.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_df = pd.concat([input_data, input_encoded], axis=1)

# Scale the input data
input_df_scaled = scaler.transform(input_df)

# Add a spinner to show processing time
with st.spinner('Making prediction...'):
    # Predict churn
    prediction = model.predict(input_df_scaled)
    prediction_proba = prediction[0][0]

# Display the result with some styling
st.markdown(f"### **Churn Probability**: {prediction_proba:.2f}")
if prediction_proba > 0.5:
    st.markdown("<h3 style='color: red;'>The customer is likely to churn.</h3>", unsafe_allow_html=True)
else:
    st.markdown("<h3 style='color: green;'>The customer is not likely to churn.</h3>", unsafe_allow_html=True)

# Additional Information
st.markdown("---")
st.write("### About this Model")
st.write("""
This machine learning model predicts customer churn based on various customer attributes.
- **Features Used**: Geography, Gender, Age, Balance, Credit Score, Estimated Salary, Tenure, and more.
- **Prediction**: The output is a probability indicating whether the customer will churn (i.e., leave the service).
""")

# Footer
st.markdown("---")
st.write("Created by Arshad Jamal | [GitHub](https://github.com/arshadjamal6002)")

