import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('regression_model.h5')

# Load encoders and scaler
with open('onehot_encoder_geo_reg.pkl', 'rb') as file:
    enc = pickle.load(file)

with open('label_encoder_gender_reg.pkl', 'rb') as file:
    le = pickle.load(file)

with open('scaler_reg.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.set_page_config(page_title="Customer Prediction", layout="wide")

# Title and subheader
st.title("Customer Prediction")
st.subheader("Predict a customer's salary based on their profile")

# Sidebar for user inputs (for better layout)
st.sidebar.header("Enter Customer Details")

# User input (using Streamlit sidebar)
geography = st.sidebar.selectbox('Geography', enc.categories_[0])
gender = st.sidebar.selectbox('Gender', le.classes_)
age = st.sidebar.slider('Age', 18, 92)
balance = st.sidebar.number_input('Balance', min_value=0.0, max_value=1e6, step=100.0)
credit_score = st.sidebar.number_input('Credit Score', min_value=0, max_value=850, step=1)
# estimated_salary = st.sidebar.number_input('Estimated Salary', min_value=0.0, max_value=1e6, step=100.0)
tenure = st.sidebar.slider('Tenure (Years)', 0, 10)
num_of_products = st.sidebar.slider('Number of Products', 1, 4)
has_cr_card = st.sidebar.selectbox('Has Credit Card', [0, 1])
is_active_member = st.sidebar.selectbox('Is Active Member', [0, 1])
exited = st.sidebar.selectbox('Exited', [0, 1])

# Display user inputs on the main page
st.write("### Customer Input Details")
input_data_dict = {
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Balance': balance,
    'Credit Score': credit_score,
    # 'Estimated Salary': estimated_salary,
    'Tenure': tenure,
    'Number of Products': num_of_products,
    'Has Credit Card': has_cr_card,
    'Is Active Member': is_active_member,
    'Exited' : exited
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
    # 'EstimatedSalary': [estimated_salary]
    'Exited' : [exited]
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
    # Predict the output (regression value)
    prediction = model.predict(input_df_scaled)
    predicted_salary = prediction[0][0]  # Assuming it's a single scalar output

# Display the result with some styling
# Display the result with green color
st.markdown(f"<h3 style='color: green;'>Predicted Estimated Salary: ${predicted_salary:,.2f}</h3>", unsafe_allow_html=True)
st.write(f"The model predicts the estimated salary of this customer is **${predicted_salary:,.2f}** based on the provided details.")

# Additional Information
st.markdown("---")
st.write("### About this Model")
st.write("""
This machine learning model predicts the **estimated salary** of a customer based on various customer attributes such as:
- Geography, Gender, Age, Balance, Credit Score, Tenure, and more.
- The prediction is a continuous value (in USD), indicating the estimated salary of the customer.

### Features Used:
- **Geography**: One-hot encoded categorical feature for location.
- **Gender**: Label encoded gender feature.
- **Age**: Customer's age.
- **Balance**: Account balance of the customer.
- **Credit Score**: Customer's credit score.
- **Tenure**: Duration (in years) the customer has been with the service.
- **Number of Products**: Number of products the customer holds with the bank.
- **Has Credit Card**: Whether the customer has a credit card (binary).
- **Is Active Member**: Whether the customer is an active member of the service (binary).

""")

# Footer
st.markdown("---")
st.write("Created by Arshad Jamal | [GitHub](https://github.com/arshadjamal6002)")