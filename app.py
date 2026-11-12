import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle
import tensorflow as tf

model= tf.keras.models.load_model('model.h5')



with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehotencoder_geo.pkl', 'rb') as f:
    onehotencoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Churn Prediction App")

geo = st.selectbox('Geography', onehotencoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

geography_encoded = onehotencoder_geo.transform([[geo]]).toarray()

geo_encoded_df = pd.DataFrame(
    geography_encoded,
    columns=onehotencoder_geo.get_feature_names_out(['Geography'])
)

input_data = pd.concat([
    input_data.reset_index(drop=True),
    geo_encoded_df
], axis=1)

input_data = input_data[scaler.feature_names_in_]
# Encode Gender
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

input_data_scaled = scaler.transform(input_data)


prediction = model.predict(input_data_scaled)
pred_probability = prediction[0][0]

if pred_probability > 0.5:
    st.error(f'The customer is likely to churn with a probability of {pred_probability:.2f}')
else:
    st.success(f'The customer is unlikely to churn with a probability of {pred_probability:.2f}')
    


