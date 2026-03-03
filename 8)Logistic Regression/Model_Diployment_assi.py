#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
import streamlit as st


# In[2]:


model = pickle.load(open('Logistic_model.pkl','rb'))


# In[3]:


scale = pickle.load(open('std_sca.pkl','rb'))


# In[4]:


st.title('Model Deployment using Logistic Regression')


# In[5]:


#st.sidebar.header("Enter Patient Details")
def user_input_parameters():
    Pregnancies = st.sidebar.number_input('Pregnancies', 0, 20, 1)
    Glucose = st.sidebar.number_input('Glucose', 0, 300, 120)
    BloodPressure = st.sidebar.number_input('Blood Pressure', 0, 200, 70)
    SkinThickness = st.sidebar.number_input('Skin Thickness', 0, 100, 20)
    Insulin = st.sidebar.number_input('Insulin', 0, 900, 80)
    BMI = st.sidebar.number_input('BMI', 0.0, 70.0, 25.0)
    DPF = st.sidebar.number_input('Diabetes Pedigree Function', 0.0, 3.0, 0.5)
    Age = st.sidebar.slider('Age', 1, 100, 30)

    data = {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DPF,
        'Age': Age
    }

    features = pd.DataFrame(data, index=[0])
    features_scaled = scale.transform(features)

    return features_scaled


# In[6]:


input_data = user_input_parameters()

predict_btn = st.button("Predict")

if predict_btn:
    prediction = model.predict(input_data)
    prediction_prob = model.predict_proba(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("Person is likely DIABETIC")
    else:
        st.success("Person is NOT Diabetic")

    st.subheader("Prediction Probability")
    st.write("Not Diabetic:", prediction_prob[0][0])
    st.write("Diabetic:", prediction_prob[0][1])


# In[ ]:





# In[ ]:




