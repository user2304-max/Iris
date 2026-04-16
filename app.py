import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model('ann_model.h5')
scaler = joblib.load('scaler.pkl')

st.title("Titanic Predictor")

pclass = st.selectbox("Class", [1,2,3])
sex = st.selectbox("Sex", ["male","female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("SibSp", 0, 10, 0)
parch = st.number_input("Parch", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 500.0, 50.0)
embarked = st.selectbox("Embarked", ["C","Q","S"])

sex = 1 if sex=="male" else 0
embarked = {"C":0,"Q":1,"S":2}[embarked]

if st.button("Predict"):
    data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    data = scaler.transform(data)

    result = model.predict(data)[0][0]

    if result > 0.5:
        st.success("Survived")
    else:
        st.error("Not Survived")