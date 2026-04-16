import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("🚢 Titanic Survival Predictor")

# Load dataset from same folder
@st.cache_data
def load_data():
    return pd.read_csv("titanic.csv")

df = load_data()

# Data preprocessing
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived']]

# Encode
df['Sex'] = df['Sex'].map({'male':1,'female':0})
df['Embarked'] = df['Embarked'].map({'C':0,'Q':1,'S':2})

# Features & target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train model
@st.cache_resource
def train_model():
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = train_model()

# User input
pclass = st.selectbox("Passenger Class", [1,2,3])
sex = st.selectbox("Sex", ["male","female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouse", 0, 10, 0)
parch = st.number_input("Parents/Children", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 500.0, 50.0)
embarked = st.selectbox("Embarked", ["C","Q","S"])

# Encode input
sex = 1 if sex=="male" else 0
embarked = {"C":0,"Q":1,"S":2}[embarked]

# Prediction
if st.button("Predict"):
    data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    result = model.predict(data)[0]

    if result == 1:
        st.success("✅ Survived")
    else:
        st.error("❌ Not Survived")
