import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Importing data
data = pd.read_csv("creditcard.csv")
legit = data[data.Class == 0]
fraud = data[data['Class'] == 1]

# Preparing features and target
x = data.drop('Class', axis=1)
y = data['Class']

# Balancing the dataset by sampling
legit_s = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_s, fraud], axis=0)
x = data.drop('Class', axis=1)
y = data['Class']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Train the model
model = LogisticRegression(max_iter=10000)
model.fit(x_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(x_train), y_train)
test_acc = accuracy_score(model.predict(x_test), y_test)

# Display model performance
st.write(f"Training Accuracy: {train_acc}")
st.write(f"Test Accuracy: {test_acc}")

# Web app
st.title("Credit Card Fraud Detection Model")
input_data = st.text_area("Enter all required feature values here (comma separated):")

submit = st.button("Submit")

if submit:
    try:
        # Split and convert input data to float64
        input_s = input_data.split(',')
        features = np.asarray(input_s, dtype=np.float64).reshape(1, -1)  # Reshape for prediction
        
        # Prediction
        detection = model.predict(features)

        # Display result
        if detection[0] == 0:
            st.write("Legitimate transaction.")
        else:
            st.write("Fraudulent transaction.")
    except Exception as e:
        st.write(f"Error: {e}. Please check your input format.")
