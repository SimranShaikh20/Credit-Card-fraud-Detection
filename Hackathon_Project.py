import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load and preprocess the dataset
data = pd.read_csv("creditcard.csv")

# Splitting into legit and fraud cases
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Balance the dataset
legit_sample = legit.sample(n=len(fraud), random_state=2)
balanced_data = pd.concat([legit_sample, fraud], axis=0)

# Splitting features and target
x = balanced_data.drop('Class', axis=1)
y = balanced_data['Class']

# Splitting into train and test datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Logistic Regression model
model = LogisticRegression(max_iter=500)  # Increase iterations for convergence
model.fit(x_train, y_train)

# Evaluate the model
train_acc = accuracy_score(model.predict(x_train), y_train)
test_acc = accuracy_score(model.predict(x_test), y_test)

# Streamlit web app
st.title("Credit Card Fraud Detection Model")
st.write(f"Model Training Accuracy: {train_acc:.2f}")
st.write(f"Model Testing Accuracy: {test_acc:.2f}")

# Input from the user
st.write("Enter feature values separated by commas (matching the feature set):")
input_data = st.text_area("Example: 1.0, 0.5, -0.2, ...")

submit = st.button("Submit")

if submit:
    try:
        # Convert input data to numpy array
        features = np.asarray([float(x) for x in input_data.split(',')], dtype=np.float64)
        
        # Validate input shape
        if features.shape[0] != x.shape[1]:
            st.error(f"Invalid number of features! Expected {x.shape[1]}, but got {features.shape[0]}.")
        else:
            # Make prediction
            detection = model.predict(features.reshape(1, -1))
            if detection[0] == 0:
                st.success("Transaction is Legitimate.")
            else:
                st.error("Transaction is Fraudulent.")
    except ValueError:
        st.error("Invalid input! Please enter numeric values only.")
