import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Importing data
data = pd.read_csv("creditcard.csv")

# Checking for missing or infinite values
if data.isnull().sum().any() or np.any(np.isinf(data)):
    st.write("Data contains missing or infinite values. Cleaning the data...")
    data = data.dropna()  # Drop rows with NaN values
    # If there are any infinite values, remove them
    data = data[~data.isin([np.inf, -np.inf]).any(axis=1)]

# Split data into legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data['Class'] == 1]

# Resampling legitimate transactions to match fraudulent transactions count
legit_s = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_s, fraud], axis=0)

# Features and target
x = data.drop('Class', axis=1)
y = data['Class']

# Checking for matching lengths of x and y
if x.shape[0] != y.shape[0]:
    st.write(f"Shape mismatch: x has {x.shape[0]} rows, but y has {y.shape[0]} rows.")
else:
    # Ensure both x and y have the correct data
    print("x and y have matching lengths.")
    print(x.head())
    print(y.head())

# Checking for class imbalance
class_counts = y.value_counts()
st.write("Class distribution:")
st.write(class_counts)

# Proceed with train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Train the Logistic Regression model
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
