import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

## Importing data
data = pd.read_csv("creditcard.csv")

# Handling missing values
if data.isnull().sum().sum() > 0:
    data = data.dropna()
    st.write(f"Dataset size after removing null values: {len(data)}")

# Splitting data into legitimate and fraud cases
legit = data[data['Class'] == 0]
fraud = data[data['Class'] == 1]

# Debugging dataset size
st.write(f"Total records: {len(data)}")
st.write(f"Legitimate transactions: {len(legit)}")
st.write(f"Fraudulent transactions: {len(fraud)}")

# Check if there is any fraud data
if len(fraud) == 0:
    raise ValueError("No fraud transactions found in the dataset.")

# Balancing the dataset
legit_sample = legit.sample(n=min(len(fraud), len(legit)), random_state=2)
balanced_data = pd.concat([legit_sample, fraud], axis=0)

# Check if the balanced dataset is empty
if len(balanced_data) == 0:
    raise ValueError("The balanced dataset is empty. Check your input data and balancing logic.")

# Splitting features and target
x = balanced_data.drop('Class', axis=1)
y = balanced_data['Class']

# Splitting into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=2
)

# Train the model
model = LogisticRegression()
model.fit(x_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(x_train), y_train)
test_acc = accuracy_score(model.predict(x_test), y_test)

st.write(f"Training Accuracy: {train_acc * 100:.2f}%")
st.write(f"Testing Accuracy: {test_acc * 100:.2f}%")
