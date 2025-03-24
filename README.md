# 🚀 Fraud Transaction Detection using Machine Learning

## 📌 Project Overview
This project applies machine learning techniques to detect fraudulent credit card transactions using a Kaggle dataset. It demonstrates key concepts in financial security, data preprocessing, model training, and evaluation.

📂 **Dataset:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 🎯 Project Aims

🔹 **Real-World Application:** Machine learning applied to financial fraud detection.

🔹 **Handling Imbalanced Data:** Uses undersampling to balance fraud and legitimate transactions.

🔹 **ML Workflow:** Covers data loading, preprocessing, model training, and evaluation.

🔹 **Python Libraries:** Utilizes `pandas`, `numpy`, and `scikit-learn`.

🔹 **Simple Model Implementation:** Starts with Logistic Regression.

🔹 **Model Evaluation:** Assesses accuracy on training and test data.

🔹 **Web Integration:** Implements a Streamlit interface for predictions.

🔹 **Reproducibility:** Open dataset and code for easy replication.

🔹 **Scalability:** Can be expanded with advanced techniques.

🔹 **Significance:** Highlights fraud detection's importance in the financial sector.

---

## 🔧 Data Preprocessing

1. **📂 Data Loading:**
   ```python
   data = pd.read_csv("creditcard.csv")
   ```

2. **🔍 Class Separation:**
   ```python
   legit = data[data.Class == 0]
   fraud = data[data['Class'] == 1]
   ```

3. **📊 Feature & Target Separation:**
   ```python
   x = data.drop('Class', axis=1)
   y = data['Class']
   ```

4. **⚖ Handling Imbalance:**
   ```python
   legit_s = legit.sample(n=len(fraud), random_state=2)
   data = pd.concat([legit_s, fraud], axis=0)
   ```

---

## 🤖 Model Training

1. **📌 Train-Test Split:**
   ```python
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
   ```

2. **📈 Model Selection:**
   ```python
   model = LogisticRegression()
   ```

3. **🎯 Model Training:**
   ```python
   model.fit(x_train, y_train)
   ```

---

## 📊 Model Evaluation

1. **📊 Accuracy Calculation:**
   ```python
   train_acc = accuracy_score(model.predict(x_train), y_train)
   test_acc = accuracy_score(model.predict(x_test), y_test)
   ```

---

## 🎯 Conclusion
This project provides a fundamental approach to fraud detection using machine learning. The use of Logistic Regression and class balancing through undersampling establishes a baseline model. The next steps could include:

✅ Trying advanced models like Random Forest, XGBoost, or Neural Networks.  
✅ Experimenting with feature engineering techniques.  
✅ Implementing real-time fraud detection with an API.  

---

## 🌐 Web App Implementation
🔹 A simple interactive web interface using **Streamlit** allows users to input transaction details and receive fraud predictions.

```python
import streamlit as st
st.title("Fraud Detection System")
```

---

## 📌 Get Started
1️⃣ Clone the repository:  
   ```bash
   git clone SimranShaikh20/Credit-Card-fraud-Detection
   ```
2️⃣ Install dependencies:  
   ```bash
   pip install pandas numpy scikit-learn streamlit
   ```
3️⃣ Run the web app:  
   ```bash
   streamlit run app.py
   ```

---

### 💡 Future Enhancements
🔹 Use **SMOTE (Synthetic Minority Over-sampling Technique)** for better balancing.  
🔹 Implement **deep learning** models for improved accuracy.  
🔹 Deploy on **AWS/GCP** for real-world use.

---

### 📢 Thank You!
🚀 Happy Coding! 💻
