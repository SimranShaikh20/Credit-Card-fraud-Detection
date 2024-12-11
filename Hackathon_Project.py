import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Importing data
data=pd.read_csv("creditcard.csv")
legit=data[data.Class==0]
fraud=data[data['Class']==1]
x=data.drop('Class',axis=1)
y=data['Class']

legit_s=legit.sample(n=len(fraud),random_state=2)
data=pd.concat([legit_s,fraud],axis=0)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)

model=LogisticRegression()
model.fit(x_train,y_train)
# evalution of model performance
train_acc=accuracy_score(model.predict(x_train),y_train)
test_acc=accuracy_score(model.predict(x_test),y_test)

#web app
st.title("Credit Card Fraud Detection Model")
input_data=st.text_area("Enter All Required features Values here :")
input_s=input_data.split(',')

submit=st.button("Submit")

if submit:
    features=np_data=np.asarray(input_s,dtype=np.float64)
    detection=model.predict(features.reshape(1,-1))

    if detection[0]==0:
        st.write("Legitment")
    else:
        st.write("Fraud ")
