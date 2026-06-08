import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="SECAP")
st.title("SECAP: Smart Email Classifier")
st.write("Enter an email message to classify it as 'spam' or 'ham'.")

#input box
user_input = st.text_area("Email Message")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        response = requests.post(API_URL, json={"text": user_input})

        if response.status_code == 200:
            result = response.json()

            st.success(f"Prediction: {result['predicted_label'].upper()}")
            st.info(f"Confidence: {result['confidence']}")
        else:
            st.error("Error occurred while making prediction. Please try again.")
            
