import streamlit as st
import pickle
import numpy as np

# Load Model and Vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Streamlit UI
st.title("📩 Spam Detection App")
st.write("Enter a message below to check if it's spam or not.")

# Text Input
user_input = st.text_area("🔤 Enter your message here:")

# Prediction
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a valid message!")
    else:
        # Transform input text using the vectorizer
        user_vector = vectorizer.transform([user_input])
        
        # Predict using the model
        prediction = model.predict(user_vector)[0]
        
        # Show Result
        if prediction == 1:
            st.error("🚨 This message is **SPAM**!")
        else:
            st.success("✅ This message is **NOT SPAM**!")

# Footer
st.write("Built with ❤️ using Streamlit & Scikit-Learn")
