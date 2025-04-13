import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# UI
st.title("📩 Email Spam Detector")
msg = st.text_area("Enter a message:")
if st.button("Predict"):
    msg_tfidf = vectorizer.transform([msg])
    prediction = model.predict(msg_tfidf)
    st.write("Prediction:", "🚫 Spam" if prediction[0] else "✅ Not Spam")
