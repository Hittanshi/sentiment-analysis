import streamlit as st
import pickle
import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Load the trained model
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def predict_sentiment(review):
    processed_review = preprocess_text(review)
    prediction = model.predict([processed_review])
    return '😊 Positive' if prediction[0] == 1 else '☹️ Negative'

# Streamlit App UI
st.title("🎭 Sentiment Analysis App")
st.write("Enter a movie review to analyze its sentiment:")

# User Input
user_input = st.text_area("Write your review here:")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.subheader(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter a review before analyzing.")

