import streamlit as st
import pickle
import re
import time
import random

# Function to preprocess text
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
st.markdown(
    """
    <style>
        @keyframes backgroundAnimation {
            0% {background-color: #ffcccc;}
            25% {background-color: #ffccff;}
            50% {background-color: #ccccff;}
            75% {background-color: #ccffcc;}
            100% {background-color: #ffcccc;}
        }
        
        .main {
            padding: 20px;
            border-radius: 10px;
        }
        .stTextArea textarea {
            
            border-radius: 5px;
        }
        .footer {
            position: fixed;
            bottom: 10px;
            width: 100%;
            text-align: center;
            font-size: 14px;
            color: gray;
        }
    </style>
    <div class='main' id='main-container'>
    """,
    unsafe_allow_html=True
)

st.title("🎭 Sentiment Analysis App")
st.write("Enter a movie review to analyze its sentiment:")

# User Input
user_input = st.text_area("Write your review here:")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.subheader(f"Sentiment: {sentiment}")
        
        # Change background color and show balloons for positive sentiment
        if "Positive" in sentiment:
            st.balloons()
            color = "#ccffcc"  # Green for positive
        else:
            color = "#ffcccc"  # Red for negative
        
        st.markdown(
            f"""
            <style>
                .main {{ background-color: {color} !important; }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Please enter a review before analyzing.")

# Footer
st.markdown(
    """
    <div class='footer'>
        Created by: Hittanshi Tikle | Mehek Uikey | Daulat Tikhole
    </div>
    </div>
    """,
    unsafe_allow_html=True
)

