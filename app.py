import streamlit as st
import pickle
import re
import time
import random
import matplotlib.pyplot as plt
import numpy as np

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Load the trained model and metrics
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Sample model metrics (replace with actual values if available)
accuracy = 0.92
precision = 0.89
recall = 0.91
f1_score = 0.90

# Sentiment counts (for visualization)
sentiment_counts = {"Positive": 0, "Negative": 0}

def predict_sentiment(review):
    processed_review = preprocess_text(review)
    prediction = model.predict([processed_review])
    sentiment = '😊 Positive' if prediction[0] == 1 else '☹️ Negative'
    
    # Update sentiment counts
    sentiment_counts["Positive" if prediction[0] == 1 else "Negative"] += 1
    return sentiment

# Streamlit App UI
st.markdown(
    """
    <style>
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

# Metrics Display
st.subheader("📊 Model Performance Metrics")
st.metric(label="Accuracy", value=f"{accuracy*100:.2f}%")
st.metric(label="Precision", value=f"{precision*100:.2f}%")
st.metric(label="Recall", value=f"{recall*100:.2f}%")
st.metric(label="F1 Score", value=f"{f1_score*100:.2f}%")

# Sentiment Distribution Graph
st.subheader("📈 Sentiment Distribution")
fig, ax = plt.subplots()
ax.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['green', 'red'])
ax.set_ylabel("Count")
ax.set_title("Sentiment Analysis Results")
st.pyplot(fig)

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

    </div>
    """,
    unsafe_allow_html=True
)

