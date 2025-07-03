# College Feedback Classifier - Web App using Streamlit with VADER Sentiment

import streamlit as st
import joblib
import pickle
import re
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load model and vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Initialize tools
lemmatizer = WordNetLemmatizer()
sentiment_analyzer = SentimentIntensityAnalyzer()

# Preprocessing function (no punkt required)
def preprocess_text(text):
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized)

# Get suggestions based on category + sentiment
def get_suggestions(categories, sentiment):
    suggestions = []
    if sentiment == "Negative":
        if "Facilities" in categories:
            suggestions.append("🔧 Improve campus facilities and services.")
        if "Faculty" in categories:
            suggestions.append("👩‍🏫 Enhance teaching quality and interaction.")
        if "Academics" in categories:
            suggestions.append("📘 Provide better academic support or clarity.")
    elif sentiment == "Neutral":
        suggestions.append("🙂 Could use more engagement or support.")
    elif sentiment == "Positive" and categories:
        suggestions.append("🎉 Keep up the great work!")
    return suggestions

# Analyze sentiment using VADER
def get_sentiment(text):
    scores = sentiment_analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return "Positive"
    elif scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# --- Streamlit UI ---
st.set_page_config(page_title="College Feedback Classifier")
st.title("🎓 College Feedback Classifier")
st.markdown("Enter student feedback and classify it into multiple categories and sentiment.")

feedback = st.text_area("✍️ Enter your feedback here:", height=150)

if st.button("🔍 Classify"):
    if feedback.strip() == "":
        st.warning("Please enter some feedback text.")
    else:
        processed = preprocess_text(feedback)
        vector = vectorizer.transform([processed])
        prediction = model.predict(vector)[0]

        # Fallback class names
        labels = ["Academics", "Facilities", "Administration"]  # manually define labels
        predicted_labels = [labels[i] for i in range(len(prediction)) if prediction[i] == 1]

        sentiment = get_sentiment(feedback)

        st.subheader("📂 Predicted Categories:")
        if predicted_labels:
            st.success(", ".join(predicted_labels))
        else:
            st.warning("⚠️ Could not classify the feedback. Try rephrasing or improve training data.")

        st.subheader("💬 Sentiment:")
        st.info(sentiment)

        suggestions = get_suggestions(predicted_labels, sentiment)
        if suggestions and predicted_labels:
            st.subheader("🛠 Suggested Improvements:")
            for tip in suggestions:
                st.write("- " + tip)

st.markdown("---")
st.caption("Built with Streamlit · Multi-label NLP Classifier with VADER")
