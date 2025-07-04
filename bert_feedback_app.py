# College Feedback Classifier - Streamlit App (BERT Sentiment + Logistic Regression Category)

import streamlit as st
import joblib
import pickle
import re
from nltk.stem import PorterStemmer
from transformers import pipeline

# Load the classifier model and vectorizer
model = joblib.load("logistic_feedback_model.pkl")
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load the BERT sentiment pipeline
@st.cache_resource(show_spinner=False)
def load_bert_pipeline():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

bert_sentiment_pipeline = load_bert_pipeline()

# Initialize tools
stemmer = PorterStemmer()

# Preprocessing

def preprocess_text(text):
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    stemmed = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed)

# BERT + heuristic-based 3-class sentiment

def get_sentiment(text):
    result = bert_sentiment_pipeline(text[:512])[0]
    label = result['label']
    score = result['score']
    if label == 'POSITIVE' and score >= 0.75:
        return "Positive"
    elif label == 'NEGATIVE' and score >= 0.75:
        return "Negative"
    else:
        return "Neutral"

# Suggestions based on sentiment + category

def get_suggestions(categories, sentiment):
    suggestions = []
    if sentiment == "Negative":
        if "Facilities" in categories:
            suggestions.append("🔧 Improve campus facilities and services.")
        if "Academics" in categories:
            suggestions.append("📘 Provide better academic support or clarity.")
        if "Administration" in categories:
            suggestions.append("🗂 Improve administrative responsiveness and processes.")
    elif sentiment == "Neutral":
        suggestions.append("🙂 Could use more engagement or support.")
    elif sentiment == "Positive" and categories:
        suggestions.append("🎉 Keep up the great work!")
    return suggestions

# Streamlit UI
st.set_page_config(page_title="BERT Feedback Classifier")
st.title("🤖 College Feedback Classifier (BERT Sentiment)")
st.markdown("Classify feedback using BERT sentiment and Logistic Regression category.")

feedback = st.text_area("✍️ Enter your feedback:", height=150)

if st.button("🔍 Classify"):
    if feedback.strip() == "":
        st.warning("Please enter some feedback.")
    else:
        processed = preprocess_text(feedback)
        vector = vectorizer.transform([processed])
        prediction = model.predict(vector)[0]

        labels = ["Academics", "Facilities", "Administration"]
        predicted_labels = [labels[i] for i in range(len(prediction)) if prediction[i] == 1]

        # Keyword-based category boost
        feedback_lower = feedback.lower()
        category_keywords = {
            "Academics": ["subject", "subjects", "math", "science", "concept", "syllabus", "lecture"],
            "Facilities": ["library", "gym", "wifi", "room", "equipment"],
            "Administration": ["registration", "admission", "fees", "complaint", "office"]
        }
        for category, keywords in category_keywords.items():
            if any(re.search(rf"\\b{word}\\b", feedback_lower) for word in keywords):
                if category not in predicted_labels:
                    predicted_labels.append(category)

        sentiment = get_sentiment(feedback)

        st.subheader("📂 Predicted Categories:")
        if predicted_labels:
            st.success(", ".join(predicted_labels))
        else:
            st.warning("Could not determine categories.")

        st.subheader("💬 Sentiment:")
        st.info(sentiment)

        suggestions = get_suggestions(predicted_labels, sentiment)
        if suggestions:
            st.subheader("🛠 Suggested Improvements:")
            for s in suggestions:
                st.write("- " + s)

st.markdown("---")
st.caption("Built with Hugging Face BERT + Logistic Regression")
