# College Feedback Classifier - Streamlit App with VADER + Keyword Boost + Stemming

import streamlit as st
import joblib
import pickle
import re
from nltk.stem import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load model and vectorizer
model = joblib.load("Basic_naive_bayes_feedback_model.pkl")
vectorizer = pickle.load(open("Basic_naive_bayes_vectorizer.pkl", "rb"))

# Initialize tools
stemmer = PorterStemmer()
sentiment_analyzer = SentimentIntensityAnalyzer()

# Preprocessing (using stemmer to avoid wordnet errors)
def preprocess_text(text):
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    stemmed = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed)

# Sentiment detection using VADER + keyword override
def get_sentiment(text):
    score = sentiment_analyzer.polarity_scores(text)['compound']
    negative_keywords = [
        "outdated", "old", "broken", "slow", "expensive", "unsafe", "unreliable",
        "inadequate", "unavailable", "unhelpful", "long wait", "no response",
        "takes too long", "inefficient", "dirty", "poor", "difficult", "confusing",
        "crashes", "problem", "issue", "not working", "low quality", "needs improvement",
        "insufficient", "doesn't work", "unfair", "lack", "delayed", "missing",
        "late", "crowded", "limited", "overwhelmed", "stressful", "bad", "slow response", "long queues", "wait"
    ]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05 or any(word in text.lower() for word in negative_keywords):
        return "Negative"
    else:
        return "Neutral"

# Improvement suggestion logic
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

# --- Streamlit Interface ---
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

        labels = ["Academics", "Facilities", "Administration"]
        predicted_labels = [labels[i] for i in range(len(prediction)) if prediction[i] == 1]
        

        sentiment = get_sentiment(feedback)
        # Keyword-based soft boost for categories
        feedback_lower = feedback.lower()
        
        # Define keyword triggers
        category_keywords = {
            "Academics": ["math", "mathematics", "science", "subjects", "concept", "curriculum", "teaching", "learning", "syllabus", "professor", "lecture", "exam", "assignment"],
            "Facilities": ["library", "gym", "wifi", "bathroom", "elevator", "hostel", "ac", "equipment", "room", "building", "printer", "cleaning", "laundry", "sports"],
            "Administration": ["registration", "admission", "fees", "complaint", "delay",
                                "office", "admin", "dean", "finance", "portal"  ]
        }
        
        # Soft add categories if matching keywords are found
        for category, keywords in category_keywords.items():
            if any(re.search(rf"\\b{word}\\b", feedback_lower) for word in keywords):
                if category not in predicted_labels:
                    predicted_labels.append(category)

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
st.caption("Built with Streamlit · NLP Classifier + VADER Sentiment + Keyword Boost. Built by - SHUBHAM.")
