# College Feedback Classifier - Streamlit App (BERT Sentiment + Logistic Regression Category + Chunking + Highlights + Confidence)

import streamlit as st
import joblib
import pickle
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from transformers import pipeline

nltk.download('punkt')

# Load the classifier model and vectorizer
model = joblib.load("bert_feedback_model.pkl")
vectorizer = pickle.load(open("bert_vectorizer.pkl", "rb"))

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

# Sentiment classification with score per sentence

def classify_sentiment_chunkwise(text):
    sentences = sent_tokenize(text)
    sentiments = {"Positive": 0, "Negative": 0, "Neutral": 0}
    sentence_scores = []
    for sent in sentences:
        result = bert_sentiment_pipeline(sent[:512])[0]
        label = result['label']
        score = result['score']
        if label == 'POSITIVE' and score >= 0.75:
            sentiment = "Positive"
        elif label == 'NEGATIVE' and score >= 0.75:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        sentiments[sentiment] += 1
        sentence_scores.append((sent, sentiment, round(score, 3)))
    overall_sentiment = max(sentiments, key=sentiments.get)
    return overall_sentiment, sentence_scores

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
st.title("🤖 College Feedback Classifier (BERT + Highlights + Confidence)")
st.markdown("Analyze long student feedback using BERT for sentiment and Logistic Regression for categories.")

feedback = st.text_area("✍️ Enter your feedback:", height=150)

if st.button("🔍 Classify"):
    if feedback.strip() == "":
        st.warning("Please enter some feedback.")
    else:
        # Chunk-wise category prediction
        sentences = sent_tokenize(feedback)
        labels = ["Academics", "Facilities", "Administration"]
        predicted_labels = set()
        for sent in sentences:
            processed = preprocess_text(sent)
            vector = vectorizer.transform([processed])
            prediction = model.predict(vector)[0]
            for i in range(len(prediction)):
                if prediction[i] == 1:
                    predicted_labels.add(labels[i])

        predicted_labels = list(predicted_labels)

        # Keyword-based category boost
        feedback_lower = feedback.lower()
        category_keywords = {
            "Academics": ["subject", "subjects", "math", "mathematics", "science", "concept", "curriculum", "syllabus", "lecture", "teaching", "learning", "professor", "exam", "assignment", "notes", "coursework", "faculty", "class", "classes", "department"],
            "Facilities": ["library", "gym", "wifi", "room", "equipment", "bathroom", "hostel", "canteen", "classroom", "projector", "computer lab", "infrastructure", "printer", "cleaning", "maintenance", "hall", "building", "air conditioning", "ac", "labs"],
            "Administration": ["registration", "admission", "fees", "complaint", "office", "admin", "dean", "finance", "form", "schedule", "delay", "exam form", "staff", "management", "rules", "documents", "notice"]
        }
        for category, keywords in category_keywords.items():
            if any(re.search(rf"\\b{word}\\b", feedback_lower) for word in keywords):
                if category not in predicted_labels:
                    predicted_labels.append(category)

        sentiment, sentence_scores = classify_sentiment_chunkwise(feedback)

        st.subheader("📂 Predicted Categories:")
        if predicted_labels:
            st.success(", ".join(predicted_labels))
        else:
            st.warning("Could not determine categories.")

        st.subheader("💬 Overall Sentiment:")
        st.info(sentiment)

        st.subheader("🧠 Sentence-wise Sentiment & Confidence:")
        for sent, sent_type, score in sentence_scores:
            st.write(f"- _{sent}_ ➜ **{sent_type}** (Confidence: {score})")

        suggestions = get_suggestions(predicted_labels, sentiment)
        if suggestions:
            st.subheader("🛠 Suggested Improvements:")
            for s in suggestions:
                st.write("- " + s)

st.markdown("---")
st.caption("Built with Hugging Face BERT + Logistic Regression + Highlights + Confidence")
