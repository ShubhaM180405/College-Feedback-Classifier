# College Feedback Classifier - Streamlit App (Styled BERT Version)

import streamlit as st
import joblib
import pickle
import re
from nltk.stem import PorterStemmer
from transformers import pipeline

# --- Custom Styling ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(to bottom right, #f2f8ff, #d0eaff);
    }
    .stApp {
        font-family: 'Segoe UI', sans-serif;
    }
    .title-text {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #154360;
    }
    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #3d5a80;
        margin-bottom: 20px;
    }
    .feedback-box textarea {
        background-color: #fdfefe;
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #ccc;
    }
    .result-box {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and vectorizer
model = joblib.load("bert_feedback_model.pkl")
vectorizer = pickle.load(open("bert_vectorizer.pkl", "rb"))

@st.cache_resource(show_spinner=False)
def load_bert_pipeline():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

bert_sentiment_pipeline = load_bert_pipeline()

# Split sentences (regex, not nltk)
def split_sentences(text):
    return re.split(r'(?<=[.!?]) +', text.strip())

stemmer = PorterStemmer()
def preprocess_text(text):
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    stemmed = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed)

def classify_sentiment_chunkwise(text):
    sentences = split_sentences(text)
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

# --- Streamlit UI ---
st.markdown("<div class='title-text'>📚 College Feedback Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Analyze sentiment & categories from student feedback using AI</div>", unsafe_allow_html=True)

feedback = st.text_area("✍️ Enter Feedback:", height=150, help="Write your comment about academics, facilities, or administration.")

if st.button("🔍 Classify Feedback"):
    if feedback.strip() == "":
        st.warning("Please enter some feedback.")
    else:
        sentences = split_sentences(feedback)
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

        st.markdown("""<div class='result-box'>
        <h5>📂 Predicted Categories</h5>
        <p><strong>{}</strong></p>
        </div>""".format(", ".join(predicted_labels) if predicted_labels else "None"), unsafe_allow_html=True)

        st.markdown("""<div class='result-box'>
        <h5>💬 Overall Sentiment</h5>
        <p><strong>{}</strong></p>
        </div>""".format(sentiment), unsafe_allow_html=True)

        st.markdown("<h5>🧠 Sentence-wise Sentiment</h5>", unsafe_allow_html=True)
        for sent, sent_type, score in sentence_scores:
            st.markdown(f"<div class='result-box'><em>{sent}</em><br/><strong>{sent_type}</strong> (Confidence: {score})</div>", unsafe_allow_html=True)

        suggestions = get_suggestions(predicted_labels, sentiment)
        if suggestions:
            st.markdown("<h5>🛠 Suggested Improvements</h5>", unsafe_allow_html=True)
            for s in suggestions:
                st.markdown(f"<div class='result-box'>- {s}</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("BUILT BY SHUBHAM..")
