# College Feedback Classifier

import streamlit as st
import joblib
import pickle
import re
from nltk.stem import PorterStemmer
from transformers import pipeline

st.markdown("""
    <style>
    body {
        background-color: #000000;
    }
    .stApp {
        font-family: 'Orbitron', sans-serif;
        color: #39ff14; /* Neon Green */
    }
    .title-text {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #ff00ff; /* Magenta */
        margin-top: 20px;
    }
    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #39ff14;
        margin-bottom: 30px;
    }
    .feedback-box textarea {
        background-color: #111111;
        color: #39ff14;
        border: 2px solid #00ffff; /* Cyan */
        border-radius: 10px;
    }
    .result-box {
        background-color: #0a0a0a;
        border: 2px solid #39ff14;
        border-radius: 10px;
        padding: 15px;
        margin-top: 15px;
        color: #39ff14;
        font-size: 16px;
        font-weight: 500;
        box-shadow: 0 0 12px #00ffff;
    }
    h5 {
        color: #ff00ff;
        font-weight: bold;
    }
    em {
        color: #00ffff;
        font-style: italic;
    }
    strong {
        color: #00ff99;
    }
    span {
        color: #ffcc00;
        font-weight: 600;
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

# Sentence splitting
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
            suggestions.append("⚠️ Upgrade gym, labs, or classroom facilities.")
        if "Academics" in categories:
            suggestions.append("📚 Improve curriculum or teaching methods.")
        if "Administration" in categories:
            suggestions.append("🛠 Streamline administrative services.")
    elif sentiment == "Neutral":
        suggestions.append("🤔 Could improve engagement or clarity.")
    elif sentiment == "Positive" and categories:
        suggestions.append("🎉 Keep up the great work!")
    return suggestions

# --- Streamlit UI ---
st.markdown("<div class='title-text'>🕶️ College Feedback Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Cyberpunk-Style Sentiment & Category Analyzer</div>", unsafe_allow_html=True)

st.markdown('<div class="feedback-box">', unsafe_allow_html=True)
feedback = st.text_area("✍️ Enter Feedback:", height=150, label_visibility="collapsed", key="feedback_area")
st.markdown('</div>', unsafe_allow_html=True)

if st.button("🚀 Classify Feedback"):
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
            "Academics": ["subject", "math", "science", "concept", "curriculum", "syllabus", "lecture", "teaching", "learning", "professor", "exam", "assignment", "notes", "faculty", "class"],
            "Facilities": ["library", "gym", "wifi", "equipment", "bathroom", "hostel", "canteen", "projector", "labs"],
            "Administration": ["registration", "admission", "fees", "complaint", "admin", "dean", "finance", "schedule", "management"]
        }
        for category, keywords in category_keywords.items():
            if any(re.search(rf"\b{word}\b", feedback_lower) for word in keywords):
                if category not in predicted_labels:
                    predicted_labels.append(category)

        sentiment, sentence_scores = classify_sentiment_chunkwise(feedback)

        st.markdown(f"""<div class='result-box'>
        <h5>📂 Predicted Categories</h5>
        <p><strong>{', '.join(predicted_labels) if predicted_labels else 'None'}</strong></p>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class='result-box'>
        <h5>💬 Overall Sentiment</h5>
        <p><strong>{sentiment}</strong></p>
        </div>""", unsafe_allow_html=True)

        st.markdown("<h5>🧠 Sentence-wise Sentiment</h5>", unsafe_allow_html=True)
        for sent, sent_type, score in sentence_scores:
            st.markdown(f"<div class='result-box'><em>{sent}</em><br/><strong>{sent_type}</strong> (Confidence: <span>{score}</span>)</div>", unsafe_allow_html=True)

        suggestions = get_suggestions(predicted_labels, sentiment)
        if suggestions:
            st.markdown("<h5>⚙️ Suggested Improvements</h5>", unsafe_allow_html=True)
            for s in suggestions:
                st.markdown(f"<div class='result-box'>- {s}</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption(" ----- BUILT BY SHUBHAM ----- ")
