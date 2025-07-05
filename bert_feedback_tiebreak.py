
# College Feedback Classifier - Cyberpunk 2077 Style 🎮
import streamlit as st
import joblib
import pickle
import re
from nltk.stem import PorterStemmer
from transformers import pipeline

# --- Cyberpunk Theme Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');
body { background-color: #000000; }
.stApp { font-family: 'Orbitron', sans-serif; color: #39ff14; }
.title-text { text-align: center; font-size: 36px; font-weight: bold; color: #ff00ff; margin-top: 20px; }
.sub-title { text-align: center; font-size: 18px; color: #39ff14; margin-bottom: 30px; }
textarea {
    background-color: #111111 !important;
    color: #39ff14 !important;
    border: 2px solid #00ffff !important;
    border-radius: 10px !important;
    caret-color: #ff00ff !important;
    font-family: 'Orbitron', monospace !important;
    animation: glow-cursor 1s infinite;
}
@keyframes glow-cursor {
    0% { box-shadow: 0 0 5px #ff00ff; }
    50% { box-shadow: 0 0 10px #ff00ff; }
    100% { box-shadow: 0 0 5px #ff00ff; }
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
h5 { color: #ff00ff; font-weight: bold; }
em { color: #00ffff; font-style: italic; }
strong { color: #00ff99; }
span { color: #ffcc00; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

model = joblib.load("bert_feedback_model.pkl")
vectorizer = pickle.load(open("bert_vectorizer.pkl", "rb"))

@st.cache_resource(show_spinner=False)
def load_bert_pipeline():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

bert_sentiment_pipeline = load_bert_pipeline()

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

    counts = sentiments
    max_count = max(counts.values())
    top_sentiments = [k for k, v in counts.items() if v == max_count]

    if len(top_sentiments) > 1:
        avg_scores = {label: sum(score for _, lbl, score in sentence_scores if lbl == label) / counts[label]
                      for label in top_sentiments if counts[label] > 0}
        if avg_scores:
            highest_avg = max(avg_scores.values())
            dominant = [k for k, v in avg_scores.items() if v == highest_avg]
            if len(dominant) == 1:
                overall_sentiment = dominant[0]
                sentiment_hint = f"(⚠️ It was a tie, but it's leaning {overall_sentiment.lower()} based on confidence scores.)"
            else:
                overall_sentiment = "Neutral"
                sentiment_hint = "(⚖️ It's evenly balanced based on confidence.)"
        else:
            overall_sentiment = "Neutral"
            sentiment_hint = "(⚖️ It's evenly balanced.)"
    else:
        overall_sentiment = top_sentiments[0]
        sentiment_hint = ""

    return overall_sentiment, sentence_scores, sentiment_hint

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

st.markdown("<div class='title-text'>🕶️ College Feedback Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Cyberpunk-Style Sentiment & Category Analyzer</div>", unsafe_allow_html=True)

feedback = st.text_area("✍️ Enter Feedback:", height=150)

if st.button("🚀 Classify Feedback"):
    if feedback.strip() == "":
        st.warning("Please enter some feedback.")
    else:
        sentences = split_sentences(feedback)
        labels = ["Academics", "Facilities", "Administration"]
        category_confidence = {}

        for sent in sentences:
            processed = preprocess_text(sent)
            vector = vectorizer.transform([processed])
            probas = model.predict_proba(vector)[0]
            for i in range(len(probas)):
                label = labels[i]
                category_confidence[label] = max(category_confidence.get(label, 0), probas[i])

        filtered = {k: round(v * 100, 1) for k, v in category_confidence.items() if v >= 0.5}

        feedback_lower = feedback.lower()
        keyword_boost = {
            "Academics": ["subject", "math", "science", "concept", "curriculum", "syllabus", "lecture", "teaching", "learning", "professor", "exam", "assignment", "notes", "faculty", "class"],
            "Facilities": ["library", "gym", "wifi", "equipment", "bathroom", "hostel", "canteen", "projector", "labs"],
            "Administration": ["registration", "admission", "fees", "complaint", "admin", "dean", "finance", "schedule", "management"]
        }
        for category, keywords in keyword_boost.items():
            if any(re.search(rf"\b{word}\b", feedback_lower) for word in keywords):
                if category not in filtered:
                    filtered[category] = 55.0

        sentiment, sentence_scores, sentiment_hint = classify_sentiment_chunkwise(feedback)

        st.markdown("""<div class='result-box'>
<h5>📂 Predicted Categories</h5>
""", unsafe_allow_html=True)
        if filtered:
            for cat, score in filtered.items():
                bar = f"<div style='background:#222;border:1px solid #39ff14;border-radius:8px;overflow:hidden;margin:4px 0;'>"
                bar += f"<div style='width:{score}%;background:#39ff14;color:#000;padding:4px 8px;font-weight:600;'>{cat}: {score}%</div></div>"
                st.markdown(bar, unsafe_allow_html=True)
        else:
            st.markdown("<p>None</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""<div class='result-box'>
<h5>💬 Overall Sentiment</h5>
<p><strong>{sentiment}</strong> {sentiment_hint}</p>
</div>""", unsafe_allow_html=True)

        st.markdown("<h5>🧠 Sentence-wise Sentiment</h5>", unsafe_allow_html=True)
        for sent, sent_type, score in sentence_scores:
            st.markdown(f"<div class='result-box'><em>{sent}</em><br/><strong>{sent_type}</strong> (Confidence: <span>{score}</span>)</div>", unsafe_allow_html=True)

        suggestions = get_suggestions(list(filtered.keys()), sentiment)
        if suggestions:
            st.markdown("<h5>⚙️ Suggested Improvements</h5>", unsafe_allow_html=True)
            for s in suggestions:
                st.markdown(f"<div class='result-box'>- {s}</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption(" ----- BUILT BY SHUBHAM ----- ")
