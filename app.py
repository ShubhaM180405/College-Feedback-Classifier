import streamlit as st
import joblib
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK models
nltk.download('punkt')
nltk.download('wordnet')

# Load model and vectorizer
model = joblib.load("naive_bayes_model.pkl")  # ✅ your filename
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Preprocessing
import re
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Tokenize using regex instead of nltk's word_tokenize
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())  # Only letters
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized)


# Suggest improvements based on predictions
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
    elif sentiment == "Positive":
        suggestions.append("🎉 Keep up the great work!")
    return suggestions

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
        labels = model.classes_ if hasattr(model, 'classes_') else ["Academics", "Facilities", "Administration", "Sentiment"]
        predicted = [label for i, label in enumerate(labels) if prediction[i] == 1]

        # Extract sentiment if present
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        sentiment_score = analyzer.polarity_scores(feedback)['compound']

        if sentiment_score >= 0.05:
            sentiment = "Positive"
        elif sentiment_score <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"


        st.subheader("📂 Predicted Categories:")
        labels = ["Academics", "Facilities", "Administration"]  # <-- use string labels directly
        predicted_labels = [labels[i] for i, val in enumerate(prediction) if val == 1]

        # Now this will work:
        st.success(", ".join(predicted_labels) if predicted_labels else "None")


        st.subheader("💬 Sentiment:")
        st.info(sentiment)

        suggestions = get_suggestions(predicted, sentiment)
        if suggestions:
            st.subheader("🛠 Suggested Improvements:")
            for tip in suggestions:
                st.write("- " + tip)

st.markdown("---")
st.caption("Built with Streamlit · Multi-label NLP Classifier")
