import streamlit as st
import joblib
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np

nltk.download('punkt')
nltk.download('wordnet')

model = joblib.load("naive_bayes_model.joblib")
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    return " ".join(lemmatized)

def get_suggestions(categories, sentiment):
    suggestions = []
    if sentiment == "Negative":
        if "Facilities" in categories:
            suggestions.append("Consider improving facility maintenance and accessibility.")
        if "Faculty" in categories:
            suggestions.append("Enhance faculty training or availability.")
        if "Academics" in categories:
            suggestions.append("Revise academic curriculum or offer extra help sessions.")
    elif sentiment == "Neutral":
        suggestions.append("Could benefit from more interactive events or support programs.")
    elif sentiment == "Positive":
        suggestions.append("Keep up the good work in highlighted areas!")
    return suggestions

st.set_page_config(page_title="College Feedback Classifier")
st.title("🎓 College Feedback Classifier")
st.write("Enter your feedback and receive automatic category and sentiment classification.")

feedback_input = st.text_area("Your Feedback", height=150)

if st.button("Classify Feedback"):
    if feedback_input.strip() == "":
        st.warning("Please enter some feedback text.")
    else:
        processed = preprocess_text(feedback_input)
        vector = vectorizer.transform([processed])
        prediction = model.predict(vector).toarray()[0]

        labels = model.classes_ if hasattr(model, 'classes_') else ["Academics", "Facilities", "Administration", "Sentiment"]
        predicted_labels = [label for i, label in enumerate(labels) if prediction[i] == 1]

        sentiment = "Positive"
        for s in ["Negative", "Neutral"]:
            if s in predicted_labels:
                sentiment = s
                predicted_labels.remove(s)
                break

        st.subheader("Predicted Categories")
        st.write(", ".join(predicted_labels) if predicted_labels else "None")

        st.subheader("Predicted Sentiment")
        st.write(sentiment)

        suggestions = get_suggestions(predicted_labels, sentiment)
        if suggestions:
            st.subheader("Suggested Improvements")
            for tip in suggestions:
                st.info(tip)

st.markdown("---")
st.caption("Built with Streamlit | NLP Multi-label Classifier")
