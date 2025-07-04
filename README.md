# College-Feedback-Classifier
# 🎓 College Feedback Classifier

A machine learning-powered feedback analyzer for educational institutions. It classifies open-ended student feedback into categories and detects sentiment using NLP.

---

## 🧠 What It Does

### 🗂 Categories (Multi-label Classification)
- 📘 Academics
- 🏢 Facilities
- 🗂 Administration

✅ Models:
- `naive_bayes_model.pkl` (baseline)
- `logistic_feedback_model.pkl` (improved)
- `bert_feedback_model.pkl` (for BERT sentiment version)

### 💬 Sentiment Analysis
- Uses `DistilBERT` from Hugging Face (`bert_feedback_app.py`)
- Labels: Positive / Negative / Neutral
- Sentence-level tone detection with confidence scores

---

## 📄 App Files

| File Name                         | Description                                |
|----------------------------------|--------------------------------------------|
| `app.py`                         | Naive Bayes version                        |
| `college_feedback_appLogReg.py`  | Logistic Regression version                |
| `bert_feedback_app.py`           | Final app with BERT sentiment + highlights |
| `requirements.txt`               | Dependencies for Streamlit Cloud           |

---

# 🎓 College Feedback Classifier

## 🔗 Live Demo
- You can Try the Basic Version here 👉 [College Feedback Classifier (Streamlit App)](https://college-feedback-classifier-naive-bayes.streamlit.app/)
- You can Try another version (Logistic Regression version) here 👉 [College Feedback Classifier (Streamlit App)](https://college-feedback-classifier-logistic-regression.streamlit.app/)
- You can Try the final version (BERT version) here 👉 [College Feedback Classifier (Streamlit App)](https://college-feedback-classifier-bert.streamlit.app/)

- A machine learning-powered feedback analyzer for educational institutions...

---

## 💬 Example Input & Output

- 📥 Sample Feedback:
The syllabus is outdated and the classrooms are too hot. But the professors are really helpful.

- ✅ Output:
   1) 📂 Predicted Categories: Academics, Facilities
    2) 💬 Overall Sentiment: Negative
     3) 🧠 Sentence-level Tone:

- "The syllabus is outdated." → Negative (Confidence: 0.94)
- "The classrooms are too hot." → Negative (Confidence: 0.92)
- "But the professors are really helpful." → Positive (Confidence: 0.91)

---

## 🚀 How to Run

### 🔧 Local (optional)
```bash
pip install -r requirements.txt
streamlit run bert_feedback_app.py

