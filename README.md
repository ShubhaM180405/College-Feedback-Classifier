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

## 🚀 How to Run

### 🔧 Local (optional)
```bash
pip install -r requirements.txt
streamlit run bert_feedback_app.py

## 👨‍💻 Author

Developed by **Shubham Bej**

> Built using Streamlit · Hugging Face BERT · scikit-learn
