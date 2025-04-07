# nlp_matcher.py

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re

# NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# ---------------------------

# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation and numbers
    tokens = nltk.word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)

# ---------------------------
# Similarity score calculator
# ---------------------------
def calculate_similarity(jd, resume):
    texts = [clean_text(jd), clean_text(resume)]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(score * 100, 2)  # return as percentage

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="Resume Matcher", layout="centered")
st.title("🧠 NLP-Powered Resume & Job Description Matcher")

st.markdown("""
Paste a **Job Description** and a **Resume**, and get a matching score using AI-powered text analysis.
""")

# Text inputs
job_description = st.text_area("📄 Job Description", height=250)
resume_text = st.text_area("🧾 Resume", height=250)

# Match button
if st.button("🔍 Match Resume to Job"):
    if not job_description.strip() or not resume_text.strip():
        st.warning("Please fill in both fields before submitting.")
    else:
        with st.spinner("Analyzing..."):
            score = calculate_similarity(job_description, resume_text)
        st.success(f"✅ Match Score: **{score}%**")
        
        # Score Feedback
        if score > 80:
            st.balloons()
            st.markdown("🎉 **Excellent Match!** This resume aligns very well with the job.")
        elif score > 60:
            st.markdown("👍 **Good Match** – With a few tweaks, it could be perfect.")
        elif score > 40:
            st.markdown("⚠️ **Average Match** – Consider tailoring your resume more.")
        else:
            st.markdown("❌ **Low Match** – Your resume may need a complete rewrite.")

st.markdown("---")
st.markdown("💡 *Powered by TF-IDF and Cosine Similarity in Python*")
