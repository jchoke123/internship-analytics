"""
=============================================================
PART 4A: AI Resume–Job Matching Tool — NLP Prototype (v1)
=============================================================
PORTFOLIO NOTE — WHY THIS FILE EXISTS:

This was my first attempt at resume-job matching, built using
classical NLP techniques (TF-IDF + cosine similarity).

After building and testing it, I identified several fundamental
limitations that led me to rethink the entire approach:

  1. QUANTITATIVE OVER QUALITATIVE
     The tool produces a numeric match score (e.g. 67.3%).
     This implies false precision — a recruiter doesn't think
     "you're a 67.3% fit." What's actually useful is qualitative
     context: what's strong, what's missing, and what to do about it.
     → In v2 (app.py), I replaced the score with a 4-tier verdict
       system (Exceptional / Strong / Solid with Gaps / Needs Work)
       with written explanations for each.

  2. KEYWORD MATCHING IS TOO RIGID
     TF-IDF matches exact words, not meaning. "Data viz" and
     "visualisation" mean the same thing but score as a gap.
     Worse, it picks up noise words like "able", "accurate",
     "strong" as skill gaps — making the output unreliable and
     confusing to the user.
     → In v2, I replaced TF-IDF gap detection with an LLM that
       reads context and identifies only real, meaningful gaps.

  3. MATCHED KEYWORDS ARE REDUNDANT
     Showing a list of matched keywords like "data, python, analysis"
     doesn't tell the user anything they don't already know — they
     wrote those words on their resume. It adds visual clutter
     without adding insight.
     → In v2, I removed matched keywords entirely and focused
       the output on what actually matters: the gaps and how to fix them.

  4. TONE IS CLINICAL, NOT HELPFUL
     A list of "missing keywords" feels like an ATS rejection,
     not career advice. It tells you what's wrong but not how
     to feel about it or what to do next.
     → In v2, I built a conversational coaching experience with
       a warm, encouraging tone — asking clarifying questions,
       recovering hidden experience, and writing specific bullet
       points the user can copy-paste into their resume.

  5. STATIC OUTPUT MISSES HIDDEN EXPERIENCE
     The biggest flaw: a keyword tool can only see what's written
     on the resume. It can't know that you used SQL in coursework
     but forgot to include it, or that your internship involved
     stakeholder presentations even if you didn't use that word.
     → In v2, the coach asks targeted clarifying questions to
       surface this hidden experience and suggests exactly how
       to add it to the resume.

CONCLUSION:
  These limitations all point to the same insight — a powerful LLM
  with a conversational interface is fundamentally better suited to
  this problem than keyword matching. The full rebuild is in app.py.
  This file is kept to document the thinking process and iteration.

=============================================================
WHAT THIS SCRIPT DOES (v1 approach):
  - Takes resume text + job description as input
  - Uses TF-IDF to extract keywords from each document
  - Computes a match score using cosine similarity + keyword overlap
  - Outputs matched/missing keywords and basic suggestions

TOOLS USED: Python, scikit-learn (TF-IDF + cosine similarity)
RUN WITH:   python3 part4_resume_matcher.py
=============================================================
"""

import os
import re
import json
import urllib.request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# SAMPLE INPUTS
# ─────────────────────────────────────────────
# Hardcoded for demonstration purposes.
# In v2 (app.py), the user uploads their own PDF resume
# and pastes a real job description interactively.

RESUME_TEXT = """
John Tan
Accounting & Data Science Student | National University of Singapore

EDUCATION
Bachelor of Accountancy + Bachelor of Science (Data Science)
National University of Singapore | Expected 2026

SKILLS
- Programming: Python, R, SQL
- Data Tools: Pandas, NumPy, Matplotlib
- Accounting: Financial Reporting, Auditing, Tax
- Microsoft Office: Excel (VLOOKUP, Pivot Tables), Word, PowerPoint
- Soft Skills: Communication, Teamwork, Attention to Detail

EXPERIENCE
Data Analytics Intern | ABC Consulting | May 2023 – Aug 2023
- Cleaned and analysed datasets using Python and Pandas
- Built Excel dashboards for client reporting
- Assisted in preparing financial statements

Project: Personal Finance Tracker
- Built a Python application to track spending using CSV data
- Visualised trends using Matplotlib

CERTIFICATIONS
- Google Data Analytics Certificate
- Excel for Data Analysis (Coursera)
"""

JOB_DESCRIPTION = """
Data Analyst Intern – DBS Bank

About the Role:
We are looking for a motivated Data Analyst Intern to join our
Consumer Banking Analytics team. You will work with large datasets
to generate insights that drive business decisions.

Responsibilities:
- Analyse customer transaction data using SQL and Python
- Build and maintain dashboards in Power BI or Tableau
- Support the team in data cleaning and data preprocessing
- Present findings to stakeholders using clear visualisations
- Collaborate with cross-functional teams on analytics projects
- Assist in financial modelling and reporting

Requirements:
- Pursuing a degree in Data Science, Statistics, Computer Science,
  Accounting, Finance, or related field
- Proficient in SQL and Python
- Experience with data visualisation tools (Power BI, Tableau, or similar)
- Knowledge of Excel and financial analysis
- Strong analytical and problem-solving skills
- Good communication and presentation skills
- Experience with machine learning or statistical modelling is a plus
- Familiarity with banking or financial services preferred
"""


# ─────────────────────────────────────────────
# STEP A: TEXT PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(text):
    """
    Cleans text for NLP processing:
    - Lowercase everything
    - Remove punctuation and numbers
    - Collapse extra whitespace
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─────────────────────────────────────────────
# STEP B: KEYWORD EXTRACTION USING TF-IDF
# ─────────────────────────────────────────────
# LIMITATION IDENTIFIED: TF-IDF frequently extracts noise words
# ("able", "accurate", "strong") as high-scoring terms, polluting
# the gap analysis. It also cannot distinguish between semantically
# equivalent phrases like "data viz" and "visualisation".
# → Addressed in v2 by using LLM-powered gap detection instead.

def extract_keywords(text, top_n=30):
    """
    Extracts top keywords using TF-IDF scoring.
    ngram_range=(1,2) captures both single words and
    two-word phrases like "power bi", "machine learning".
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=200
    )
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    return [word for word, score in ranked[:top_n] if score > 0]


# ─────────────────────────────────────────────
# STEP C: COMPUTE MATCH SCORE
# ─────────────────────────────────────────────
# LIMITATION IDENTIFIED: The 50/50 weighting between cosine
# similarity and keyword overlap is arbitrary. The resulting
# numeric score (e.g. 67.3%) implies false precision and is not
# meaningful to a real candidate or recruiter.
# → Addressed in v2 with a qualitative 4-tier verdict system.

def compute_match(resume_text, jd_text):
    """
    Computes a match score using two methods combined:
    1. Cosine similarity — overall language overlap (0 to 1)
    2. Keyword overlap  — % of JD keywords found in resume
    Final score = (cosine × 0.5) + (overlap × 0.5) × 100
    """
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    cosine_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    jd_keywords     = set(extract_keywords(jd_text, top_n=30))
    resume_keywords = set(extract_keywords(resume_text, top_n=50))

    matched       = jd_keywords & resume_keywords
    missing       = jd_keywords - resume_keywords
    overlap_score = len(matched) / len(jd_keywords) if jd_keywords else 0

    final_score = round((cosine_score * 0.5 + overlap_score * 0.5) * 100, 1)
    return final_score, sorted(matched), sorted(missing)


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_matcher(resume, jd):
    """
    Runs the full v1 matching pipeline and prints results.
    """
    print("=" * 60)
    print("   AI RESUME-JOB MATCHING — v1 NLP Prototype")
    print("   See app.py for the improved v2 career coach")
    print("=" * 60)

    resume_clean = preprocess(resume)
    jd_clean     = preprocess(jd)
    score, matched, missing = compute_match(resume_clean, jd_clean)

    print(f"\n📊 MATCH SCORE: {score}%")
    print(f"   Note: This numeric score is a known limitation of v1.")
    print(f"   See PORTFOLIO NOTE at top of file for full explanation.")

    print(f"\n❌ SKILL GAPS DETECTED ({len(missing)}):")
    print("   " + ", ".join(missing) if missing else "   None detected")
    print(f"\n   Note: TF-IDF gap detection is unreliable — it may include")
    print(f"   noise words and miss semantically equivalent terms.")
    print(f"   → v2 uses LLM-powered gap detection to fix this.")

    print("\n" + "=" * 60)
    print("KEY LIMITATIONS OF THIS v1 APPROACH:")
    print("  1. Quantitative score implies false precision")
    print("  2. Keyword matching is too rigid — misses semantic meaning")
    print("  3. Matched keywords shown to user are redundant")
    print("  4. Clinical output — no coaching tone or encouragement")
    print("  5. Cannot surface experience not written on the resume")
    print("\n  All of these were addressed in app.py (v2).")
    print("  → These limitations are what motivated the full rebuild.")
    print("=" * 60)


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_matcher(RESUME_TEXT, JOB_DESCRIPTION)
