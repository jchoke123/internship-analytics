"""
=============================================================
PART 1: Internship Application Data Tracking
=============================================================
What this script does:
  1. Generates a realistic sample dataset of ~80 internship applications
  2. Cleans and preprocesses the data using Pandas
  3. Standardises categories and formats dates
  4. Exports the clean dataset as a CSV file

Tools used: Python, Pandas, NumPy, Random
=============================================================
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# STEP 1: GENERATE SAMPLE DATASET
# ─────────────────────────────────────────────
# Why: Real-world projects start with messy, real data.
# We simulate that here — including intentional inconsistencies
# (mixed case, typos, missing values) that we'll clean later.

random.seed(42)   # Ensures results are reproducible every time you run this
np.random.seed(42)

# --- Define realistic value pools ---

companies = [
    "Deloitte", "PwC", "KPMG", "EY", "Goldman Sachs",
    "JP Morgan", "DBS Bank", "OCBC", "UOB", "Grab",
    "Shopee", "Sea Limited", "Accenture", "McKinsey",
    "Boston Consulting Group", "Lazada", "Singtel",
    "ST Engineering", "CapitaLand", "Temasek Holdings",
    "Mastercard", "Visa", "Stripe", "Palantir", "Meta",
]

roles = [
    "Data Analyst Intern", "Finance Intern", "Accounting Intern",
    "Business Analyst Intern", "Risk Analyst Intern",
    "Data Science Intern", "Audit Intern", "Strategy Intern",
    "Operations Intern", "Product Analyst Intern",
]

# Intentionally messy — mixed capitalisation, abbreviations
# This simulates how real data often looks before cleaning
statuses_messy = [
    "applied", "Applied", "APPLIED",
    "interview", "Interview", "INTERVIEW",
    "rejected", "Rejected", "REJECTED",
    "offer",                              # small chance of offer
]

resume_types_messy = [
    "tailored", "Tailored", "TAILORED",
    "generic", "Generic", "GENERIC",
]

industries = [
    "Finance", "Technology", "Consulting", "Banking",
    "E-Commerce", "Telecommunications", "Real Estate",
    "Government", "Healthcare",
]

# --- Assign weights to statuses (realistic distribution) ---
# Most applications end in rejection; few reach interview or offer
status_weights   = [10, 10, 10, 15, 15, 15, 20, 20, 20, 2]
resume_weights   = [20, 20, 20, 15, 15, 15]   # slightly more tailored

# --- Generate rows ---
n = 80  # number of applications

data = {
    "company_name": [random.choice(companies) for _ in range(n)],
    "role":         [random.choice(roles) for _ in range(n)],

    # Random application dates spread across ~8 months
    "date_applied": [
        (datetime(2024, 1, 1) + timedelta(days=random.randint(0, 240))).strftime("%d/%m/%Y")
        for _ in range(n)
    ],

    # Messy statuses — will be cleaned in Step 2
    "status": random.choices(statuses_messy, weights=status_weights, k=n),

    # Response time: 0–60 days; some NaN to simulate missing data
    "response_time_days": [
        round(random.uniform(1, 60)) if random.random() > 0.15 else np.nan
        for _ in range(n)
    ],

    # Messy resume types — will be standardised
    "resume_type": random.choices(resume_types_messy, weights=resume_weights, k=n),

    "industry": [random.choice(industries) for _ in range(n)],
}

# Create a Pandas DataFrame from the dictionary above
# A DataFrame is essentially a table — rows are applications, columns are attributes
df_raw = pd.DataFrame(data)

print("=" * 55)
print("STEP 1 COMPLETE: Raw dataset generated")
print(f"  Shape : {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
print("\nFirst 5 rows (raw, messy):")
print(df_raw.head())
print("\nData types:")
print(df_raw.dtypes)
print("\nMissing values per column:")
print(df_raw.isnull().sum())


# ─────────────────────────────────────────────
# STEP 2: CLEAN & PREPROCESS
# ─────────────────────────────────────────────
# Why: Dirty data produces wrong insights.
# We fix capitalisation, handle missing values, and correct types.

df = df_raw.copy()   # Always work on a copy — keep the original intact

# --- 2a. Standardise text columns to Title Case ---
# .str.strip()  → removes accidental leading/trailing spaces
# .str.title()  → converts "APPLIED" / "applied" → "Applied"
text_cols = ["company_name", "role", "industry"]
for col in text_cols:
    df[col] = df[col].str.strip().str.title()

# --- 2b. Standardise categorical columns to lowercase ---
# We want "applied" not "Applied" — lowercase is conventional for categories
df["status"]      = df["status"].str.strip().str.lower()
df["resume_type"] = df["resume_type"].str.strip().str.lower()

# --- 2c. Convert date column to proper datetime format ---
# Why: Stored as strings (e.g. "15/03/2024"), we can't do date math on strings.
# pd.to_datetime() converts them to real date objects.
# dayfirst=True tells Pandas our format is DD/MM/YYYY
df["date_applied"] = pd.to_datetime(df["date_applied"], dayfirst=True)

# --- 2d. Handle missing response times ---
# Strategy: fill NaN with the column median (robust to outliers)
# Why median and not mean? Mean is skewed by extreme values (e.g. 60-day outliers)
median_response = df["response_time_days"].median()
df["response_time_days"] = df["response_time_days"].fillna(median_response)

# Round to whole number for readability
df["response_time_days"] = df["response_time_days"].round(0).astype(int)

print("\n" + "=" * 55)
print("STEP 2 COMPLETE: Data cleaned")
print(f"\n  Unique statuses     : {sorted(df['status'].unique())}")
print(f"  Unique resume types : {sorted(df['resume_type'].unique())}")
print(f"  Date range          : {df['date_applied'].min().date()} → {df['date_applied'].max().date()}")
print(f"  Missing values left : {df.isnull().sum().sum()}")


# ─────────────────────────────────────────────
# STEP 3: ENGINEER ADDITIONAL COLUMNS
# ─────────────────────────────────────────────
# Why: Derived columns make SQL queries and Power BI much simpler later.

# Month name (e.g. "January") — useful for timeline charts
df["month_applied"] = df["date_applied"].dt.strftime("%B")

# Month number (1–12) — for correct chronological sorting
df["month_num"] = df["date_applied"].dt.month

# Year (in case your data spans multiple years)
df["year_applied"] = df["date_applied"].dt.year

# Binary flag: did this application get a response? (interview or offer counts)
df["got_response"] = df["status"].isin(["interview", "offer"]).astype(int)
# → 1 = yes, 0 = no

# Status rank — useful for funnel ordering in Power BI
status_rank = {"applied": 1, "interview": 2, "offer": 3, "rejected": 0}
df["status_rank"] = df["status"].map(status_rank)

print("\n" + "=" * 55)
print("STEP 3 COMPLETE: Feature engineering done")
print(f"\nNew columns added: month_applied, month_num, year_applied,")
print(f"                   got_response, status_rank")
print(f"\nStatus distribution (cleaned):")
print(df["status"].value_counts())
print(f"\nResume type distribution:")
print(df["resume_type"].value_counts())


# ─────────────────────────────────────────────
# STEP 4: EXPORT TO CSV
# ─────────────────────────────────────────────
# Why: CSV is the universal format — works with SQLite, Power BI, Excel, etc.

output_path = "internship_applications.csv"
df.to_csv(output_path, index=False)
# index=False → don't write the row numbers (0, 1, 2...) as a column

print("\n" + "=" * 55)
print(f"STEP 4 COMPLETE: Clean dataset exported")
print(f"  File  : {output_path}")
print(f"  Shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nFinal column list:")
for col in df.columns:
    print(f"  - {col} ({df[col].dtype})")

print("\n✅ Part 1 complete. Ready for SQL analysis (Part 2).")
