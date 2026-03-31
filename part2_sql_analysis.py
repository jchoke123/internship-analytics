"""
=============================================================
PART 2: SQL Analysis Layer (SQLite)
=============================================================
What this script does:
  1. Loads the clean CSV into a SQLite database
  2. Creates a proper SQL table
  3. Runs 6 analytical queries with results printed clearly

Tools used: Python, Pandas, SQLite3 (built into Python — no install needed)
=============================================================
"""

import sqlite3
import pandas as pd

# ─────────────────────────────────────────────
# STEP 1: LOAD CSV INTO SQLITE DATABASE
# ─────────────────────────────────────────────
# Why SQLite?
# It's a lightweight database that lives as a single file on your computer.
# No server setup needed. Perfect for portfolios and small projects.
# sqlite3 is built into Python — nothing extra to install.

# Connect to (or create) a database file
# If "applications.db" doesn't exist, SQLite creates it automatically
conn = sqlite3.connect("applications.db")

# Load the clean CSV into a Pandas DataFrame first
df = pd.read_csv("data/internship_applications.csv")

# Write the DataFrame into a SQL table called "applications"
# if_exists="replace" → drops and recreates the table on each run (safe for dev)
# index=False        → don't write the Pandas row index as a column
df.to_sql("applications", conn, if_exists="replace", index=False)

print("=" * 60)
print("STEP 1 COMPLETE: CSV loaded into SQLite database")
print(f"  Database file : applications.db")
print(f"  Table name    : applications")
print(f"  Rows loaded   : {len(df)}")


# ─────────────────────────────────────────────
# HELPER FUNCTION
# ─────────────────────────────────────────────
# This small function runs any SQL query and prints the result cleanly.
# We'll reuse it for every query below.

def run_query(title, sql, conn):
    print("\n" + "=" * 60)
    print(f"QUERY: {title}")
    print("-" * 60)
    result = pd.read_sql_query(sql, conn)
    print(result.to_string(index=False))   # index=False = cleaner printout
    return result


# ─────────────────────────────────────────────
# QUERY 1: OVERALL APPLICATION COUNTS BY STATUS
# ─────────────────────────────────────────────
# Insight: Tells you how your applications are distributed.
# Are most still "applied" (no response)? How many reached interview stage?
# This is the foundation of your application funnel.

q1 = """
SELECT
    status,
    COUNT(*)                                    AS total_applications,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS percentage
FROM applications
GROUP BY status
ORDER BY
    CASE status
        WHEN 'applied'   THEN 1
        WHEN 'interview' THEN 2
        WHEN 'offer'     THEN 3
        WHEN 'rejected'  THEN 4
    END;
"""
# Note: OVER() is a window function — it calculates the percentage
# against the TOTAL count, not just within each group.

run_query("Application Counts by Status", q1, conn)


# ─────────────────────────────────────────────
# QUERY 2: RESPONSE RATE (INTERVIEW RATE)
# ─────────────────────────────────────────────
# Insight: Of all applications sent, what % received a meaningful response?
# (i.e. progressed to interview or offer — not just silence or rejection)
# This is a key KPI for your Power BI dashboard.

q2 = """
SELECT
    COUNT(*)                                            AS total_applied,
    SUM(got_response)                                   AS total_responses,
    ROUND(SUM(got_response) * 100.0 / COUNT(*), 1)     AS response_rate_pct
FROM applications;
"""
# got_response is the 0/1 flag we created in Part 1.
# SUM(got_response) counts all the 1s = applications that got a response.

run_query("Overall Response Rate (Interview Rate)", q2, conn)


# ─────────────────────────────────────────────
# QUERY 3: SUCCESS RATE BY RESUME TYPE
# ─────────────────────────────────────────────
# Insight: Does tailoring your resume actually improve results?
# This directly answers one of the most common internship questions.
# Great talking point in interviews: "I tracked and proved it with data."

q3 = """
SELECT
    resume_type,
    COUNT(*)                                                AS total_sent,
    SUM(got_response)                                       AS responses,
    ROUND(SUM(got_response) * 100.0 / COUNT(*), 1)         AS response_rate_pct,
    SUM(CASE WHEN status = 'interview' THEN 1 ELSE 0 END)  AS interviews,
    SUM(CASE WHEN status = 'offer'     THEN 1 ELSE 0 END)  AS offers
FROM applications
GROUP BY resume_type
ORDER BY response_rate_pct DESC;
"""
# CASE WHEN is SQL's version of an if-statement.
# SUM(CASE WHEN status = 'interview' THEN 1 ELSE 0 END)
# → counts rows where status is 'interview', ignores everything else.

run_query("Success Rate by Resume Type", q3, conn)


# ─────────────────────────────────────────────
# QUERY 4: APPLICATIONS OVER TIME (MONTHLY)
# ─────────────────────────────────────────────
# Insight: When were you most active in applying?
# Reveals application patterns — useful for planning future job searches.
# This feeds directly into the timeline chart in Power BI.

q4 = """
SELECT
    month_applied,
    month_num,
    COUNT(*)                                            AS applications_sent,
    SUM(got_response)                                   AS responses_received,
    ROUND(SUM(got_response) * 100.0 / COUNT(*), 1)     AS monthly_response_rate
FROM applications
GROUP BY month_applied, month_num
ORDER BY month_num;
"""
# We ORDER BY month_num (1,2,3...) not month_applied ("April","August"...)
# because alphabetical ordering would give wrong results.

run_query("Applications Over Time (Monthly)", q4, conn)


# ─────────────────────────────────────────────
# QUERY 5: AVERAGE RESPONSE TIME
# ─────────────────────────────────────────────
# Insight: How long does each company/industry typically take to respond?
# Helps set realistic expectations and follow-up timing.

q5 = """
SELECT
    industry,
    COUNT(*)                                    AS applications,
    ROUND(AVG(response_time_days), 1)           AS avg_response_days,
    MIN(response_time_days)                     AS fastest_days,
    MAX(response_time_days)                     AS slowest_days
FROM applications
GROUP BY industry
ORDER BY avg_response_days ASC;
"""
# AVG(), MIN(), MAX() are aggregate functions — they collapse many rows
# into a single summary value per group.

run_query("Average Response Time by Industry", q5, conn)


# ─────────────────────────────────────────────
# QUERY 6: INDUSTRY BREAKDOWN
# ─────────────────────────────────────────────
# Insight: Which industries are you targeting most?
# Which have the best response rates?
# Helps identify where to focus future applications.

q6 = """
SELECT
    industry,
    COUNT(*)                                            AS total_applications,
    SUM(CASE WHEN status = 'interview' THEN 1 ELSE 0 END) AS interviews,
    SUM(CASE WHEN status = 'offer'     THEN 1 ELSE 0 END) AS offers,
    SUM(CASE WHEN status = 'rejected'  THEN 1 ELSE 0 END) AS rejections,
    ROUND(SUM(got_response) * 100.0 / COUNT(*), 1)        AS response_rate_pct
FROM applications
GROUP BY industry
ORDER BY total_applications DESC;
"""

run_query("Industry Breakdown", q6, conn)


# ─────────────────────────────────────────────
# BONUS: TOP 5 MOST APPLIED-TO COMPANIES
# ─────────────────────────────────────────────

q7 = """
SELECT
    company_name,
    COUNT(*)    AS applications,
    GROUP_CONCAT(DISTINCT role) AS roles_applied
FROM applications
GROUP BY company_name
ORDER BY applications DESC
LIMIT 5;
"""
# GROUP_CONCAT is SQLite-specific — it joins multiple values into one string.
# Useful for showing all roles applied to at the same company.

run_query("Top 5 Most Applied-To Companies", q7, conn)


# ─────────────────────────────────────────────
# CLOSE CONNECTION
# ─────────────────────────────────────────────
conn.close()

print("\n" + "=" * 60)
print("✅ Part 2 complete. Database saved as applications.db")
print("   Ready for Power BI Dashboard (Part 3).")
