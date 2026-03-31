"""
=============================================================
AI CAREER COACH — v2 (app.py)
=============================================================
PORTFOLIO NOTE — HOW THIS EVOLVED FROM part4_resume_matcher.py:

After building and testing the v1 NLP prototype, I identified
5 key limitations that led to a complete rebuild:

  PROBLEM 1: Quantitative score → false precision
  SOLUTION:  Replaced numeric score with a 4-tier qualitative
             verdict (Exceptional / Strong / Solid with Gaps /
             Needs Targeted Work) with written explanations.

  PROBLEM 2: TF-IDF keyword matching → too rigid, extracts noise
  SOLUTION:  Replaced with LLM-powered gap detection that reads
             context and identifies only real, meaningful gaps.

  PROBLEM 3: Matched keywords shown to user → redundant
  SOLUTION:  Removed entirely. Focused output on what matters:
             the gaps and exactly how to address each one.

  PROBLEM 4: Clinical output → no coaching tone
  SOLUTION:  Built a conversational coaching experience with a
             warm, encouraging tone throughout. The tool feels
             like talking to a knowledgeable senior friend, not
             an ATS system.

  PROBLEM 5: Static analysis → misses hidden experience
  SOLUTION:  Added dynamic clarifying questions that surface
             experience the user forgot to include, then writes
             exact resume bullets they can copy-paste.

RESULT: A full AI Career Coach web app that goes far beyond
keyword matching — it conducts a structured coaching session,
generates personalised qualitative reports, and tracks multiple
role analyses in one session.

=============================================================
FEATURES:
  - Upload resume once — reused across all roles in a session
  - Quick Assessment (instant) vs Full Coaching Session (Q&A)
  - LLM-powered gap detection (Groq — free API)
  - Dynamic clarifying questions with open text answers
  - Colour-coded report cards with 4-tier verdict system
  - Sidebar navigation between past role analyses
  - Per-role chat history saved and reviewable
  - Meaningful session summary with insights per role
  - Warm, encouraging coaching tone throughout

TOOLS: Python, Streamlit, Groq API (LLaMA 3.3 70B + 3.1 8B),
       PyPDF2, JSON structured outputs
RUN:   streamlit run app.py
=============================================================
"""

import re
import json
import streamlit as st
import PyPDF2
from groq import Groq

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Career Coach",
    page_icon="🎯",
    layout="centered"
)

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .coach-title  { font-size: 2.8rem; font-weight: 700; color: #1a1a2e; margin:0; text-align:center; padding-top:0.5rem }
    .coach-sub    { color: #555; font-size: 1.15rem; margin-top:0.5rem; text-align:center; margin-bottom:0.5rem }
    .mode-card    { border:1.5px solid #e9ecef; border-radius:12px; padding:1.1rem 1.5rem; margin:0.5rem 0; }
    .mode-title   { font-weight:600; font-size:1.05rem; margin-bottom:0.3rem; }
    .mode-desc    { color:#555; font-size:0.95rem; line-height:1.5; }
    .role-card    { background:white; border:1px solid #e9ecef; border-radius:10px; padding:0.85rem 1.1rem; margin:0.4rem 0; }
    .role-card-active { background:#eff6ff; border:1.5px solid #3b82f6; border-radius:10px; padding:0.85rem 1.1rem; margin:0.4rem 0; }
    .tag-exceptional { background:#fef9c3; color:#713f12; padding:5px 16px; border-radius:20px; font-size:0.92rem; font-weight:700; display:inline-block; }
    .tag-strong   { background:#d4edda; color:#155724; padding:5px 16px; border-radius:20px; font-size:0.92rem; font-weight:700; display:inline-block; }
    .tag-moderate { background:#fff3cd; color:#856404; padding:5px 16px; border-radius:20px; font-size:0.92rem; font-weight:700; display:inline-block; }
    .tag-weak     { background:#f8d7da; color:#721c24; padding:5px 16px; border-radius:20px; font-size:0.92rem; font-weight:700; display:inline-block; }
    .report-card  { border-radius:12px; padding:1.4rem 1.6rem; margin-bottom:1.1rem; }
    .block-container { padding-top:2.5rem !important; padding-bottom:2rem !important; max-width:780px !important; }
    p, li, .stMarkdown { font-size:1.05rem !important; line-height:1.8 !important; }
    .stChatMessage p { font-size:1.05rem !important; line-height:1.8 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# Key design: each application stores its own
# chat_history and report_data separately.
# "active_app_idx" controls which role is shown.
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "stage":           "welcome",
        "resume_text":     None,
        "resume_name":     None,
        "current_jd":      None,
        "current_role":    None,
        "current_company": None,
        "mode":            None,
        "questions":       [],
        "answers":         [],
        "q_index":         0,
        "gaps":            [],
        # applications: list of dicts, each with its own chat + report
        "applications":    [],
        # active_app_idx: which app we're viewing in sidebar (-1 = current session)
        "active_app_idx":  -1,
        # current session chat (not yet saved to an application)
        "chat_history":    [],
        "report_data":     None,
        "report_str":      None,
        # one-shot flags
        "welcomed":        False,
        "mode_prompted":   False,
        "qa_prompted":     False,
        "summary_shown":   False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────
# GROQ CLIENT
# ─────────────────────────────────────────────
def get_client():
    try:
        key = st.secrets.get("GROQ_API_KEY", None)
        if key:
            return Groq(api_key=key)
    except Exception:
        pass
    return None

COACH_SYSTEM = """You are a warm, encouraging AI career coach helping university
students land internships. Your tone is always:
- Supportive and patient — like a knowledgeable senior friend
- Honest but kind — highlight gaps gently, always with solutions
- Specific — give advice tailored to THIS person, not generic tips
- Encouraging — especially when the resume needs work
Never make the user feel bad. Always end on an encouraging note."""

# ─────────────────────────────────────────────
# API CALLS
# ─────────────────────────────────────────────
def call_llm(prompt, max_tokens=1500):
    client = get_client()
    if not client: return None
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": COACH_SYSTEM},
                {"role": "user",   "content": prompt}
            ]
        )
        return resp.choices[0].message.content
    except Exception:
        return None


def call_llm_fast(prompt, max_tokens=400):
    client = get_client()
    if not client: return None
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": COACH_SYSTEM},
                {"role": "user",   "content": prompt}
            ]
        )
        return resp.choices[0].message.content
    except Exception:
        return None

# ─────────────────────────────────────────────
# PDF
# ─────────────────────────────────────────────
def extract_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        return "\n".join(p.extract_text() or "" for p in reader.pages).strip()
    except Exception as e:
        st.error(f"Could not read PDF: {e}")
        return None

# ─────────────────────────────────────────────
# GAP DETECTION
# ─────────────────────────────────────────────
def detect_gaps(resume, jd):
    prompt = f"""
Analyse a student's resume against a job description.
JOB DESCRIPTION: {jd[:1000]}
RESUME: {resume[:1000]}

Identify 3-5 real skill gaps. Rules:
- Only REAL skills (e.g. "SQL", "Power BI", "financial modelling")
- NOT vague words like "able", "strong", "good", "experience"
- NOT things already in the resume
Return ONLY a valid JSON array. Example: ["Power BI", "SQL"]
"""
    resp = call_llm_fast(prompt, 300)
    if not resp: return []
    try:
        clean = re.sub(r'```json|```', '', resp).strip()
        gaps = json.loads(clean)
        vague = {"able","strong","good","excellent","various","using","used","based",
                 "required","motivated","degree","years","accurate","general","desired"}
        return [g for g in gaps if g.lower() not in vague and len(g) > 3]
    except Exception:
        return []

# ─────────────────────────────────────────────
# QUESTION GENERATION
# ─────────────────────────────────────────────
def generate_questions(resume, jd, gaps, n):
    prompt = f"""
You are a friendly career coach. Resume: {resume[:600]}
Job: {jd[:500]}. Gaps: {', '.join(gaps) if gaps else 'none'}.

Generate exactly {n} warm open-ended questions about the real gaps.
Sound natural. Help surface forgotten experience. Never ask about vague words.
Return ONLY a valid JSON array of question strings.
"""
    resp = call_llm_fast(prompt, 500)
    if not resp:
        return [f"Tell me about your experience with {g} — even coursework counts!" for g in gaps[:n]]
    try:
        clean = re.sub(r'```json|```', '', resp).strip()
        qs = json.loads(clean)
        bad = ["with able","with strong","with good","with accurate"]
        valid = [q for q in qs if not any(b in q.lower() for b in bad)]
        while len(valid) < n:
            valid.append("Is there any relevant experience not currently on your resume?")
        return valid[:n]
    except Exception:
        return [f"Tell me about your experience with {g}!" for g in gaps[:n]]

# ─────────────────────────────────────────────
# REPORT GENERATION — returns structured dict
# ─────────────────────────────────────────────
def generate_report(resume, jd, role, company, gaps, questions, answers, mode):
    qa_section = ""
    if mode == "full" and questions and answers:
        qa_section = "CONVERSATION:\n" + "\n".join(
            [f"Coach: {q}\nStudent: {a}" for q, a in zip(questions, answers)]
        )
    prompt = f"""
Write a career coaching report for a university student.
ROLE: {role} at {company}
JD: {jd[:800]}
RESUME: {resume[:1000]}
GAPS: {', '.join(gaps) if gaps else 'none'}
{qa_section}

Return a JSON object with EXACTLY these keys. Be warm, specific, concise (2-3 bullets per section):
{{
  "verdict": MUST be exactly one of these four — be CONSERVATIVE and honest:
    "Exceptional Match" — only if resume already covers 80%+ of requirements with strong relevant experience. Very rare.
    "Strong Candidate" — good alignment, only 1-2 minor gaps. Resume is genuinely competitive.
    "Solid with Gaps" — clear potential but 3+ real gaps that need addressing before applying.
    "Needs Targeted Work" — significant gaps, resume needs substantial improvement for this role.
    When in doubt, go one tier LOWER. It is better to be honest than falsely encouraging.
  "verdict_text": "One honest, encouraging sentence explaining the verdict and what it means for the application.",
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "gaps": [
    {{"skill": "name", "why": "why it matters for this role", "fix": "concrete action"}}
  ],
  "hidden": ["ready-to-paste resume bullet" or empty list],
  "tips": [
    {{"section": "Experience", "tip": "specific tip"}},
    {{"section": "Skills", "tip": "specific tip"}},
    {{"section": "Education/Projects", "tip": "specific tip"}}
  ],
  "action_plan": ["step 1", "step 2", "step 3"],
  "encouragement": "One warm closing sentence."
}}
Return ONLY valid JSON. No markdown, no extra text.
"""
    resp = call_llm(prompt, 2000)
    if not resp:
        return _fallback_report(role, company, gaps)
    try:
        clean = re.sub(r'```json|```', '', resp).strip()
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        return json.loads(match.group() if match else clean)
    except Exception:
        return _fallback_report(role, company, gaps)


def _fallback_report(role, company, gaps):
    return {
        "verdict": "Solid Candidate",
        "verdict_text": f"You've got real potential for the {role} role at {company}.",
        "strengths": ["Strong academic background relevant to this role",
                      "Combination of accounting and data skills is genuinely rare",
                      "Hands-on project experience demonstrates initiative"],
        "gaps": [{"skill": g, "why": "Listed as a key requirement", "fix": f"Add '{g}' to your skills section"} for g in (gaps[:2] if gaps else [])],
        "hidden": [],
        "tips": [{"section": "Experience", "tip": "Quantify every bullet with numbers"},
                 {"section": "Skills", "tip": "List all tools explicitly"},
                 {"section": "Education/Projects", "tip": "Add business impact to each project"}],
        "action_plan": ["Add missing skills to your skills section",
                        "Quantify 3 experience bullets with real numbers",
                        "Rewrite profile summary to mirror this JD"],
        "encouragement": "You're on the right track — keep going! 💪"
    }


def generate_qa_response(question, resume, jd, report_str):
    prompt = f"""
Student question: {question}
Resume: {resume[:400]}
Role: {jd[:200]}
Report summary: {report_str[:400]}
Answer warmly, specifically, helpfully. Give real actionable advice. Be concise.
"""
    return call_llm(prompt, 600) or "Great question! Feel free to ask me anything more specific! 😊"

# ─────────────────────────────────────────────
# REPORT CARD RENDERER
# ─────────────────────────────────────────────
def render_report_cards(report_data, role, company, mode, collapsed=False):
    """
    Renders structured report as colour-coded cards.
    If collapsed=True, wraps everything in a st.expander.
    """
    label = "Quick Assessment" if mode == "quick" else "Coaching Report"

    if collapsed:
        verdict = report_data.get("verdict", "Solid Candidate")
        emoji = "🌟" if "Exceptional" in verdict else "✅" if "Strong" in verdict else "⚠️" if "Solid" in verdict else "🔧"
        with st.expander(f"{emoji} Review your {label} — {role} at {company}"):
            _render_cards(report_data)
    else:
        st.markdown(f"## 📋 Your {label}")
        st.markdown(f"### {role} at {company}")
        st.divider()
        _render_cards(report_data)
        st.divider()


def _render_cards(r):
    """Internal: renders all report section cards."""

    # Card 1 — Verdict
    verdict = r.get("verdict", "Solid Candidate")
    vtext   = r.get("verdict_text", "")
    if "Exceptional" in verdict:
        color, emoji, bg, border = "#713f12", "🌟", "#ffffff", "#f59e0b"
    elif "Strong" in verdict:
        color, emoji, bg, border = "#155724", "✅", "#ffffff", "#22c55e"
    elif "Solid" in verdict:
        color, emoji, bg, border = "#856404", "⚠️", "#ffffff", "#f59e0b"
    else:
        color, emoji, bg, border = "#721c24", "🔧", "#ffffff", "#ef4444"

    st.markdown(f"""
<div class="report-card" style="background:{bg};border:2px solid {border};border-left:6px solid {border}">
    <div style="font-weight:700;color:{color};font-size:1.1rem;margin-bottom:8px">{emoji} Overall Verdict — {verdict}</div>
    <div style="color:#374151;font-size:0.97rem;margin-bottom:12px;line-height:1.6">{vtext}</div>
    <div style="font-size:0.78rem;color:#9ca3af;border-top:1px solid #e5e7eb;padding-top:8px">
        Assessment scale: &nbsp;🌟 Exceptional &nbsp;·&nbsp; ✅ Strong Candidate &nbsp;·&nbsp; ⚠️ Solid with Gaps &nbsp;·&nbsp; 🔧 Needs Targeted Work
    </div>
</div>""", unsafe_allow_html=True)

    # Card 2 — Strengths
    strengths = r.get("strengths", [])
    if strengths:
        items = "".join([f"<li style='margin-bottom:5px'>{s}</li>" for s in strengths])
        st.markdown(f"""
<div class="report-card" style="background:#f0fdf4;border-left:4px solid #22c55e">
    <div style="font-weight:700;color:#166534;font-size:1rem;margin-bottom:8px">✅ What You're Doing Well</div>
    <ul style="margin:0;padding-left:1.2rem;color:#166534;font-size:0.92rem">{items}</ul>
</div>""", unsafe_allow_html=True)

    # Card 3 — Gaps
    gaps_data = r.get("gaps", [])
    if gaps_data:
        gap_html = ""
        for g in gaps_data:
            gap_html += f"""
<div style="margin-bottom:0.75rem;padding:0.75rem;background:rgba(255,255,255,0.6);border-radius:8px">
    <div style="font-weight:600;color:#9a3412;font-size:0.9rem">🔧 {g.get('skill','')}</div>
    <div style="color:#7c2d12;font-size:0.85rem;margin-top:3px"><b>Why it matters:</b> {g.get('why','')}</div>
    <div style="color:#7c2d12;font-size:0.85rem;margin-top:2px"><b>How to fix:</b> {g.get('fix','')}</div>
</div>"""
        st.markdown(f"""
<div class="report-card" style="background:#fff7ed;border-left:4px solid #f97316">
    <div style="font-weight:700;color:#9a3412;font-size:1rem;margin-bottom:8px">🔧 Areas to Strengthen</div>
    {gap_html}
</div>""", unsafe_allow_html=True)

    # Card 4 — Hidden Experience
    hidden = r.get("hidden", [])
    if hidden:
        items = "".join([f"<li style='margin-bottom:6px;font-family:monospace;font-size:0.85rem'>{h}</li>" for h in hidden])
        st.markdown(f"""
<div class="report-card" style="background:#eff6ff;border-left:4px solid #3b82f6">
    <div style="font-weight:700;color:#1e40af;font-size:1rem;margin-bottom:8px">💡 Add These to Your Resume</div>
    <ul style="margin:0;padding-left:1.2rem;color:#1e40af;font-size:0.92rem">{items}</ul>
</div>""", unsafe_allow_html=True)

    # Card 5 — Resume Tips
    tips = r.get("tips", [])
    if tips:
        tip_html = "".join([f"""
<div style="margin-bottom:0.6rem">
    <span style="font-weight:600;color:#5b21b6;font-size:0.88rem">📝 {t.get('section','')}</span>
    <span style="color:#4c1d95;font-size:0.88rem"> — {t.get('tip','')}</span>
</div>""" for t in tips])
        st.markdown(f"""
<div class="report-card" style="background:#f5f3ff;border-left:4px solid #7c3aed">
    <div style="font-weight:700;color:#5b21b6;font-size:1rem;margin-bottom:8px">📝 Section-by-Section Tips</div>
    {tip_html}
</div>""", unsafe_allow_html=True)

    # Card 6 — Action Plan
    action_plan = r.get("action_plan", [])
    if action_plan:
        items = "".join([f"<li style='margin-bottom:8px'><b>Step {i+1}:</b> {a}</li>" for i, a in enumerate(action_plan)])
        st.markdown(f"""
<div class="report-card" style="background:#fdf4ff;border-left:4px solid #a855f7">
    <div style="font-weight:700;color:#6b21a8;font-size:1rem;margin-bottom:8px">🚀 Your 3-Step Action Plan</div>
    <ol style="margin:0;padding-left:1.2rem;color:#6b21a8;font-size:0.92rem">{items}</ol>
</div>""", unsafe_allow_html=True)

    # Encouragement
    enc = r.get("encouragement", "You've got this! 💪")
    st.markdown(f"""
<div style="background:#f8fafc;border-radius:12px;padding:1rem 1.5rem;margin-bottom:1rem;text-align:center;color:#475569;font-style:italic">
    {enc}
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHAT HELPERS
# ─────────────────────────────────────────────
def add_msg(role, content):
    st.session_state.chat_history.append({"role": role, "content": content})

def show_chat():
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🎯" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    st.markdown("<h1 style='text-align:center;font-size:2.8rem;font-weight:800;color:#1a1a2e'>🎯 AI Career Coach</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#555;font-size:1.15rem;margin-top:-0.8rem'>Your personal internship advisor — upload once, analyse any role</p>", unsafe_allow_html=True)
    st.divider()

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("### 📋 My Applications")

        if not st.session_state.applications:
            st.caption("No roles analysed yet!")
        else:
            for i, app in enumerate(st.session_state.applications):
                v   = app.get("verdict", "")
                tag = "tag-exceptional" if "Exceptional" in v else "tag-strong" if "Strong" in v else "tag-moderate" if "Solid" in v else "tag-weak"
                is_active = st.session_state.active_app_idx == i

                st.markdown(f"""
<div style="background:{'#eff6ff' if is_active else 'white'};
     border:{'1.5px solid #3b82f6' if is_active else '1px solid #e9ecef'};
     border-radius:10px; padding:0.75rem 1rem; margin:0.4rem 0;">
    <strong>{app['company']}</strong><br>
    <small style="color:#666">{app['role']}</small><br>
    <span class="{tag}" style="margin-top:4px;display:inline-block">{v}</span>
</div>""", unsafe_allow_html=True)
                if st.button("👁 View report", key=f"view_{i}", use_container_width=True):
                    st.session_state.active_app_idx = i
                    st.rerun()

        if st.session_state.resume_name:
            st.divider()
            st.caption(f"📎 **{st.session_state.resume_name}**")

        st.divider()

        # New role button (only show if in qa stage)
        if st.session_state.stage == "qa":
            if st.button("➕ Analyse New Role", use_container_width=True, type="primary"):
                # Save current chat to current application
                if st.session_state.applications:
                    st.session_state.applications[-1]["chat_history"] = \
                        st.session_state.chat_history.copy()
                # Reset for new role
                for k in ["current_jd","current_role","current_company","questions","answers","report_data","report_str"]:
                    st.session_state[k] = [] if isinstance(st.session_state.get(k), list) else None
                st.session_state.q_index       = 0
                st.session_state.mode          = None
                st.session_state.mode_prompted = False
                st.session_state.qa_prompted   = False
                st.session_state.active_app_idx = -1
                st.session_state.chat_history  = []
                st.session_state.stage         = "get_jd"
                add_msg("assistant", f"Let's go! 🎯 Your resume (**{st.session_state.resume_name}**) is still loaded — just paste the new job description!")
                st.rerun()

        if st.button("🔄 Start Over", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ══════════════════════════════════════════
    # VIEWING A PAST APPLICATION
    # ══════════════════════════════════════════
    idx = st.session_state.active_app_idx
    if idx >= 0 and idx < len(st.session_state.applications):
        app = st.session_state.applications[idx]
        st.markdown(f"## 📋 {app['role']} at {app['company']}")

        v   = app.get("verdict", "")
        tag = "tag-exceptional" if "Exceptional" in v else "tag-strong" if "Strong" in v else "tag-moderate" if "Solid" in v else "tag-weak"
        st.markdown(f'<span class="{tag}">{v}</span>', unsafe_allow_html=True)
        st.divider()

        # Show report cards
        if app.get("report_data"):
            _render_cards(app["report_data"])
        st.divider()

        # Show chat history for this role
        if app.get("chat_history"):
            st.markdown("### 💬 Coaching Conversation")
            for msg in app["chat_history"]:
                with st.chat_message(msg["role"], avatar="🎯" if msg["role"] == "assistant" else "👤"):
                    st.markdown(msg["content"])

        if st.button("← Back to current session", type="primary"):
            st.session_state.active_app_idx = -1
            st.rerun()
        return  # Don't render the main flow when viewing past app

    # ══════════════════════════════════════════
    # MAIN COACHING FLOW
    # ══════════════════════════════════════════

    # Render current chat
    show_chat()

    # ── WELCOME ──
    if st.session_state.stage == "welcome":
        if not st.session_state.welcomed:
            add_msg("assistant", """
Hi there! 👋 I'm your AI Career Coach — really glad you're here.

Here's how I work:
- **Upload your resume once** — I'll keep it for the whole session
- **Paste any job description** — as many roles as you want
- **Choose your mode** — quick snapshot or a full coached session
- **Get personalised advice** — specific to your resume and the role

Let's start — upload your resume below! 😊
""")
            st.session_state.welcomed = True
            st.rerun()

        f = st.file_uploader("📎 Upload your resume (PDF)", type=["pdf"])
        if f:
            with st.spinner("Reading your resume..."):
                text = extract_pdf(f)
            if text:
                st.session_state.resume_text = text
                st.session_state.resume_name = f.name
                st.session_state.stage = "get_jd"
                add_msg("assistant", f"""
Got your resume — **{f.name}** ✅

You won't need to upload it again today. Now tell me about the role you're applying for!
Paste the job description below with the company name and role title. 💼
""")
                st.rerun()

    # ── GET JD ──
    elif st.session_state.stage == "get_jd":
        col1, col2 = st.columns(2)
        with col1:
            company = st.text_input("🏢 Company", placeholder="e.g. DBS Bank")
        with col2:
            role = st.text_input("💼 Role", placeholder="e.g. Data Analyst Intern")
        jd = st.text_area("📋 Paste job description", height=160,
                           placeholder="Paste the Requirements and Responsibilities sections...")

        if st.button("Continue →", type="primary", use_container_width=True):
            if not all([jd.strip(), company.strip(), role.strip()]):
                st.warning("Please fill in the company name, role title and job description.")
            else:
                st.session_state.current_jd      = jd
                st.session_state.current_role    = role
                st.session_state.current_company = company
                st.session_state.mode_prompted   = False
                st.session_state.stage           = "choose_mode"
                add_msg("user", f"I'm applying for **{role}** at **{company}**.")
                st.rerun()

    # ── CHOOSE MODE ──
    elif st.session_state.stage == "choose_mode":
        if not st.session_state.mode_prompted:
            add_msg("assistant", f"""
Got it — **{st.session_state.current_role}** at **{st.session_state.current_company}**!

How would you like to proceed? 👇
""")
            st.session_state.mode_prompted = True
            st.rerun()

        st.markdown("""
        <div class="mode-card">
            <div class="mode-title">⚡ Quick Assessment</div>
            <div class="mode-desc">Get your report instantly — no questions asked.</div>
        </div>
        <div class="mode-card">
            <div class="mode-title">🎓 Full Coaching Session</div>
            <div class="mode-desc">I'll ask you a few questions first for a more personalised report.</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⚡ Quick Assessment", use_container_width=True):
                st.session_state.mode  = "quick"
                st.session_state.stage = "detecting"
                add_msg("user", "I'll go with the Quick Assessment.")
                add_msg("assistant", "On it! Analysing your resume against this role... 🔍")
                st.rerun()
        with col2:
            if st.button("🎓 Full Coaching Session", type="primary", use_container_width=True):
                st.session_state.mode  = "full"
                st.session_state.stage = "detecting"
                add_msg("user", "I'd like the Full Coaching Session.")
                add_msg("assistant", "Perfect — give me a moment to review your resume... 🔍")
                st.rerun()

    # ── DETECTING ──
    elif st.session_state.stage == "detecting":
        with st.spinner("Analysing your resume... (~5 seconds)"):
            gaps = detect_gaps(st.session_state.resume_text, st.session_state.current_jd)
            st.session_state.gaps = gaps

            if st.session_state.mode == "full":
                n  = max(2, min(len(gaps), 4)) if gaps else 2
                qs = generate_questions(st.session_state.resume_text, st.session_state.current_jd, gaps, n)
                st.session_state.questions = qs
                st.session_state.q_index   = 0
                st.session_state.answers   = []
                st.session_state.stage     = "questioning"
                add_msg("assistant", f"""
I've reviewed your resume for the **{st.session_state.current_role}** role.

I have **{len(qs)} question{'s' if len(qs) > 1 else ''}** for you. No wrong answers — just share what feels relevant! 😊
""")
                st.rerun()
            else:
                st.session_state.stage = "reporting"
                st.rerun()

    # ── QUESTIONING ──
    elif st.session_state.stage == "questioning":
        idx = st.session_state.q_index
        qs  = st.session_state.questions

        if idx < len(qs):
            last_assistant = next(
                (m["content"] for m in reversed(st.session_state.chat_history) if m["role"] == "assistant"), ""
            )
            if qs[idx] not in last_assistant:
                add_msg("assistant", f"**Question {idx+1} of {len(qs)}:**\n\n{qs[idx]}")
                st.rerun()

            ans = st.chat_input("Your answer...")
            if ans:
                add_msg("user", ans)
                st.session_state.answers.append(ans)
                st.session_state.q_index += 1
                if st.session_state.q_index < len(qs):
                    add_msg("assistant", "Thanks for sharing that! Here's my next question:")
                else:
                    add_msg("assistant", "That's everything I needed — thank you! 🙏 Writing your report now...")
                    st.session_state.stage = "reporting"
                st.rerun()
        else:
            st.session_state.stage = "reporting"
            st.rerun()

    # ── REPORTING ──
    elif st.session_state.stage == "reporting":
        with st.spinner("Writing your coaching report... (~15 seconds)"):
            report_data = generate_report(
                st.session_state.resume_text,
                st.session_state.current_jd,
                st.session_state.current_role,
                st.session_state.current_company,
                st.session_state.gaps,
                st.session_state.questions,
                st.session_state.answers,
                st.session_state.mode
            )
            st.session_state.report_data = report_data
            st.session_state.report_str  = json.dumps(report_data)

            raw_v = report_data.get("verdict","")
            verdict = (
                raw_v if raw_v in ["Exceptional Match","Strong Candidate","Solid with Gaps","Needs Targeted Work"]
                else "Solid with Gaps"
            )

            # Save application with its own chat history + report
            st.session_state.applications.append({
                "company":      st.session_state.current_company,
                "role":         st.session_state.current_role,
                "verdict":      verdict,
                "report_data":  report_data,
                "chat_history": st.session_state.chat_history.copy(),
                "mode":         st.session_state.mode,
            })

            st.session_state.qa_prompted = False
            st.session_state.stage = "qa"
            st.rerun()

    # ── Q&A ──
    elif st.session_state.stage == "qa":
        # Show report collapsed by default
        if st.session_state.report_data:
            render_report_cards(
                st.session_state.report_data,
                st.session_state.current_role,
                st.session_state.current_company,
                st.session_state.mode,
                collapsed=True   # ← collapsed by default
            )

        if not st.session_state.qa_prompted:
            if st.session_state.mode == "quick":
                add_msg("assistant", """
That's your quick assessment! 😊

**Want to go deeper?** Say **"go deeper"** for a more personalised report with clarifying questions!

Or ask me anything — interview tips, how to word something, whether this role is right for you.
Say **"next role"** or click **➕ Analyse New Role** in the sidebar to analyse another position.

*When you're all set, just type **'exit'** and I'll wrap up with a session summary!* 🚀
""")
            else:
                add_msg("assistant", """
That's my full assessment! 😊

**Anything else you'd like to talk through?** Interview prep, resume wording, anything at all!

Say **"next role"** or click **➕ Analyse New Role** in the sidebar to analyse another position.

*When you're all set, just type **'exit'** and I'll wrap up with a session summary!* 🚀
""")
            st.session_state.qa_prompted = True
            st.rerun()

        user_in = st.chat_input("Ask me anything, or type 'exit' to end the session...")
        if user_in:
            add_msg("user", user_in)

            if user_in.strip().lower() == "exit":
                st.session_state.stage = "summary"
                add_msg("assistant", "Here's your session summary! 📊")

            elif "go deeper" in user_in.lower() and st.session_state.mode == "quick":
                st.session_state.mode  = "full"
                st.session_state.stage = "detecting"
                add_msg("assistant", "Let's dive deeper! Give me a moment... 🔍")

            elif any(x in user_in.lower() for x in ["next role","another role","new role"]):
                # Save chat to current application before switching
                if st.session_state.applications:
                    st.session_state.applications[-1]["chat_history"] = \
                        st.session_state.chat_history.copy()
                for k in ["current_jd","current_role","current_company","questions","answers","report_data","report_str"]:
                    st.session_state[k] = [] if isinstance(st.session_state.get(k), list) else None
                st.session_state.q_index       = 0
                st.session_state.mode          = None
                st.session_state.mode_prompted = False
                st.session_state.qa_prompted   = False
                st.session_state.active_app_idx = -1
                st.session_state.chat_history  = []
                st.session_state.stage         = "get_jd"
                add_msg("assistant", f"Let's go! 🎯 Your resume (**{st.session_state.resume_name}**) is still loaded — just paste the new job description!")

            else:
                with st.spinner("Thinking..."):
                    resp = generate_qa_response(
                        user_in,
                        st.session_state.resume_text,
                        st.session_state.current_jd or "",
                        st.session_state.report_str or ""
                    )
                # Add response + follow-up prompt
                add_msg("assistant", resp + "\n\n---\n💬 *Any other questions or clarifications? If you're all set, just type **'exit'** and I'll wrap up with a session summary!* 😊")

            # Update saved chat history for current application
            if st.session_state.applications:
                st.session_state.applications[-1]["chat_history"] = \
                    st.session_state.chat_history.copy()
            st.rerun()

    # ── SUMMARY ──
    elif st.session_state.stage == "summary":
        if not st.session_state.summary_shown:
            apps = st.session_state.applications
            st.markdown("## 🎯 Session Summary")
            st.markdown(f"You analysed **{len(apps)} role{'s' if len(apps)>1 else ''}** today. Here's what to take away:")
            st.divider()

            for app in apps:
                v   = app.get("verdict","")
                e   = "✅" if "Strong" in v else "⚠️" if "Solid" in v else "🔧"
                tag = "tag-exceptional" if "Exceptional" in v else "tag-strong" if "Strong" in v else "tag-moderate" if "Solid" in v else "tag-weak"
                rd  = app.get("report_data", {})

                st.markdown(f"### {e} {app['company']} — {app['role']}")
                st.markdown(f'<span class="{tag}">{v}</span>', unsafe_allow_html=True)
                st.markdown("")

                if rd:
                    # Strengths
                    strengths = rd.get("strengths", [])
                    if strengths:
                        st.markdown("**✅ Key strengths:**")
                        for s in strengths:
                            st.markdown(f"- {s}")

                    # Gaps
                    gaps = rd.get("gaps", [])
                    if gaps:
                        st.markdown("**🔧 Things to work on:**")
                        for g in gaps:
                            st.markdown(f"- **{g.get('skill','')}** — {g.get('fix','')}")

                    # Action plan
                    plan = rd.get("action_plan", [])
                    if plan:
                        st.markdown("**🚀 Priority actions:**")
                        for i, p in enumerate(plan):
                            st.markdown(f"{i+1}. {p}")

                st.divider()

            # Best fit
            best = next((a for a in apps if "Strong" in a.get("verdict","")), None)
            if best:
                st.success(f"💡 **Best fit this session:** {best['company']} — {best['role']}")
            elif len(apps) > 1:
                # Pick the one with fewest gaps
                fewest = min(apps, key=lambda a: len(a.get("report_data",{}).get("gaps",[])))
                st.info(f"💡 **Closest fit:** {fewest['company']} — {fewest['role']} (fewest gaps to address)")

            st.markdown("""
---
Keep refining, keep applying — every step forward counts.
Best of luck out there. You've genuinely got this! 🌟
""")
            st.session_state.summary_shown = True


if __name__ == "__main__":
    main()
