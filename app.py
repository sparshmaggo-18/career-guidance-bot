\
import os
import re
import io
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import streamlit as st
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader

# Embeddings / Vector search
from sentence_transformers import SentenceTransformer
import faiss

# Optional LLM (OpenAI). We'll gracefully handle absence of key or package issues.
LLM_DEFAULT_MODEL = "gpt-4o-mini"  # change if needed

# -----------------------------
# Utilities
# -----------------------------
def load_pdf_text(file: io.BytesIO) -> str:
    """Extract text from a PDF file object. Returns empty string on failure."""
    try:
        reader = PdfReader(file)
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                pass
        return "\n".join(texts)
    except Exception:
        return ""


def chunk_text(text: str, max_tokens: int = 400, overlap: int = 80) -> List[str]:
    """Simple word-based chunker suitable for small RAG demos."""
    if not text:
        return []
    words = text.split()
    chunks = []
    i = 0
    step = max_tokens - overlap if max_tokens > overlap else max_tokens
    while i < len(words):
        chunk = words[i : i + max_tokens]
        chunks.append(" ".join(chunk))
        i += step if step > 0 else max_tokens
    return chunks


@dataclass
class VectorStore:
    index: faiss.IndexFlatIP
    id_to_text: Dict[int, str]
    model_name: str


@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)


def build_faiss_index(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> Optional[VectorStore]:
    if not texts:
        return None
    embedder = get_embedder(model_name)
    embs = embedder.encode(texts, normalize_embeddings=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs.astype(np.float32))
    id_to_text = {i: t for i, t in enumerate(texts)}
    return VectorStore(index=index, id_to_text=id_to_text, model_name=model_name)


def retrieve(query: str, vs: VectorStore, k: int = 4) -> List[str]:
    if not vs or not query:
        return []
    embedder = get_embedder(vs.model_name)
    q = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    D, I = vs.index.search(q, k)
    results = []
    for idx in I[0]:
        if int(idx) in vs.id_to_text:
            results.append(vs.id_to_text[int(idx)])
    return results


# -----------------------------
# Minimal LLM wrapper (optional)
# -----------------------------
def call_openai(messages: List[Dict], model: str = LLM_DEFAULT_MODEL, temperature: float = 0.4) -> Tuple[Optional[str], Optional[str]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, "OPENAI_API_KEY not set. Switch OFF LLM in the sidebar or set your key."
    # Try new SDK (openai>=1.0)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        return resp.choices[0].message.content, None
    except Exception as e1:
        # Try legacy SDK fallback
        try:
            import openai
            openai.api_key = api_key
            resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature)
            return resp.choices[0].message["content"], None
        except Exception as e2:
            return None, f"LLM error: {e1}\n{e2}"


# -----------------------------
# Domain logic (rules/templates)
# -----------------------------
ROLE_TEMPLATES = {
    "Software Engineer": {
        "skills": [
            "DSA (arrays, strings, trees, graphs, DP)",
            "OOP, Design Patterns, System Design basics",
            "Git/GitHub, CI basics",
            "Web backend (Node/Express or Django/FastAPI)",
            "SQL + one NoSQL",
        ],
        "projects": [
            "Full‑stack app with auth + CRUD + tests",
            "URL shortener or notes app with rate limiting",
            "API with pagination + caching + logging",
        ],
    },
    "Data Scientist": {
        "skills": [
            "Python stack (NumPy, Pandas, Matplotlib)",
            "EDA, feature engineering, model selection",
            "Regression, classification, metrics",
            "SQL, dashboards (Streamlit/Gradio)",
            "Statistics (hypothesis tests, A/B)"
        ],
        "projects": [
            "End‑to‑end ML project with data cleaning + model",
            "Business case study with insights and dashboard",
            "Time‑series or NLP mini‑project",
        ],
    },
    "ML Engineer": {
        "skills": [
            "PyTorch or TensorFlow basics",
            "MLOps (data versioning, experiment tracking)",
            "Serving (FastAPI) + Docker",
            "Vector DB / embeddings basics",
            "GPU fundamentals",
        ],
        "projects": [
            "Train a custom classifier and deploy",
            "RAG chatbot with FAISS and embeddings",
            "Model monitoring: drift + alerts",
        ],
    },
    "Web Developer": {
        "skills": [
            "HTML/CSS/JS, React + state mgmt",
            "REST APIs, auth, secure storage",
            "Testing (Jest/PyTest)",
            "Accessibility and performance",
            "Deployment (Vercel/Render/Heroku)",
        ],
        "projects": [
            "Portfolio + blog (MDX/Notion CMS)",
            "E‑commerce mini with cart + payments (test mode)",
            "Real‑time chat with WebSocket",
        ],
    },
    "Android Developer": {
        "skills": [
            "Kotlin + Jetpack Compose",
            "Room DB, Retrofit, coroutines",
            "Clean architecture (MVVM)",
            "Play Store policies, app signing",
            "Basic CI/CD (Gradle)"
        ],
        "projects": [
            "Offline‑first notes app",
            "Fitness tracker with sensors",
            "News app with caching + dark mode",
        ],
    },
    "DevOps": {
        "skills": [
            "Linux, shell scripting",
            "Docker, basic Kubernetes",
            "CI/CD (GitHub Actions)",
            "Cloud basics (AWS/GCP/Azure)",
            "Observability (logs/metrics)"
        ],
        "projects": [
            "CI pipeline for test+build+deploy",
            "Infra as code (Terraform) demo",
            "Monitoring stack (Prometheus/Grafana)",
        ],
    },
}

INDIA_INTERNSHIP_TIPS = [
    "Use college TPO + seniors’ referrals; maintain a neat LinkedIn headline (Role | Skills | Projects | Open to Internship).",
    "Shortlist 30‑50 companies aligned to your role; track outreach in a sheet (date, contact, response).",
    "Cold email format: 3 lines—who you are, 1 impact bullet (with numbers), ask for a 15‑min chat or internship consideration.",
    "Participate in hackathons/open‑source; recruiters frequently scan winners’ lists and contributor graphs.",
    "Customize resume to JD: mirror 6‑8 keywords (skills/tools) in your bullets (naturally, not keyword stuffing).",
]

ACTION_VERBS = ["Built", "Led", "Optimized", "Designed", "Implemented", "Automated", "Deployed", "Improved", "Integrated", "Analyzed", "Reduced", "Increased"]


def quick_resume_diagnostics(text: str) -> List[str]:
    tips = []
    if not text or len(text.strip()) < 200:
        tips.append("Resume text seems short. Aim for 1 page with 4–6 impactful bullets.")
    # Contact info
    if not re.search(r"[\\w\\.-]+@[\\w\\.-]+", text):
        tips.append("Add a professional email (firstname.lastname@).")
    if not re.search(r"\\b\\+?\\d{10,}\\b", text):
        tips.append("Add a phone number (with country code).")
    # Numbers
    if not re.search(r"\\b(\\d+%?|[0-9]+\\.[0-9]+%?)\\b", text):
        tips.append("Quantify impact with numbers (e.g., 'reduced latency by 35%').")
    # Action verbs
    if not any(v.lower() in text.lower() for v in ACTION_VERBS):
        tips.append("Start bullets with strong action verbs (Built, Optimized, Led, …).")
    # Sections
    essentials = ["Education", "Projects", "Skills"]
    for sec in essentials:
        if sec.lower() not in text.lower():
            tips.append(f"Add a '{sec}' section.")
    # ATS hints
    if len(re.findall(r"\\b(Java|C\\+\\+|Python|SQL|React|Node|AWS|Docker|PyTorch|TensorFlow)\\b", text)) < 3:
        tips.append("Mirror core keywords from target JD (3–6 exact matches).")
    return tips


def offline_chat_answer(query: str, profile: Dict) -> str:
    role = profile.get("target_role") or "Software Engineer"
    rt = ROLE_TEMPLATES.get(role, ROLE_TEMPLATES["Software Engineer"])  # fallback
    lines = [
        f"Here’s a practical plan for {role} internships:",
        "\nSkill checklist (start with gaps):",
        *[f"- {s}" for s in rt["skills"][:5]],
        "\nProjects that get callbacks:",
        *[f"- {p}" for p in rt["projects"][:3]],
        "\nInternship tactics:",
        *[f"- {t}" for t in INDIA_INTERNSHIP_TIPS[:4]],
        "\nNext 7 days:",
        "- Shortlist 20 companies and send 10 tailored reach‑outs.",
        "- Finish 1 mini‑project with README + demo GIF.",
        "- Post a short LinkedIn thread summarizing what you built.",
    ]
    return "\n".join(lines)


def build_system_prompt(profile: Dict, retrieved: List[str]) -> str:
    profile_str = json.dumps(profile, indent=2)
    ctx = "\n\n".join(retrieved)
    return f"""
You are CareerMentor, a concise and practical career + internship advisor for tech students.
Rules:
- Be specific, bullet‑point heavy, India‑friendly, and action‑oriented.
- Prefer checklists, timelines, measurable outcomes, and short examples.
- If you use the context, never hallucinate—say 'Not in context' if unsure.

User profile:
{profile_str}

Retrieved context:
{ctx if ctx.strip() else 'No extra context.'}
"""


def llm_chat_response(query: str, profile: Dict, retrieved: List[str]) -> Tuple[Optional[str], Optional[str]]:
    system_prompt = build_system_prompt(profile, retrieved)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    return call_openai(messages)


# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="Career & Internship Mentor", page_icon="🎯", layout="wide")

# Sidebar
st.sidebar.title("🎯 Career & Internship Mentor")
use_llm = st.sidebar.toggle("Use LLM (OpenAI)", value=False, help="Turn OFF to run fully offline (rule‑based replies).")
model_name = st.sidebar.text_input("LLM model", value=LLM_DEFAULT_MODEL, help="Example: gpt-4o-mini. Change if your account uses a different model name.")

with st.sidebar.expander("Profile", expanded=True):
    name = st.text_input("Name", value="")
    degree = st.text_input("Degree / Year", value="BTech CSE, 4th sem")
    target_role = st.selectbox("Target role", list(ROLE_TEMPLATES.keys()), index=0)
    skills = st.text_area("Your skills (comma‑separated)", value="C, C++, Python, DSA, basic ML")
    interests = st.text_area("Interests", value="AIML, full‑stack, research")
    locations = st.text_input("Preferred locations", value="Remote, Bangalore, Gurgaon")
    profile = {
        "name": name,
        "degree": degree,
        "target_role": target_role,
        "skills": [s.strip() for s in skills.split(",") if s.strip()],
        "interests": interests,
        "locations": [s.strip() for s in locations.split(",") if s.strip()],
    }

with st.sidebar.expander("Knowledge Base (RAG)"):
    st.write("Upload PDFs or TXT files. We'll chunk, embed, and retrieve snippets for better answers.")
    kb_files = st.file_uploader("Add files", type=["pdf", "txt"], accept_multiple_files=True)
    if "kb_store" not in st.session_state:
        st.session_state.kb_store = None
    if st.button("Build / Rebuild Index"):
        all_chunks = []
        if kb_files:
            for f in kb_files:
                if f.name.lower().endswith(".pdf"):
                    text = load_pdf_text(f)
                else:
                    text = f.read().decode("utf-8", errors="ignore")
                chunks = chunk_text(text, max_tokens=380, overlap=60)
                all_chunks.extend(chunks)
        # Starter context even if user uploads nothing
        if not all_chunks:
            starter_docs = [
                "Internship strategy: referrals > cold apply. Keep a tracker, tailor resume to JD keywords, build 1‑2 standout projects with numbers.",
                "Resume tips: one page, impact bullets with metrics, skills section mirroring JD, links to GitHub/Portfolio.",
                "Interview prep: DSA practice (arrays, strings, trees, graphs, DP), 20 system design basics, behavioral STAR stories.",
            ]
            for s in starter_docs:
                all_chunks.extend(chunk_text(s, 120, 20))
        st.session_state.kb_store = build_faiss_index(all_chunks)
        st.success("Knowledge index built.")

# Tabs
chat_tab, resume_tab, roadmap_tab, interview_tab = st.tabs(["💬 Chat", "📄 Resume Review", "🗺️ Roadmap", "🎤 Mock Interview"])

# -----------------------------
# Chat Tab
# -----------------------------
with chat_tab:
    st.subheader("Ask anything about internships, roles, resumes, or projects")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role_msg, content in st.session_state.chat_history:
        with st.chat_message(role_msg):
            st.markdown(content)

    user_query = st.chat_input("Type your question… (e.g., 'How do I get an ML internship by Dec?')")

    if user_query:
        st.session_state.chat_history.append(("user", user_query))
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                retrieved = []
                if st.session_state.get("kb_store"):
                    retrieved = retrieve(user_query, st.session_state.kb_store, k=4)

                if use_llm:
                    answer, err = llm_chat_response(user_query, profile, retrieved)
                    if err:
                        st.warning(err)
                        answer = offline_chat_answer(user_query, profile)
                else:
                    answer = offline_chat_answer(user_query, profile)

                st.markdown(answer)
                st.session_state.chat_history.append(("assistant", answer))

# -----------------------------
# Resume Review Tab
# -----------------------------
with resume_tab:
    st.subheader("Upload resume for instant feedback")
    rfile = st.file_uploader("PDF or TXT", type=["pdf", "txt"], accept_multiple_files=False, key="resume")
    col1, col2 = st.columns(2)
    with col1:
        max_len = st.slider("Max suggestions", 3, 15, 8)
    with col2:
        use_llm_resume = st.checkbox("Use LLM for critique (if enabled in sidebar)", value=False)

    if rfile:
        if rfile.name.lower().endswith(".pdf"):
            text = load_pdf_text(rfile)
        else:
            text = rfile.read().decode("utf-8", errors="ignore")
        basic_tips = quick_resume_diagnostics(text)[:max_len]

        st.markdown("**Quick diagnostics:**")
        for t in basic_tips:
            st.write("- ", t)

        if use_llm and use_llm_resume:
            prompt = f"""
Act as an ATS‑aware resume reviewer for internships. Provide:
1) 5 bullet strengths
2) 5 bullet improvements (with concrete rewrites)
3) 5 keywords to mirror from a typical JD for the user's target role
Resume text below:
{text[:8000]}
"""
            answer, err = call_openai(
                [
                    {"role": "system", "content": "You are a precise, practical resume reviewer."},
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
            )
            if err:
                st.warning(err)
            else:
                st.markdown(answer)
        else:
            st.info("LLM critique is OFF. Showing rule‑based suggestions only.")

# -----------------------------
# Roadmap Tab
# -----------------------------
with roadmap_tab:
    st.subheader("12‑Week Roadmap (customizable)")
    role = st.selectbox("Select role", list(ROLE_TEMPLATES.keys()), index=0, key="rm_role")
    focus_hours = st.slider("Hours per week", 5, 30, 12)

    rt = ROLE_TEMPLATES[role]
    plan = []
    weeks = [
        (1, "Set up, baseline, choose project"),
        (2, "Core skills sprint #1"),
        (3, "Core skills sprint #2"),
        (4, "Mini‑project #1"),
        (5, "Deepen topic A"),
        (6, "Mini‑project #2"),
        (7, "System/ML basics + notes"),
        (8, "Polish projects (tests/docs)") ,
        (9, "Interview prep sprint"),
        (10, "Apply + referrals week"),
        (11, "Mock interviews + fixes"),
        (12, "Final push + showcase"),
    ]

    for wk, theme in weeks:
        focus = ", ".join(rt["skills"][:3]) if wk <= 3 else (", ".join(rt["projects"][:2]) if wk in (4,6,12) else "Interview + outreach")
        deliverable = "LinkedIn post + GitHub README" if wk in (4,6,12) else "Practice log / tracker"
        plan.append({
            "Week": wk,
            "Theme": theme,
            "Focus": focus,
            "Hours": focus_hours,
            "Deliverable": deliverable,
        })

    df = pd.DataFrame(plan)
    st.dataframe(df, use_container_width=True)
    st.caption("Tip: Export this table from the three‑dot menu, then paste into your Notion/Docs.")

# -----------------------------
# Mock Interview Tab
# -----------------------------
with interview_tab:
    st.subheader("Practice interview questions")
    role = st.selectbox("Target role", list(ROLE_TEMPLATES.keys()), index=0, key="int_role")
    q_bank = {
        "Software Engineer": [
            "Explain time/space of binary search.",
            "Design a URL shortener (high level).",
            "Race condition vs deadlock?",
            "When to use a NoSQL DB?",
        ],
        "Data Scientist": [
            "Bias‑variance tradeoff?",
            "How do you handle class imbalance?",
            "Interpret precision vs recall.",
            "A/B test design for a new feature.",
        ],
        "ML Engineer": [
            "Steps to productionize an ML model.",
            "What is data drift vs concept drift?",
            "Vector DB use‑cases in RAG.",
            "Latency vs throughput trade‑offs.",
        ],
        "Web Developer": [
            "CSR vs SSR vs SSG?",
            "What is CORS and how to handle it?",
            "Explain JWT flow.",
            "How to optimize Largest Contentful Paint?",
        ],
        "Android Developer": [
            "State hoisting in Compose?",
            "When to use Room vs DataStore?",
            "MVVM flow for a network screen.",
            "Coroutines vs RxJava?",
        ],
        "DevOps": [
            "Docker image layers and caching?",
            "Blue‑green vs rolling deployment.",
            "K8s readiness vs liveness probes.",
            "Observability pillars.",
        ],
    }

    curr_q = st.selectbox("Pick a question", q_bank[role])
    user_answer = st.text_area("Your answer", height=180)

    if st.button("Get feedback"):
        if use_llm:
            prompt = f"""
You are an interview coach. Rate the answer 1‑10, then give:
- 3 strengths
- 3 improvements
- A concise model answer
Question: {curr_q}
Answer: {user_answer}
Role: {role}
"""
            fb, err = call_openai(
                [
                    {"role": "system", "content": "Be concise, concrete, and technical."},
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
            )
            if err:
                st.warning(err)
            else:
                st.markdown(fb)
        else:
            # Offline heuristic feedback
            strengths = []
            improvements = []
            score = 5
            if len(user_answer.split()) > 120:
                improvements.append("Be more concise (under 90 words).")
                score -= 1
            if any(k in user_answer.lower() for k in ["example", "e.g.", "for instance"]):
                strengths.append("Includes an example.")
                score += 1
            if any(n in user_answer for n in ["%", "ms", "MB", "QPS", "p95", "accuracy"]):
                strengths.append("Uses metrics/units.")
                score += 1
            if "trade" in user_answer.lower():
                strengths.append("Mentions trade‑offs.")
                score += 1
            score = min(max(score, 1), 9)
            st.write(f"**Score:** {score}/10")
            st.write("**Strengths:**")
            for s in strengths or ["Clear structure."]:
                st.write("- ", s)
            st.write("**Improvements:**")
            for im in improvements or ["Add 1 metric and 1 trade‑off."]:
                st.write("- ", im)

# Footer tip
st.markdown("---")
st.caption("Pro tip: Keep LLM OFF for speed while iterating. Turn it ON for richer, personalized feedback.")
