import streamlit as st
from dotenv import load_dotenv
import os
import PyPDF2
from groq import Groq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------------------
# Initialize Groq Client
# -------------------------------
client = Groq(api_key=GROQ_API_KEY)

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Career Guidance Bot", page_icon="🎓", layout="wide")

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["👤 Profile Info", "📄 Resume Analyzer", "💬 Career Chatbot"])

# -------------------------------
# Profile Info Page
# -------------------------------
if page == "👤 Profile Info":
    st.title("👤 Student Profile")

    with st.form("profile_form"):
        name = st.text_input("Full Name")
        degree = st.text_input("Degree / Specialization")
        year = st.selectbox("Current Year", ["1st Year", "2nd Year", "3rd Year", "4th Year", "Graduate", "Other"])
        email = st.text_input("Email (optional)")
        submitted = st.form_submit_button("Save Profile")

    if submitted:
        st.session_state["name"] = name
        st.session_state["degree"] = degree
        st.session_state["year"] = year
        st.session_state["email"] = email
        st.success(f"✅ Profile saved for {name} ({degree}, {year})")

    if "name" in st.session_state:
        st.info(f"Current Profile: {st.session_state['name']} | {st.session_state['degree']} | {st.session_state['year']}")

# -------------------------------
# Resume Analyzer Page
# -------------------------------
elif page == "📄 Resume Analyzer":
    st.title("📄 Resume Analyzer")
    st.write("Upload your resume to get AI-powered feedback.")

    uploaded_file = st.file_uploader("Upload your resume (PDF only)", type="pdf")

    if uploaded_file:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        resume_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text() or ""
            text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
            resume_text += text

        if st.button("🔍 Analyze Resume"):
            with st.spinner("Analyzing your resume..."):
                try:
                    response = client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[
                            {"role": "system", "content": "You are an AI resume reviewer and career mentor. Provide structured feedback, list strengths, weaknesses, and suggest internship/job roles."},
                            {"role": "user", "content": f"Student Info:\nName: {st.session_state.get('name','')}\nDegree: {st.session_state.get('degree','')}\nYear: {st.session_state.get('year','')}\n\nResume:\n{resume_text[:4000]}"}
                        ],
                        max_tokens=700,
                    )
                    feedback = response.choices[0].message.content
                except Exception as e:
                    feedback = f"⚠️ Error: {e}"

                st.subheader("📊 AI Feedback on Resume & Career")
                st.write(feedback)

# -------------------------------
# Career Chatbot Page
# -------------------------------
elif page == "💬 Career Chatbot":
    st.title("💬 Career Guidance Bot")
    st.write("Ask me anything about careers, internships, or skills 🚀")

    

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display previous chats
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if prompt := st.chat_input("Type your question here..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            response = client.chat.completions.create(
                model="llama3-70b-8192",  # Free model from Groq
                messages=[
                    {"role": "system", "content": "You are a helpful AI career and internship mentor."},
                    *st.session_state["messages"]
                ],
                max_tokens=512,
            )
            reply = response.choices[0].message.content
        except Exception as e:
            reply = f"⚠️ Error: {e}"

        st.session_state["messages"].append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

