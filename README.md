# Career Guidance Bot 🤖🎓  
An AI-powered chatbot that helps students and job seekers with **career guidance, internship suggestions, resume feedback, and personalized learning roadmaps**. Built with **Python, Streamlit, and OpenAI API**.  

---

## 🚩 Problem Statement  
Choosing the right career path and finding suitable internships are among the most common challenges faced by students and fresh graduates.  
Some of the major issues include:  
- **Lack of proper guidance**: Many students do not have access to mentors or career counselors.  
- **Resume quality issues**: Resumes are often poorly structured and do not highlight key strengths.  
- **Information overload**: With so many resources available online, students struggle to identify relevant learning paths.  
- **Limited preparation for interviews**: Students often lack practice for technical and HR interview rounds.  

These challenges often result in **missed opportunities, rejections, and lack of clarity** in career growth.  

---

## ✅ How Our Prototype Solves the Problem  
Career Coach AI acts as a **virtual career mentor** that provides students with **personalized and actionable guidance**.  
- 💼 Suggests **internship and career options** based on skills and interests.  
- 📄 Offers **resume review** with improvement tips.  
- 🛣️ Generates **learning roadmaps** for different career paths (e.g., AI/ML, Web Development, Data Science).  
- 🎤 Provides **mock interview questions** to build confidence.  
- ⚡ Uses **LLM (OpenAI GPT models)** for real-time, context-aware answers (work in progress).  

This ensures that students can **make informed career decisions** without relying on random advice or unverified sources.  

---

## 📊 Current Progress Status  
- ✅ Streamlit-based interface is developed.  
- ✅ Resume review, learning roadmap, and mock interview modules are functional.  
- ✅ Offline rule-based guidance system is working.  
- ⏳ **LLM connectivity (OpenAI API integration) is pending** – will enable real-time AI-powered answers.  

---

## 🛠️ Technologies & Tools Used  
- **Frontend & Backend**: [Streamlit](https://streamlit.io/)  
- **Programming Language**: Python 3.10+  
- **AI/ML**: OpenAI GPT (for LLM-based responses, in progress)  
- **Libraries**:  
  - `streamlit` → Web UI  
  - `openai` → LLM connectivity  
  - `python-dotenv` → Environment management  
  - `PyPDF2` → Resume parsing  
  - `os`, `re`, etc. → Utility functions  
- **Version Control**: Git + GitHub  

---

## 🖼️ Screenshots  

### Career Guidance Chatbot  
![Chatbot Screenshot](images/screenshot1.png)  

### Resume Review  
![Resume Review Screenshot](images/screenshot2.png)  

### Learning Roadmap  
![Roadmap Screenshot](images/screenshot3.png)  
  
---

## 🚀 Setup & Installation  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/sparshmaggo-18/career-guidance-bot.git
cd career-guidance-bot
```

### 2️⃣ Create Virtual Environment  
```powershell
python -m venv .venv
.venv\Scripts\Activate    # On Windows
# OR
source .venv/bin/activate  # On Mac/Linux
```

### 3️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 4️⃣ Setup Environment Variables  
Create a `.env` file in the project root:  
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 5️⃣ Run the App  
```bash
streamlit run app.py
```

---

## 📌 Roadmap  
- [ ] Connect to OpenAI API for real-time LLM guidance  
- [ ] Add LinkedIn/Indeed integration for internship listings  
- [ ] Advanced resume scoring with NLP  
- [ ] Save user sessions & progress tracking  
- [ ] Deploy on Streamlit Cloud / AWS  

---

## 🤝 Contributing  
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.  

---

## 📜 License  
This project is licensed under the MIT License.  
