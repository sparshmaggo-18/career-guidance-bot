# Career Coach AI 🤖🎓  
An AI-powered chatbot that helps students and job seekers with **career guidance, internship suggestions, resume feedback, and personalized learning roadmaps**. Built with **Python, Streamlit, and OpenAI API**.  

---

## ✨ Features
- 💼 **Career & Internship Guidance** – Personalized suggestions based on user interests.  
- 📄 **Resume Review** – Upload your resume and get instant improvement tips.  
- 🛣️ **Learning Roadmaps** – Step-by-step guidance for skills & career paths.  
- 🎤 **Mock Interview Q&A** – Practice with AI-generated questions.  
- ⚡ **LLM Support** – Get real-time answers via OpenAI API (optional).  

---

## 🛠️ Tech Stack
- **Frontend & Backend**: [Streamlit](https://streamlit.io/)  
- **AI/ML**: OpenAI GPT models (or rule-based fallback)  
- **Python Libraries**: `streamlit`, `python-dotenv`, `openai`, `PyPDF2`, etc.  

---

## 🚀 Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/career-coach-ai.git
cd career-coach-ai
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

## 📂 Project Structure
```
career-coach-ai/
│── app.py                # Main Streamlit app
│── requirements.txt      # Dependencies
│── .env.example          # Example environment variables
│── README.md             # Documentation
│── /utils                # Helper functions
│── /data                 # Sample resumes or datasets
```

---

## 🖼️ Screenshots
*(Add screenshots of your app UI here)*  
- Career guidance chatbot  
- Resume review tool  
- Learning roadmap  

---

## 📌 Roadmap
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
