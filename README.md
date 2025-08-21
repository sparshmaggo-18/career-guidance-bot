# Career Guidance Bot 🤖🎓

## 📌 Problem Statement
Students and young professionals often struggle to find the right career path, internships, and opportunities aligned with their skills and academic background. 
Traditional career counseling is limited, not always accessible, and often lacks personalized feedback. 
Additionally, resume evaluation is typically done manually, which can be time-consuming and subjective. 

Our solution addresses these issues by building an **AI-powered Career & Internship Guidance Bot** that helps students with:  
- Personalized career guidance  
- Internship suggestions  
- Resume analysis with AI  
- A structured interface for profile building  

This ensures that students receive **real-time, AI-driven advice** tailored to their academic journey and career goals.  

---

## 🚀 Current Progress Status
- ✅ Profile Management (collecting name, degree, year, and other basic info)  
- ✅ Career Chatbot integrated with **Groq API** for real-time AI answers  
- ✅ Resume Analysis section that provides insights on uploaded resumes  
- ✅ Sidebar navigation for seamless access to different sections  
- ✅ Fully functional Streamlit app with a user-friendly UI  

---

## 💡 How the Prototype Solves the Problem
1. **Career Chatbot** – Provides personalized career and internship guidance using AI-powered responses.  
2. **Resume Analyzer** – Automatically analyzes resumes and provides suggestions for improvement.  
3. **Profile Section** – Collects student details to give **context-aware recommendations**.  
4. **Streamlined Interface** – Easy-to-use sidebar navigation to switch between Profile, Career Chatbot, and Resume Analyzer.  

This system reduces dependency on manual guidance and ensures that **students can get quick, accurate, and personalized insights** anytime.  

---

## 🛠️ Technologies & Tools Used
- **Python** – Core programming language  
- **Streamlit** – Frontend & interactive UI  
- **Groq API** – LLM integration for real-time AI answers  
- **PyPDF2** – To parse and extract text from resumes  
- **dotenv** – To manage API keys securely  
- **Other Python Libraries** – os, io, etc.  

---

## 📸 Screenshots
*(Add your screenshots here in GitHub using `![Alt Text](image.png)` format)*  

Example:  
```markdown
![Career Bot Screenshot](screenshots/career_bot.png)
```  

---

## ⚙️ How to Run the Project
1. Clone this repository  
   ```bash
   git clone https://github.com/sparshmaggo-18/career-guidance-bot.git
   cd career-internship-bot
   ```  

2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```  

3. Set your **Groq API Key**  
   - On Linux/Mac:  
     ```bash
     export GROQ_API_KEY="your_api_key_here"
     ```  
   - On Windows (PowerShell):  
     ```powershell
     setx GROQ_API_KEY "your_api_key_here"
     ```  

4. Run the Streamlit app  
   ```bash
   streamlit run app_updated.py
   ```  

---

## 🔮 Future Scope
- Adding **job/internship recommendation system** by scraping real-time data  
- Multi-language support for global accessibility  
- Integration with LinkedIn/Job Portals  
- Advanced AI models for **career path prediction**  

---

## ✅ Conclusion
The **Career & Internship Guidance AI Bot** provides a **personalized, AI-driven platform** for students to explore career options, internships, and resume improvements.  
By leveraging **Groq API with Streamlit**, the solution ensures accessibility, scalability, and reliability.  

---
