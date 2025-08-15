# OptiCare – AI-Powered Keratoplasty Patient Monitoring Agent
# 📌 Overview

OptiCare is an AI-powered, Telegram-integrated assistant designed to monitor keratoplasty (corneal transplant) patients post-surgery. It ensures timely symptom reporting, intelligent follow-ups, and empathetic patient engagement. The agent is optimized for both patients and clinicians, providing real-time insights and tracking recovery progress while reducing the risk of complications.

# 📹 Video Recording of Demo and Code Review
https://drive.google.com/file/d/1EYXKsmqpupAGbDNqITQZ4gJ-nGQ1K3tv/view?usp=drive_link

This project blends AI, conversational empathy, and retrieval-augmented generation (RAG) to deliver a supportive and clinically valuable recovery assistant.

# ✨ Key Features

* Post-Surgery Symptom Monitoring

  * Patients can report symptoms in natural language.

  * AI interprets severity levels and flags urgent cases.

* RAG-Powered Medical Guidance

  * Retrieves answers from official keratoplasty FAQs and guidelines.

  * Cites relevant sources (question numbers or document sections).

* Empathetic Conversation Design

  * AI communicates with warmth and reassurance.

  * Avoids off-topic or irrelevant advice.

* Follow-Up Scheduling

  * Automated reminders for check-ins, medication, and follow-up appointments.

  * Tracks both completed and pending tasks.

* Clinician Dashboard Ready (Optional)

  * Supports structured data output for medical record integration.

# 🛠️ Technology Stack

* Language Model: OpenAI GPT (with prompt engineering for empathy + precision)

* Frameworks: Python, LangChain

* Knowledge Retrieval: FAISS / Vector DB for storing keratoplasty documents

* Messaging Interface: Telegram Bot API

* Data Sources:

  * Something.pdf
  * Somethings.csv

# 🚀 Getting Started
1️⃣ Prerequisites

* Python 3.10+

* Telegram bot token (from BotFather)

* OpenAI API key

* FAISS installed


2️⃣ Installation
git clone https://github.com/<your-username>/OptiCare.git
cd OptiCare


3️⃣ Environment Setup

* Create a .env file:

OPENAI_API_KEY=your_openai_api_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token


# 4️⃣ Run the Bot
python app.py

# 🧠 How It Works

User Sends a Message – Patient sends symptom or question via Telegram.

* AI Analysis – Model analyzes message and determines intent/severity.

* Knowledge Retrieval – RAG searches keratoplasty documents for precise guidance.

* Response Generation – AI responds empathetically, citing sources where applicable.

* Follow-Up Scheduling – If necessary, bot schedules reminders or prompts further action.

#💡 Use Cases

* For Patients: Daily symptom tracking, reassurance, and early detection of risks.

* For Clinicians: Structured, summarized reports to aid follow-up consultations.

* For Research: Aggregate anonymized patient feedback for medical studies.

# 🔒 Privacy & Security

* No patient data is stored without consent.

* Conversations are processed securely via encrypted connections.

* Designed for compliance with HIPAA-aligned best practices.

# 📈 Future Enhancements

* 8 Multi-language support for broader patient access.

* Integration with hospital EMR systems.

* Predictive analytics for complication risk.
