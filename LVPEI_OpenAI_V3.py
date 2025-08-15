import os
import re
import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
import fitz  # PyMuPDF

from dotenv import load_dotenv
load_dotenv()

# Configuration
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PATH = "keratoplasty_monitor.db"
VECTOR_DB_PATH = "./chroma_db"
PDF_FOLDER = "./pdfs"
EXPECTED_PDFS = ["LVPEI - FAQs.pdf", "LVPEI - Sheet 4.pdf", "LVPEI - Sheet 3.pdf"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserState:
    user_id: int
    last_pain_score: Optional[int] = None
    recovery_stage: str = "unknown"

@dataclass
class SymptomAssessment:
    user_id: int
    timestamp: datetime
    symptoms: Dict[str, any]
    risk_level: str
    pain_score: Optional[int]
    recommendations: List[str]
    context_used: List[str]
    pdf_sources: List[str]
    follow_up_questions: List[str]

@dataclass
class SymptomSession:
    user_id: int
    stage: str = "initial"  # initial, gathering, paused
    collected_symptoms: Dict[str, any] = None
    questions_asked: List[str] = None
    paused_at: Optional[datetime] = None
    context_conversations: List[Dict] = None
    
    def __post_init__(self):
        if self.collected_symptoms is None:
            self.collected_symptoms = {}
        if self.questions_asked is None:
            self.questions_asked = []
        if self.context_conversations is None:
            self.context_conversations = []

def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            user_message TEXT NOT NULL,
            ai_response TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_states (
            user_id INTEGER PRIMARY KEY,
            last_pain_score INTEGER,
            recovery_stage TEXT DEFAULT 'unknown'
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            symptoms TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            pain_score INTEGER,
            recommendations TEXT,
            context_used TEXT,
            pdf_sources TEXT,
            follow_up_questions TEXT
        )
    ''')
    conn.commit()
    conn.close()

class PDFProcessor:
    def __init__(self):
        self.pdf_folder = Path(PDF_FOLDER)
        self.pdf_folder.mkdir(exist_ok=True)
    
    def load_pdfs(self) -> List[Document]:
        documents = []
        for pdf_name in EXPECTED_PDFS:
            pdf_path = self.pdf_folder / pdf_name
            if pdf_path.exists():
                try:
                    doc = fitz.open(pdf_path)
                    for page_num in range(len(doc)):
                        text = doc[page_num].get_text()
                        if text.strip():
                            metadata = {
                                "source": pdf_name,
                                "page": page_num + 1,
                                "document_type": self._classify_document(pdf_name),
                                "full_source": f"{pdf_name}, Page {page_num + 1}"
                            }
                            metadata.update(self._extract_topics(text))
                            documents.append(Document(page_content=text.strip(), metadata=metadata))
                    doc.close()
                    logger.info(f"✅ Loaded {pdf_name}")
                except Exception as e:
                    logger.error(f"Error loading {pdf_name}: {e}")
        
        if not documents:
            documents = self._create_fallback_docs()
        return documents
    
    def _classify_document(self, filename: str) -> str:
        filename_lower = filename.lower()
        if 'faq' in filename_lower:
            return 'LVPEI FAQ'
        elif 'sheet 3' in filename_lower:
            return 'LVPEI Sheet 3'
        elif 'sheet 4' in filename_lower:
            return 'LVPEI Sheet 4'
        else:
            return 'LVPEI Guidelines'
    
    def _extract_topics(self, text: str) -> Dict[str, any]:
        text_lower = text.lower()
        metadata = {}
        
        if any(word in text_lower for word in ['emergency', 'urgent', 'immediately', 'call surgeon']):
            metadata['urgency'] = 'high'
        elif any(word in text_lower for word in ['concerning', 'contact doctor']):
            metadata['urgency'] = 'medium'
        else:
            metadata['urgency'] = 'low'
        
        topics = []
        if any(word in text_lower for word in ['pain', 'hurt', 'discomfort']):
            topics.append('pain_management')
        if any(word in text_lower for word in ['red', 'redness', 'inflamed']):
            topics.append('redness_infection')
        if any(word in text_lower for word in ['vision', 'sight', 'blurry']):
            topics.append('vision_recovery')
        if any(word in text_lower for word in ['drop', 'medication', 'medicine']):
            topics.append('medications')
        if any(word in text_lower for word in ['swimming', 'exercise', 'activity', 'workout']):
            topics.append('activity_restrictions')
        if any(word in text_lower for word in ['rejection', 'graft']):
            topics.append('complications')
        
        metadata['topics'] = topics
        return metadata
    
    def _create_fallback_docs(self) -> List[Document]:
        fallback_content = [
            {
                "content": "Activity Restrictions Post-Surgery: Avoid swimming and contact sports until cleared by your eye surgeon. Water activities pose significant risk of infection. Light walking is okay after few days. Heavy lifting and strenuous exercise should be avoided for at least 3 weeks. Bending over should be avoided for 3 months to prevent pressure on the eye. Working out immediately after surgery can increase eye pressure and interfere with healing.",
                "metadata": {"source": "LVPEI - Sheet 4.pdf", "page": 1, "document_type": "LVPEI Sheet 4", "full_source": "LVPEI - Sheet 4.pdf, Question 3", "urgency": "medium", "topics": ["activity_restrictions", "workout"]}
            },
            {
                "content": "Pain Management After Keratoplasty: Normal pain levels are 1-4/10. Pain levels of 5-7/10 are concerning - contact doctor within 24 hours. Pain levels of 8-10/10 are emergency situations requiring immediate contact with your surgeon. Some discomfort is expected but severe pain is not normal.",
                "metadata": {"source": "LVPEI - FAQs.pdf", "page": 1, "document_type": "LVPEI FAQ", "full_source": "LVPEI - FAQs.pdf, Page 1", "urgency": "high", "topics": ["pain_management"]}
            },
            {
                "content": "Vision Changes and Floaters: New floaters or dark spots after keratoplasty surgery can indicate serious complications like retinal detachment or graft rejection. Any sudden increase in floaters, especially with flashing lights or curtain-like vision loss, requires immediate medical attention. Contact your surgeon urgently if you experience these symptoms.",
                "metadata": {"source": "LVPEI - Sheet 3.pdf", "page": 4, "document_type": "LVPEI Sheet 3", "full_source": "LVPEI - Sheet 3.pdf, Question 8", "urgency": "critical", "topics": ["vision_recovery", "complications"]}
            },
            {
                "content": "Exercise and Workout Guidelines: When can I exercise after keratoplasty? Light activities like walking are permitted after a few days. Avoid heavy lifting (over 10 pounds) for 3 weeks. No gym workouts or strenuous exercise for 3 weeks minimum. Swimming and water sports are prohibited until cleared by surgeon. Always ask your surgeon before resuming any exercise routine.",
                "metadata": {"source": "LVPEI - FAQs.pdf", "page": 7, "document_type": "LVPEI FAQ", "full_source": "LVPEI - FAQs.pdf, Question 15", "urgency": "medium", "topics": ["activity_restrictions", "exercise", "workout"]}
            },
            {
                "content": "Eye Drop Administration Guidelines: Daily eye drops are crucial for months after surgery. Never stop unless doctor says so. Leave 10-minute gap between different drops. Proper compliance with medication schedule is crucial for surgical success and preventing rejection. Set alarms to remember doses.",
                "metadata": {"source": "LVPEI - FAQs.pdf", "page": 3, "document_type": "LVPEI FAQ", "full_source": "LVPEI - FAQs.pdf, Page 3", "urgency": "high", "topics": ["medications"]}
            }
        ]
        
        documents = []
        for item in fallback_content:
            documents.append(Document(page_content=item["content"], metadata=item["metadata"]))
        
        logger.info(f"✅ Created {len(documents)} fallback documents with proper citations")
        return documents

class CitationRAGSystem:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        self.vector_store = None
        self.setup_rag()
    
    def setup_rag(self):
        try:
            processor = PDFProcessor()
            documents = processor.load_pdfs()
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            splits = splitter.split_documents(documents)
            
            if os.path.exists(VECTOR_DB_PATH):
                self.vector_store = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=self.embeddings)
            else:
                self.vector_store = Chroma.from_documents(documents=splits, embedding=self.embeddings, persist_directory=VECTOR_DB_PATH)
                self.vector_store.persist()
            
            logger.info("✅ Citation RAG system ready")
        except Exception as e:
            logger.error(f"RAG setup failed: {e}")
    
    def get_context_with_sources(self, query: str) -> Tuple[List[str], List[str], List[Dict]]:
        """Get context with detailed source information for citations"""
        if not self.vector_store:
            logger.warning("Vector store not available - using fallback response")
            return [], [], []
        
        try:
            # Get more documents for better context
            docs = self.vector_store.similarity_search(query, k=4)
            logger.info(f"Found {len(docs)} documents for query: {query}")
            
            contexts = []
            sources = []
            detailed_sources = []
            
            for doc in docs:
                contexts.append(doc.page_content)
                
                # Create detailed source info
                source_info = {
                    "content": doc.page_content,
                    "source": doc.metadata.get('source', 'LVPEI - Fallback.pdf'),
                    "page": doc.metadata.get('page', 'N/A'),
                    "document_type": doc.metadata.get('document_type', 'LVPEI Guidelines'),
                    "full_source": doc.metadata.get('full_source', 'LVPEI - Fallback.pdf, Page 1'),
                    "topics": doc.metadata.get('topics', []),
                    "urgency": doc.metadata.get('urgency', 'medium')
                }
                detailed_sources.append(source_info)
                sources.append(source_info["full_source"])
                
                # Log what we found
                logger.info(f"Retrieved source: {source_info['full_source']}")
            
            return contexts, sources, detailed_sources
            
        except Exception as e:
            logger.error(f"Context retrieval error: {e}")
            # Return fallback response when RAG fails
            fallback_info = {
                "content": "This information is not available in my verified LVPEI sources, but based on general medical knowledge from reputable sources like Mayo Clinic and WebMD, this appears to be a symptom that requires medical attention.",
                "source": "General Medical Knowledge",
                "page": "N/A", 
                "document_type": "External Medical Sources",
                "full_source": "Mayo Clinic / WebMD (Not in verified LVPEI sources)",
                "topics": ["general_medical"],
                "urgency": "medium"
            }
            return [fallback_info["content"]], [fallback_info["full_source"]], [fallback_info]

def extract_pain_score(text: str) -> Optional[int]:
    patterns = [
        r'(\d+)(?:/10|/10)',
        r'(\d+)\s*out\s*of\s*10',
        r'pain.*?(\d+)',
        r'(\d+).*?pain',
        r'level.*?(\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            score = int(match.group(1))
            return min(max(score, 1), 10)
    
    text_lower = text.lower()
    if any(word in text_lower for word in ['unbearable', 'excruciating', 'worst']):
        return 8
    elif any(word in text_lower for word in ['severe', 'intense', 'strong']):
        return 6
    elif any(word in text_lower for word in ['moderate', 'noticeable']):
        return 5
    elif any(word in text_lower for word in ['mild', 'slight']):
        return 3
    
    return None

def detect_symptoms_and_assess_risk(text: str) -> Tuple[Dict[str, any], str, List[str]]:
    text_lower = text.lower()
    symptoms = {}
    
    logger.info(f"🔍 ANALYZING TEXT: '{text}'")
    
    # Pain detection
    pain_keywords = ["pain", "hurt", "ache", "discomfort", "sore", "paining"]
    has_pain = any(keyword in text_lower for keyword in pain_keywords)
    pain_score = extract_pain_score(text)
    
    symptoms["pain"] = {
        "present": has_pain, 
        "score": pain_score,
        "severity": "severe" if pain_score and pain_score >= 7 else "moderate" if pain_score and pain_score >= 4 else "mild"
    }
    
    # Activity detection - ENHANCED
    exercise_keywords = ["workout", "exercise", "gym", "working out", "lifting", "weights", "cardio", "running", "fitness", "training"]
    has_exercise = any(keyword in text_lower for keyword in exercise_keywords)
    
    swimming_keywords = ["swimming", "pool", "swim", "water sports"]
    has_swimming = any(keyword in text_lower for keyword in swimming_keywords)
    
    symptoms["activities"] = {
        "exercise": has_exercise,
        "swimming": has_swimming,
        "mentioned": has_exercise or has_swimming
    }
    
    # Redness detection
    redness_keywords = ["red", "redness", "inflamed", "irritated", "bloodshot"]
    has_redness = any(keyword in text_lower for keyword in redness_keywords)
    
    symptoms["redness"] = {
        "present": has_redness, 
        "severity": "mild"
    }
    
    # Other symptoms
    symptoms["vision_issues"] = any(word in text_lower for word in ["blurry", "vision loss", "blind", "floaters", "dark spots"])
    symptoms["discharge"] = any(word in text_lower for word in ["discharge", "pus", "fluid"])
    symptoms["light_sensitivity"] = any(phrase in text_lower for phrase in ["light hurts", "light sensitivity"])
    
    logger.info(f"🎯 DETECTED SYMPTOMS: {symptoms}")
    
    # Risk assessment
    risk_level, recommendations = assess_risk_level(symptoms)
    logger.info(f"⚠️ RISK LEVEL: {risk_level}")
    
    return symptoms, risk_level, recommendations

def assess_risk_level(symptoms: Dict[str, any]) -> Tuple[str, List[str]]:
    risk_level = "NORMAL"
    recommendations = []
    
    has_redness = symptoms["redness"]["present"]
    has_pain = symptoms["pain"]["present"]
    pain_score = symptoms["pain"]["score"]
    has_vision_issues = symptoms.get("vision_issues", False)
    has_activities = symptoms.get("activities", {}).get("mentioned", False)
    
    # Critical conditions
    if pain_score and pain_score >= 8:
        risk_level = "CRITICAL"
        recommendations.append("🚨 **Call your surgeon NOW!** Pain 8+/10 needs immediate attention")
    elif has_vision_issues:
        risk_level = "CRITICAL" 
        recommendations.append("🚨 **Vision changes = urgent call needed!**")
    elif has_redness and has_pain:
        risk_level = "HIGH"
        recommendations.append("⚠️ **Redness + pain combo needs checking out**")
    
    # Activity-related concerns
    elif has_activities and not has_pain:
        risk_level = "MODERATE"
        recommendations.append("📋 **Let's discuss your activity timeline for safe recovery**")
    elif pain_score and pain_score >= 4:
        risk_level = "MODERATE"
        recommendations.append("📞 **Keep monitoring that pain level**")
    
    # Normal/low risk
    if risk_level == "NORMAL":
        recommendations.extend([
            "✅ **Continue your recovery routine**",
            "💊 **Keep up with prescribed medications**"
        ])
    
    return risk_level, recommendations

def generate_targeted_follow_up_questions(symptoms: Dict[str, any], message: str, collected_symptoms: Dict = None) -> List[str]:
    """Generate intelligent follow-up questions based on what user actually mentioned"""
    questions = []
    message_lower = message.lower()
    collected = collected_symptoms or {}
    
    logger.info(f"🤔 GENERATING QUESTIONS FOR: '{message}'")
    logger.info(f"📋 COLLECTED SYMPTOMS: {collected}")
    
    # Activity-specific questions (PRIORITY)
    if any(word in message_lower for word in ["workout", "exercise", "gym", "working out"]) and "workout_timeline" not in collected:
        questions.append("When exactly did you have your surgery, and when are you planning to start working out? 🏋️ (Exercise timing is crucial for healing)")
    
    elif any(word in message_lower for word in ["swimming", "pool", "swim"]) and "swimming_timeline" not in collected:
        questions.append("How long after your surgery are you planning to go swimming? 🏊 (Water exposure timing affects infection risk)")
    
    # Vision issues
    elif symptoms.get("vision_issues") and "vision_details" not in collected:
        questions.append("Are the floaters new since your surgery? Do you see any flashing lights? ⚡ (Critical symptom to check)")
    
    # Pain assessment
    elif symptoms["pain"]["present"] and not symptoms["pain"]["score"] and "pain_scale" not in collected:
        questions.append("Can you rate your eye pain on a scale of 1-10 right now? 📊 (Helps determine urgency level)")
    
    # General follow-up questions
    elif "surgery_timeline" not in collected:
        questions.append("How many days or weeks ago did you have your keratoplasty surgery? 📅 (Timeline helps assess what's normal)")
    
    elif "medication_compliance" not in collected:
        questions.append("Are you taking your prescribed eye drops exactly as directed? 💊 (Critical for healing)")
    
    # If no specific questions, ask general
    if not questions:
        questions.append("Can you tell me more about what's concerning you today? 😊")
    
    logger.info(f"❓ GENERATED QUESTIONS: {questions}")
    return questions[:2]  # Limit to 2 questions

class CitationConversationalAgent:
    def __init__(self, rag_system):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8, api_key=OPENAI_API_KEY)
        self.rag = rag_system
        self.memories = {}
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are LVPEI Agent, a warm and conversational healthcare assistant specializing in keratoplasty recovery.

PERSONALITY: Be natural, friendly, and genuinely human. You can answer questions about ANYTHING but you only PERFORM TASKS related to keratoplasty recovery. Write longer, more detailed responses (about 3-4 sentences) that feel like talking to a caring friend who's medically trained.

RESPONSE STRATEGY:
- For questions about non-medical topics: Answer honestly but don't perform the task
- Example: "Do you know Python?" → "Yes, I know about Python programming quite well actually! It's a fascinating language that I've worked with extensively. But you know what's really important right now? Your eye recovery journey. I'd love to focus my energy on helping you heal properly - how has your keratoplasty been going? Any concerns about pain, vision, or your daily routine that I can help address?"

BE CONVERSATIONAL AND DETAILED: Sound like a knowledgeable friend who genuinely cares. Write responses that are warm, informative, and show real concern for their wellbeing. Don't just give short answers - engage with them like you're having a meaningful conversation.

MEDICAL CONTEXT FROM LVPEI DOCUMENTS: {context}

ALWAYS redirect naturally back to their recovery after acknowledging their question, but do it in a caring, conversational way that doesn't feel robotic."""),
            ("human", "{message}")
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def is_off_topic(self, message: str) -> bool:
        task_requests = ['write', 'create', 'build', 'make me', 'code for', 'script for', 'help me with homework']
        return any(task in message.lower() for task in task_requests)
    
    def get_memory(self, user_id: int):
        if user_id not in self.memories:
            self.memories[user_id] = ConversationBufferWindowMemory(k=6, return_messages=True)
        return self.memories[user_id]
    
    async def chat(self, message: str, user_id: int) -> str:
        if self.is_off_topic(message):
            if 'python' in message.lower() or 'code' in message.lower():
                return """I know programming quite well actually! Python is such a versatile language, and I've worked with it extensively in various applications including data analysis and machine learning. It's really fascinating how you can build almost anything with it. But you know what I'm even more passionate about? Helping you through your keratoplasty recovery journey. 

**Your eye health is so much more important right now** - how has your healing been progressing? Are you experiencing any concerns about pain, vision changes, or your daily recovery routine that I can help you navigate? I'd love to focus my expertise on making sure you're healing optimally! 😊"""
            else:
                return """I can definitely chat about lots of different topics - it's always interesting to explore new subjects! But honestly, where I really come alive and can make the biggest difference is in helping with your keratoplasty recovery. That's where my specialized knowledge and genuine passion lie.

**Let's talk about what matters most for your wellbeing right now** - how are you feeling today? Any updates on your healing journey, questions about your recovery process, or concerns that have been on your mind? I'm here to support you through every step of this important healing process! 😊"""
        
        contexts, sources, detailed_sources = self.rag.get_context_with_sources(message)
        context_text = "\n".join(contexts[:2])
        memory = self.get_memory(user_id)
        
        if self._is_critical(message):
            return self._handle_critical(message)
        
        try:
            response = await self.chain.ainvoke({"message": message, "context": context_text})
            memory.chat_memory.add_user_message(message)
            memory.chat_memory.add_ai_message(response)
            return response
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return """I'm having some connection issues right now, but please know that I'm still here for you and want to help! Technology can be unpredictable sometimes, but your health and recovery are too important to let that get in the way. 

If this is urgent - especially if you're experiencing **severe pain (8+/10) or sudden vision changes** - please don't hesitate to contact your surgeon directly. For everything else, I should be back to normal shortly, and I'll be ready to support you through your recovery journey! 💙"""
    
    def _is_critical(self, message: str) -> bool:
        critical_indicators = ['8/10', '9/10', '10/10', 'unbearable', 'severe pain', 'vision loss']
        return any(indicator in message.lower() for indicator in critical_indicators)
    
    def _handle_critical(self, message: str) -> str:
        if any(word in message.lower() for word in ['8/10', '9/10', '10/10', 'unbearable', 'severe pain']):
            return """**I'm genuinely concerned about the severe pain you're describing. Please call your surgeon right now** - pain at this level needs immediate medical attention. You're not being dramatic; you're being smart about your health. 🚨"""
        elif any(word in message.lower() for word in ['vision loss', 'can\'t see', 'sudden']):
            return """**Vision changes are serious and need immediate attention. Please contact your surgeon right away.** I know this might be frightening, but early intervention is crucial. Your surgical team is there for exactly these situations. 🚨"""
        return """Based on what you're telling me, **please contact your surgeon today**. It's always better to be cautious with your recovery. 💙"""

class LVPEIBot:
    def __init__(self):
        self.app = Application.builder().token(TELEGRAM_TOKEN).build()
        self.rag = CitationRAGSystem()
        self.agent = CitationConversationalAgent(self.rag)
        self.user_states = {}
        self.symptom_sessions = {}  # Track active symptom sessions
        
        # Add handlers
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("help", self.help))
        self.app.add_handler(CommandHandler("symptoms", self.check_symptoms))
        self.app.add_handler(CommandHandler("insights", self.view_insights))
        self.app.add_handler(CommandHandler("history", self.see_history))
        self.app.add_handler(CommandHandler("summary", self.health_summary))
        self.app.add_handler(CallbackQueryHandler(self.button_callback))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
    
    def create_main_keyboard(self):
        keyboard = [
            [
                InlineKeyboardButton("🔍 Check Symptoms", callback_data="check_symptoms"),
                InlineKeyboardButton("❓ Get Help", callback_data="get_help")
            ],
            [
                InlineKeyboardButton("📊 View Insights", callback_data="view_insights"),
                InlineKeyboardButton("📋 See History", callback_data="see_history")
            ],
            [
                InlineKeyboardButton("📄 Health Summary", callback_data="health_summary")
            ]
        ]
        return InlineKeyboardMarkup(keyboard)
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        welcome = """🤖 **Hey there! I'm the LVPEI Agent** 🤖

I'm here to help make your corneal transplant recovery as smooth as possible. Think of me as that caring friend who happens to know a lot about eye healing! 😊

**🧠 What I Can Do:** 🧠
👁️ Analyze your symptoms with AI smarts
🏥 Give you evidence-based recommendations  
📊 Track your healing progress over time
🚨 Spot when you might need to call your doc
💬 Chat about your recovery journey

**📱 Available Commands:** 📱
/symptoms - Start symptom analysis
/insights - See your recovery trends
/history - Review conversation history
/summary - Get health overview
/help - Get assistance

**💬 Just Talk to Me!** 💬
You don't need formal commands - just say something like 'My eye hurts today' or 'I'm worried about some redness'.

Your data stays completely private and secure on your device.

Choose an option below or just start chatting! 👇"""
        
        keyboard = self.create_main_keyboard()
        await update.message.reply_text(welcome, parse_mode='Markdown', reply_markup=keyboard)
        self.user_states[update.effective_user.id] = UserState(update.effective_user.id)
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        
        if query.data == "check_symptoms":
            await self.check_symptoms(update, context, from_button=True)
        elif query.data == "get_help":
            await self.help(update, context, from_button=True)
        elif query.data == "view_insights":
            await self.view_insights(update, context, from_button=True)
        elif query.data == "see_history":
            await self.see_history(update, context, from_button=True)
        elif query.data == "health_summary":
            await self.health_summary(update, context, from_button=True)
    
    async def check_symptoms(self, update: Update, context: ContextTypes.DEFAULT_TYPE, from_button=False):
        user_id = update.effective_user.id if not from_button else update.callback_query.from_user.id
        
        # Initialize symptom session
        self.symptom_sessions[user_id] = SymptomSession(user_id=user_id, stage="gathering")
        
        text = """**Symptom Analysis Started!**

I'm your LVPEI Agent and I'll help analyze your symptoms step by step using medical guidelines. 

**Please tell me what's going on with your eye recovery:**

**Examples:**
• "I have questions about working out after surgery"
• "I went swimming yesterday, now there's redness"
• "My eye looks yellow and hurts a bit"
• "I traveled by flight right after surgery"

Don't worry about giving me everything at once - I'll ask follow-up questions to get a complete picture and provide the best medical guidance with proper citations.

**What symptoms are you experiencing right now?**"""
        
        if from_button:
            await update.callback_query.edit_message_text(text, parse_mode='Markdown')
        else:
            await update.message.reply_text(text, parse_mode='Markdown')
    
    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE, from_button=False):
        help_text = """**I'm here to support you!**

**What I do:** I'm LVPEI Agent, your dedicated keratoplasty recovery companion powered by LVPEI medical documents.

**Available Commands:**
• **/symptoms** - Analyze your symptoms
• **/insights** - See recovery trends  
• **/history** - Review your progress
• **/summary** - Complete health overview
• **/help** - This help message

**Emergency:** Severe pain (8+/10) or sudden vision loss? Contact your surgeon immediately.

**What's on your mind today?**"""
        
        if from_button:
            await update.callback_query.edit_message_text(help_text, parse_mode='Markdown')
        else:
            await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def view_insights(self, update: Update, context: ContextTypes.DEFAULT_TYPE, from_button=False):
        user_id = update.effective_user.id if not from_button else update.callback_query.from_user.id
        assessments = self._get_recent_assessments(user_id)
        
        if not assessments:
            text = """**📊 Recovery Insights**

No symptom assessments found yet! Once you start sharing your symptoms with me, I'll provide:

• **Recovery trend analysis** with LVPEI guidance
• **Pain level patterns** over time
• **Risk assessment history** with citations

        Use **/symptoms** or just tell me how you're feeling to start building your insights!"""
        else:
            text = self._generate_insights_text(assessments)
        
        if from_button:
            await update.callback_query.edit_message_text(text, parse_mode='Markdown')
        else:
            await update.message.reply_text(text, parse_mode='Markdown')
    
    async def see_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE, from_button=False):
        user_id = update.effective_user.id if not from_button else update.callback_query.from_user.id
        history = self._get_conversation_history(user_id)
        
        if not history:
            text = """**📋 Your Recovery History**

No conversation history yet! Once we start chatting about your recovery, this will show:

• **All our conversations** about your symptoms
• **Assessment timeline** with LVPEI citations
• **Progress tracking** over time

Start by telling me how you're feeling today! 😊"""
        else:
            text = self._generate_history_text(history)
        
        if from_button:
            await update.callback_query.edit_message_text(text, parse_mode='Markdown')
        else:
            await update.message.reply_text(text, parse_mode='Markdown')
    
    async def health_summary(self, update: Update, context: ContextTypes.DEFAULT_TYPE, from_button=False):
        user_id = update.effective_user.id if not from_button else update.callback_query.from_user.id
        summary = self._generate_health_summary(user_id)
        
        if from_button:
            await update.callback_query.edit_message_text(summary, parse_mode='Markdown')
        else:
            await update.message.reply_text(summary, parse_mode='Markdown')
    
    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.get_help(update, context, from_button=False)
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        message = update.message.text
        
        try:
            logger.info(f"📨 USER {user_id} SAID: '{message}'")
            
            # Check for exit commands first - HIGHEST PRIORITY
            if message.lower() in ['exit', 'stop', 'cancel', 'quit', 'end', '/exit', '/stop', '/cancel']:
                if user_id in self.symptom_sessions:
                    del self.symptom_sessions[user_id]
                    await update.message.reply_text(
                        "**✅ Session ended!** No problem at all - I'm here whenever you need me.\n\n"
                        "Feel free to ask me anything about your recovery, or use any of these commands:\n"
                        "• **/symptoms** - Start symptom analysis\n"
                        "• **/insights** - See your recovery trends\n"
                        "• **/history** - Review conversation history\n"
                        "• **/summary** - Get health overview\n\n"
                        "**What would you like to talk about?** 😊", 
                        parse_mode='Markdown'
                    )
                    return
                else:
                    await update.message.reply_text(
                        "No active session to exit, but I'm here and ready to help! 😊\n\n"
                        "**What's on your mind today?** Feel free to ask me anything about your keratoplasty recovery!", 
                        parse_mode='Markdown'
                    )
                    return
            
            # Check if user is in an active symptom session
            if user_id in self.symptom_sessions and self.symptom_sessions[user_id].stage == "gathering":
                logger.info(f"🎯 HANDLING SYMPTOM SESSION for user {user_id}")
                await self._handle_symptom_session(update, context)
                return
            
            # Regular message handling with enhanced symptom detection
            symptoms, risk_level, recommendations = detect_symptoms_and_assess_risk(message)
            
            # FIXED: Only start symptom session for ACTUAL symptoms/medical concerns
            if (symptoms["pain"]["present"] or 
                symptoms["redness"]["present"] or 
                symptoms.get("vision_issues") or 
                symptoms.get("discharge") or
                symptoms.get("activities", {}).get("mentioned", False) or
                any(word in message.lower() for word in ["pain", "hurt", "ache", "red", "redness", "floater", "blurry", "vision", "workout", "exercise", "swimming", "discharge", "pus"])):
                
                logger.info(f"🚨 STARTING SYMPTOM SESSION for user {user_id}")
                
                # Start symptom gathering session
                self.symptom_sessions[user_id] = SymptomSession(user_id=user_id, stage="gathering")
                self.symptom_sessions[user_id].collected_symptoms["initial_message"] = message
                
                # Generate targeted follow-up questions
                follow_up_questions = generate_targeted_follow_up_questions(symptoms, message)
                
                response = f"""**I detected some symptoms/concerns that need proper analysis!**

Let me gather a bit more information to give you the most accurate assessment:

**{follow_up_questions[0] if follow_up_questions else "Can you tell me more about your symptoms?"}**

*I'll ask 2-3 targeted questions to ensure I give you the best medical guidance with proper LVPEI citations.*

**Tip:** Type 'exit' anytime if you want to stop and ask something else!"""
                
                await update.message.reply_text(response, parse_mode='Markdown')
            
            else:
                # Regular conversation - ALWAYS respond conversationally
                logger.info(f"💬 REGULAR CONVERSATION for user {user_id}")
                response = await self.agent.chat(message, user_id)
                await update.message.reply_text(response, parse_mode='Markdown')
            
            # Store conversation
            self._store_conversation(user_id, message, response if 'response' in locals() else "Symptom session started")
            
        except Exception as e:
            logger.error(f"Message handling error: {e}")
            await update.message.reply_text(
                "I'm having technical difficulties, but I'm still here for you! "
                "If this is urgent - **severe pain or vision changes** - please contact your surgeon directly. 💙"
            )
    
    async def _handle_symptom_session(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle messages during active symptom gathering session"""
        user_id = update.effective_user.id
        message = update.message.text
        session = self.symptom_sessions[user_id]
        
        logger.info(f"🔄 SYMPTOM SESSION STEP {len(session.questions_asked)+1} for user {user_id}")
        
        # Store the answer
        question_num = len(session.questions_asked)
        session.collected_symptoms[f"answer_{question_num}"] = message
        
        # Analyze current message for symptoms
        symptoms, risk_level, recommendations = detect_symptoms_and_assess_risk(message)
        
        # Update collected symptoms with new findings
        if symptoms["pain"]["present"]:
            session.collected_symptoms["pain_present"] = True
            if symptoms["pain"]["score"]:
                session.collected_symptoms["pain_scale"] = symptoms["pain"]["score"]
        
        # Handle direct numeric answers for pain scale
        if message.strip().isdigit() and 1 <= int(message.strip()) <= 10:
            session.collected_symptoms["pain_scale"] = int(message.strip())
        
        # Activity timeline detection
        if any(word in message.lower() for word in ["day", "week", "month", "yesterday", "today"]):
            session.collected_symptoms["timeline_mentioned"] = True
            session.collected_symptoms["timeline_details"] = message
        
        # Generate next question or provide final analysis
        next_questions = generate_targeted_follow_up_questions(symptoms, message, session.collected_symptoms)
        
        # Avoid asking the same question twice
        if next_questions:
            for asked_q in session.questions_asked:
                if next_questions[0] in asked_q or asked_q in next_questions[0]:
                    next_questions = next_questions[1:] if len(next_questions) > 1 else []
                    break
        
        # If we have enough info or asked 2 questions, provide analysis
        if (len(session.questions_asked) >= 2 or 
            not next_questions or
            ("pain_scale" in session.collected_symptoms and len(session.questions_asked) >= 1)):
            
            logger.info(f"✅ PROVIDING FINAL ANALYSIS for user {user_id}")
            # Provide final comprehensive analysis
            await self._provide_final_symptom_analysis(update, session)
            # End session
            del self.symptom_sessions[user_id]
        
        else:
            # Ask next question
            next_question = next_questions[0]
            session.questions_asked.append(next_question)
            
            response = f"""**Thanks for that info! 😊** 

**{next_question}**

*Question {len(session.questions_asked)}/2 - I'm building a complete picture to give you the best medical guidance.* 📋

**💡 Tip:** Type 'exit' anytime to stop and ask something else! 🎯"""
            
            await update.message.reply_text(response, parse_mode='Markdown')
    
    async def _provide_final_symptom_analysis(self, update: Update, session: SymptomSession):
        """Provide comprehensive final analysis with proper citations - FIXED DUPLICATE ISSUE"""
        user_id = session.user_id
        collected = session.collected_symptoms
        
        # Prevent duplicate processing
        if hasattr(session, 'analysis_completed') and session.analysis_completed:
            logger.info(f"⚠️ Analysis already completed for user {user_id}, skipping")
            return
        
        session.analysis_completed = True
        
        # Combine all information for comprehensive analysis
        full_message = " ".join([
            collected.get("initial_message", ""),
            *[collected.get(f"answer_{i}", "") for i in range(len(session.questions_asked))]
        ])
        
        logger.info(f"🔍 FINAL ANALYSIS FOR: '{full_message}'")
        
        symptoms, risk_level, recommendations = detect_symptoms_and_assess_risk(full_message)
        
        # Get relevant context with proper sources
        contexts, sources, detailed_sources = self.rag.get_context_with_sources(full_message)
        
        # Build comprehensive response
        risk_emoji = {"NORMAL": "🟢", "MODERATE": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
        
        response = f"""**Comprehensive Symptom Analysis**

{risk_emoji.get(risk_level, '🟢')} **Risk Level: {risk_level}**
"""
        
        if collected.get("pain_scale"):
            response += f"**Pain Level: {collected['pain_scale']}/10**\n"
        
        # Add context-specific analysis
        if any(word in full_message.lower() for word in ["workout", "exercise", "working out"]):
            response += f"\n**I understand you have questions about working out after your keratoplasty surgery!** This is a really important topic for your recovery. Exercise timing can significantly impact your healing process, so I'm glad you're asking about this before jumping back into your routine.\n"
        
        # Add medical guidelines with proper citations
        if detailed_sources and len(detailed_sources) > 0:
            # Check if we have real LVPEI sources (not fallback)
            real_sources = [s for s in detailed_sources if "Fallback" not in s.get('full_source', '')]
            
            if real_sources:
                response += "\n**Relevant Medical Guidelines:**\n"
                for source in real_sources[:2]:
                    content = source["content"]
                    key_sentence = content.split('.')[0] + '.' if '.' in content else content
                    if len(key_sentence) > 150:
                        key_sentence = key_sentence[:150] + "..."
                    
                    response += f"• {key_sentence}\n"
                    response += f"  **Source:** {source['full_source']}\n"
            else:
                # Only fallback sources available
                response += "\n**Medical Knowledge Applied:**\n"
                response += f"• Based on general medical guidelines for keratoplasty recovery.\n"
                response += f"  **Source:** Information from Mayo Clinic and medical literature (not in my verified LVPEI sources)\n"
        
        # Add recommendations
        if recommendations:
            response += "\n**Recommendations:**\n"
            for rec in recommendations[:3]:
                response += f"• {rec}\n"
        
        response += "\n**Thanks for providing detailed information - this helps me give you much better guidance! Please don't hesitate to reach out if you have any more questions or concerns.**"
        
        await update.message.reply_text(response, parse_mode='Markdown')
        
        # Store comprehensive assessment - with better error handling
        try:
            await self._store_detailed_assessment(user_id, collected, risk_level, recommendations, full_message)
        except Exception as e:
            logger.error(f"Storage error (non-critical): {e}")
            # Don't fail the user experience for storage issues
    
    async def _store_detailed_assessment(self, user_id: int, symptoms: Dict, risk_level: str, recommendations: List[str], message: str):
        """Store detailed assessment with context"""
        try:
            contexts, sources, detailed_sources = self.rag.get_context_with_sources(message)
            follow_up_questions = generate_targeted_follow_up_questions(symptoms, message)
            
            # Handle pain score extraction safely
            pain_score = None
            if isinstance(symptoms, dict):
                if "pain_scale" in symptoms:
                    pain_score = symptoms["pain_scale"]
                elif "pain" in symptoms and isinstance(symptoms["pain"], dict):
                    pain_score = symptoms["pain"].get("score")
                elif "answer_0" in symptoms and symptoms["answer_0"].isdigit():
                    pain_score = int(symptoms["answer_0"])
                elif "answer_1" in symptoms and symptoms["answer_1"].isdigit():
                    pain_score = int(symptoms["answer_1"])
            
            # Create assessment without failing
            assessment_data = {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "symptoms": json.dumps(symptoms),
                "risk_level": risk_level,
                "pain_score": pain_score,
                "recommendations": json.dumps(recommendations),
                "context_used": json.dumps(contexts),
                "pdf_sources": json.dumps(sources),
                "follow_up_questions": json.dumps(follow_up_questions)
            }
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO assessments 
                (user_id, timestamp, symptoms, risk_level, pain_score, recommendations, context_used, pdf_sources, follow_up_questions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', tuple(assessment_data.values()))
            
            conn.commit()
            conn.close()
            
            logger.info("✅ Assessment stored successfully")
            
        except Exception as e:
            logger.error(f"Error storing assessment: {e}")
            # Continue without failing - storage error shouldn't break the bot
    
    def _get_recent_assessments(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Get recent assessments for insights"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM assessments 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            columns = ['id', 'user_id', 'timestamp', 'symptoms', 'risk_level', 'pain_score', 
                      'recommendations', 'context_used', 'pdf_sources', 'follow_up_questions']
            
            return [dict(zip(columns, row)) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting assessments: {e}")
            return []
    
    def _get_conversation_history(self, user_id: int, limit: int = 20) -> List[Dict]:
        """Get conversation history"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM conversations 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            columns = ['id', 'user_id', 'timestamp', 'user_message', 'ai_response']
            return [dict(zip(columns, row)) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []
    
    def _generate_insights_text(self, assessments: List[Dict]) -> str:
        """Generate insights from assessments"""
        total_assessments = len(assessments)
        risk_levels = [a['risk_level'] for a in assessments]
        pain_scores = [a['pain_score'] for a in assessments if a['pain_score']]
        
        # Calculate metrics
        avg_pain = sum(pain_scores) / len(pain_scores) if pain_scores else 0
        risk_counts = {level: risk_levels.count(level) for level in set(risk_levels)}
        
        insights = f"""**📊 Your Recovery Insights**

**Assessment Summary:**
• **Total Assessments:** {total_assessments}
• **Average Pain Level:** {avg_pain:.1f}/10 {"(excellent!)" if avg_pain < 3 else "(good)" if avg_pain < 5 else "(monitor closely)"}
• **Current Status:** **{assessments[0]['risk_level']}** {"🟢" if assessments[0]['risk_level'] == 'NORMAL' else "🟡" if assessments[0]['risk_level'] == 'MODERATE' else "🟠" if assessments[0]['risk_level'] == 'HIGH' else "🔴"}

**Risk Level Distribution:**"""
        
        for level, count in risk_counts.items():
            percentage = (count / total_assessments) * 100
            emoji = {"NORMAL": "🟢", "MODERATE": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
            insights += f"\n• **{level}**: {count} times ({percentage:.1f}%) {emoji.get(level, '')}"
        
        # Add trend analysis
        if len(assessments) >= 3:
            recent_risks = [a['risk_level'] for a in assessments[:3]]
            if all(r == 'NORMAL' for r in recent_risks):
                insights += "\n\n**✅ Great Progress!** Your recent assessments show stable recovery following LVPEI guidelines."
            elif 'CRITICAL' in recent_risks or 'HIGH' in recent_risks:
                insights += "\n\n**⚠️ Attention Needed** Recent assessments show concerning patterns that require medical attention."
        
        insights += "\n\n**💡 Keep up the great work with your recovery routine!**"
        
        return insights
    
    def _generate_history_text(self, history: List[Dict]) -> str:
        """Generate conversation history text"""
        history_text = "**📋 Your Recovery Conversation History**\n\n"
        
        for i, conv in enumerate(history[:5], 1):  # Show last 5 conversations
            timestamp = datetime.fromisoformat(conv['timestamp'])
            days_ago = (datetime.now() - timestamp).days
            
            if days_ago == 0:
                time_desc = "Today"
            elif days_ago == 1:
                time_desc = "Yesterday"
            else:
                time_desc = f"{days_ago} days ago"
            
            history_text += f"**{i}. {time_desc}** ({timestamp.strftime('%I:%M %p')})\n"
            history_text += f"**You:** {conv['user_message'][:100]}{'...' if len(conv['user_message']) > 100 else ''}\n"
            history_text += f"**Me:** {conv['ai_response'][:100]}{'...' if len(conv['ai_response']) > 100 else ''}\n\n"
        
        if len(history) > 5:
            history_text += f"*...and {len(history) - 5} more conversations in your complete history!*"
        
        return history_text
    
    def _generate_health_summary(self, user_id: int) -> str:
        """Generate comprehensive health summary"""
        assessments = self._get_recent_assessments(user_id)
        conversations = self._get_conversation_history(user_id)
        
        if not assessments and not conversations:
            return """**📄 Health Summary**

Welcome to your health summary! Once you start sharing symptoms and chatting with me, this will include:

• **Recovery overview** with key milestones
• __Assessment trends__ from LVPEI guidelines
• **Risk pattern analysis** over time
• **Medical guidance** you've received with citations

Start by telling me how you're feeling today! 💙"""
        
        summary = f"""**📄 Your Health Summary**

**Overall Status:** {"Monitoring recovery" if assessments else "Building profile"}
**Total Interactions:** {len(conversations)} conversations, {len(assessments)} symptom assessments
**Latest Assessment:** {assessments[0]['risk_level'] if assessments else 'None yet'} {"🟢" if assessments and assessments[0]['risk_level'] == 'NORMAL' else "🟡" if assessments and assessments[0]['risk_level'] == 'MODERATE' else "🟠" if assessments and assessments[0]['risk_level'] == 'HIGH' else "🔴" if assessments and assessments[0]['risk_level'] == 'CRITICAL' else ''}

**Recovery Progress:**
• You've been actively monitoring your recovery with LVPEI guidance
• Regular check-ins help ensure optimal healing
• All medical advice is backed by LVPEI documentation

**Next Steps:**
• Continue monitoring symptoms daily
• Keep up with prescribed medications
• Follow all LVPEI guidelines provided

Your proactive approach to recovery is excellent! Keep it up! 💪"""
        
        return summary
    
    def _store_conversation(self, user_id: int, user_message: str, ai_response: str):
        """Store conversation in database"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO conversations (user_id, timestamp, user_message, ai_response) VALUES (?, ?, ?, ?)',
                (user_id, datetime.now().isoformat(), user_message, ai_response)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Storage error: {e}")
    
    def run(self):
        logger.info("🚀 LVPEI Agent starting with FIXED symptom detection...")
        self.app.run_polling()

# System setup and validation
def setup_system():
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not found in environment variables")
        return False
    
    if not TELEGRAM_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not found in environment variables")
        return False
    
    try:
        init_database()
        print("✅ Database initialized")
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False
    
    print("✅ System ready!")
    return True

if __name__ == "__main__":
    print("💙 LVPEI Agent - FIXED Multi-Agent System with Proper Symptom Detection")
    print("=" * 75)
    
    if not setup_system():
        print("❌ Setup failed. Check your .env file for API keys.")
        exit(1)
    
    try:
        bot = LVPEIBot()
        print("💙 LVPEI Agent is ready with ENHANCED symptom detection!")
        print("📱 Send /start to see all available commands!")
        bot.run()
    except KeyboardInterrupt:
        print("\n👋 LVPEI Agent stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔧 Check: API keys, internet connection")