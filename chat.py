# MUST BE THE VERY FIRST LINES IN THE FILE
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import time
import datetime
import os
import re
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
TODAYS_DATE = datetime.datetime.now()
API_KEY = os.environ.get("GOOGLE_API_KEY")
PERSIST_DIR = "./chroma_db_1"
MAX_HISTORY = 20
SUPPORTED_LANGUAGES = {
    "English": {"code": "en", "script": "Latin"},
    "हिन्दी": {"code": "hi", "script": "Devanagari"},
    "தமிழ்": {"code": "ta", "script": "Tamil"},
    "తెలుగు": {"code": "te", "script": "Telugu"},
    "ಕನ್ನಡ": {"code": "kn", "script": "Kannada"},
    "മലയാളം": {"code": "ml", "script": "Malayalam"},
    "मराठी": {"code": "mr", "script": "Devanagari"},
    "বাংলা": {"code": "bn", "script": "Bengali"},
    "ગુજરાતી": {"code": "gu", "script": "Gujarati"},
    "ਪੰਜਾਬੀ": {"code": "pa", "script": "Gurmukhi"},
    "اردو": {"code": "ur", "script": "Arabic"}
}

SCRIPT_FONTS = {
    "Devanagari": "Noto Sans Devanagari",
    "Tamil": "Noto Sans Tamil",
    "Telugu": "Noto Sans Telugu",
    "Kannada": "Noto Sans Kannada",
    "Malayalam": "Noto Sans Malayalam",
    "Bengali": "Noto Sans Bengali",
    "Gujarati": "Noto Sans Gujarati",
    "Gurmukhi": "Noto Sans Gurmukhi",
    "Arabic": "Noto Sans Arabic",
    "Latin": "Arial"
}

UI_TRANSLATIONS = {
    "en": {
        "title": "GSV Academic Assistant",
        "placeholder": "Ask about college information...",
        "clear_chat": "Clear Chat History",
        "analyzing": "Analyzing your question...",
        "complete": "Analysis complete",
        "error": "Error processing request",
        "processed": "Processed in {time:.2f} seconds"
    },
    "hi": {
        "title": "जीएसवी शैक्षणिक सहायक",
        "placeholder": "कॉलेज की जानकारी पूछें...",
        "clear_chat": "चैट इतिहास साफ करें",
        "analyzing": "आपके प्रश्न का विश्लेषण किया जा रहा है...",
        "complete": "विश्लेषण पूर्ण",
        "error": "अनुरोध प्रसंस्करण में त्रुटि",
        "processed": "{time:.2f} सेकंड में संसाधित"
    },
    "ta": {
        "title": "GSV கல்வி உதவியாளர்",
        "placeholder": "கல்லூரி தகவலைக் கேட்க...",
        "clear_chat": "அரட்டை வரலாற்றை அழி",
        "analyzing": "உங்கள் கேள்வி பகுப்பாய்வு செய்யப்படுகிறது...",
        "complete": "பகுப்பாய்வு முடிந்தது",
        "error": "கோரிக்கையை செயலாக்குவதில் பிழை",
        "processed": "{time:.2f} வினாடிகளில் செயலாக்கப்பட்டது"
    },
    "te": {
        "title": "GSV ఎడ్యుకేషనల్ అసిస్టెంట్",
        "placeholder": "కళాశాల సమాచారం అడగండి...",
        "clear_chat": "చాట్ చరిత్ర తొలగించు",
        "analyzing": "మీ ప్రశ్నను విశ్లేషిస్తున్నాము...",
        "complete": "విశ్లేషణ పూర్తయింది",
        "error": "విన్నపం ప్రాసెస్ చేయడంలో లోపం",
        "processed": "{time:.2f} సెకన్లలో ప్రాసెస్ చేయబడింది"
    },
    "kn": {
        "title": "GSV ಶೈಕ್ಷಣಿಕ ಸಹಾಯಕ",
        "placeholder": "ಕಾಲೇಜು ಮಾಹಿತಿಯನ್ನು ಕೇಳಿ...",
        "clear_chat": "ಚಾಟ್ ಇತಿಹಾಸವನ್ನು ಅಳಿಸಿ",
        "analyzing": "ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ...",
        "complete": "ವಿಶ್ಲೇಷಣೆ ಪೂರ್ಣಗೊಂಡಿದೆ",
        "error": "ವಿನಂತಿಯನ್ನು ಸಂಸ್ಕರಿಸುವಲ್ಲಿ ದೋಷ",
        "processed": "{time:.2f} ಸೆಕೆಂಡುಗಳಲ್ಲಿ ಸಂಸ್ಕರಿಸಲಾಗಿದೆ"
    },
    "ml": {
        "title": "GSV വിദ്യാഭ്യാസ സഹായി",
        "placeholder": "കോളേജ് വിവരങ്ങൾ ചോദിക്കുക...",
        "clear_chat": "ചാറ്റ് ചരിത്രം മായ്ക്കുക",
        "analyzing": "നിങ്ങളുടെ ചോദ്യം വിശകലനം ചെയ്യുന്നു...",
        "complete": "വിശകലനം പൂർത്തിയായി",
        "error": "അഭ്യർത്ഥന പ്രോസസ്സ് ചെയ്യുന്നതിൽ പിശക്",
        "processed": "{time:.2f} സെക്കൻഡിൽ പ്രോസസ്സ് ചെയ്തു"
    },
    "mr": {
        "title": "GSV शैक्षणिक सहाय्यक",
        "placeholder": "कॉलेज माहिती विचारा...",
        "clear_chat": "चॅट इतिहास साफ करा",
        "analyzing": "तुमच्या प्रश्नाचे विश्लेषण केले जात आहे...",
        "complete": "विश्लेषण पूर्ण",
        "error": "विनंती प्रक्रिया करताना त्रुटी",
        "processed": "{time:.2f} सेकंदात प्रक्रिया केले"
    },
    "bn": {
        "title": "GSV একাডেমিক সহায়ক",
        "placeholder": "কলেজের তথ্য জিজ্ঞাসা করুন...",
        "clear_chat": "চ্যাট ইতিহাস সাফ করুন",
        "analyzing": "আপনার প্রশ্ন বিশ্লেষণ করা হচ্ছে...",
        "complete": "বিশ্লেষণ সম্পূর্ণ",
        "error": "অনুরোধ প্রক্রিয়াকরণে ত্রুটি",
        "processed": "{time:.2f} সেকেন্ডে প্রক্রিয়াকৃত"
    },
    "gu": {
        "title": "GSV શૈક્ષણિક સહાયક",
        "placeholder": "કોલેજની માહિતી પૂછો...",
        "clear_chat": "ચેટ ઇતિહાસ સાફ કરો",
        "analyzing": "તમારા પ્રશ્નનું વિશ્લેષણ કરવામાં આવી રહ્યું છે...",
        "complete": "વિશ્લેષણ પૂર્ણ",
        "error": "રિક્વેસ્ટ પ્રોસેસ કરવામાં ભૂલ",
        "processed": "{time:.2f} સેકન્ડમાં પ્રોસેસ કર્યું"
    },
    "pa": {
        "title": "GSV ਐਕਾਡਮਿਕ ਸਹਾਇਕ",
        "placeholder": "ਕਾਲਜ ਦੀ ਜਾਣਕਾਰੀ ਪੁੱਛੋ...",
        "clear_chat": "ਚੈਟ ਇਤਿਹਾਸ ਸਾਫ਼ ਕਰੋ",
        "analyzing": "ਤੁਹਾਡੇ ਸਵਾਲ ਦਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕੀਤਾ ਜਾ ਰਿਹਾ ਹੈ...",
        "complete": "ਵਿਸ਼ਲੇਸ਼ਣ ਪੂਰਾ",
        "error": "ਬੇਨਤੀ ਪ੍ਰਕਿਰਿਆ ਕਰਨ ਵਿੱਚ ਗਲਤੀ",
        "processed": "{time:.2f} ਸਕਿੰਟਾਂ ਵਿੱਚ ਪ੍ਰਕਿਰਿਆ ਕੀਤੀ"
    },
    "ur": {
        "title": "GSV تعلیمی معاون",
        "placeholder": "کالج کی معلومات پوچھیں...",
        "clear_chat": "چیٹ تاریخ صاف کریں",
        "analyzing": "آپ کے سوال کا تجزیہ کیا جا رہا ہے...",
        "complete": "تجزیہ مکمل",
        "error": "درخواست پراسیس کرنے میں خرابی",
        "processed": "{time:.2f} سیکنڈ میں پراسیس کیا گیا"
    }
}

def sanitize_text(text: str, script: str) -> str:
    """Sanitize input/output based on script requirements"""
    script_ranges = {
        "Devanagari": r'\u0900-\u097F',
        "Tamil": r'\u0B80-\u0BFF',
        "Telugu": r'\u0C00-\u0C7F',
        "Kannada": r'\u0C80-\u0CFF',
        "Malayalam": r'\u0D00-\u0D7F',
        "Bengali": r'\u0980-\u09FF',
        "Gujarati": r'\u0A80-\u0AFF',
        "Gurmukhi": r'\u0A00-\u0A7F',
        "Arabic": r'\u0600-\u06FF',
        "Latin": r'a-zA-Z'
    }
    
    # Common characters with proper escaping and ordering
    common_chars = r" .,!?'()\-"
    
    # Build pattern with proper escaping
    base_range = script_ranges.get(script, "a-zA-Z")
    pattern = f'[^{base_range}{common_chars}]'
    
    # Compile with UNICODE flag
    regex = re.compile(pattern, flags=re.UNICODE)
    return regex.sub('', text)

def apply_font(script: str):
    """Inject font CSS for proper script rendering"""
    font_family = SCRIPT_FONTS.get(script, "Arial")
    css = f"""
    <style>
    * {{
        font-family: '{font_family}', sans-serif;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="GSV Academic Assistant",
        page_icon="🎓",
        layout="centered"
    )

    @st.cache_resource
    def initialize_components():
        embedding_model = GoogleGenerativeAIEmbeddings(
            google_api_key=API_KEY,
            model="models/embedding-001"
        )
        db_connection = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding_model
        )
        db_connection = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding_model,
            client_settings=chromadb.config.Settings(
            is_persistent=True,
            allow_reset=True,
            anonymized_telemetry=False
            )
        )
        return db_connection.as_retriever(search_kwargs={"k": 10})

    retriever = initialize_components()

    with st.sidebar:
        lang_name = st.selectbox("भाषा / Language", list(SUPPORTED_LANGUAGES.keys()))
        lang_data = SUPPORTED_LANGUAGES[lang_name]
        apply_font(lang_data["script"])
        
        clear_chat_text = UI_TRANSLATIONS[lang_data["code"]].get(
            "clear_chat", 
            UI_TRANSLATIONS["en"]["clear_chat"]
        )
        if st.button(clear_chat_text):
            st.session_state.messages = []
            st.rerun()

    # Get translations with fallback
    tr = {
        key: UI_TRANSLATIONS[lang_data["code"]].get(key, UI_TRANSLATIONS["en"][key])
        for key in ["title", "placeholder", "analyzing", "complete", "error", "processed"]
    }
    
    st.title(tr["title"])
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    user_input = st.chat_input(tr["placeholder"])
    
    if user_input:
        sanitized_input = sanitize_text(user_input, lang_data["script"])
        st.session_state.messages.append({"role": "user", "content": sanitized_input})
        
        with st.status(tr["analyzing"], expanded=True) as status:
            start_time = time.time()
            try:
                response = process_query(
                    sanitized_input,
                    "\n".join([m["content"] for m in st.session_state.messages[:-1]]),
                    retriever,
                    lang_data
                )
            except Exception as e:
                response = f"{tr['error']}: {str(e)}"
            
            processing_time = time.time() - start_time
            status.update(label=tr["complete"], state="complete")
        
        sanitized_response = sanitize_text(response, lang_data["script"])
        processed_text = tr["processed"].format(time=processing_time)
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"{sanitized_response}\n\n*({processed_text})*"
        })
        
        if len(st.session_state.messages) > MAX_HISTORY:
            st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]
        
        st.rerun()

def process_query(user_input, chat_history, retriever, lang_data):
    try:
        model = ChatGoogleGenerativeAI(
            google_api_key=API_KEY,
            model="gemini-1.5-flash",
            temperature=0.2
        )
        
        system_prompt = f"""
        You are a college information assistant. Follow these rules:
        1. Respond in {lang_data['code']} language using {lang_data['script']} script
        2. If unable to respond in target language, provide English response
        3. Base answers on provided context
        4. Maintain academic tone with cultural sensitivity
        5. You are a knowledgeable chatbot assistant specializing in providing detailed college information.
    Maintain a friendly and helpful tone while providing accurate information.
        6. Ask follow-up questions to clarify user intent
        7. Provide comprehensive answers with relevant details
        8. Use proper grammar and punctuation
        9. Avoid using slang or informal language
        10. Provide accurate and up-to-date information
        """
        
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessagePromptTemplate.from_template("""
            Context: {context}
            History: {history}
            Question: {question}
            """)
        ])
        
        chain = (
            RunnablePassthrough.assign(
                context=lambda x: retriever.get_relevant_documents(x["question"])
            ) | chat_template | model | StrOutputParser()
        )
        
        response = chain.invoke({
            "question": user_input,
            "history": chat_history
        })
        
        # Fallback mechanism
        if not contains_script(response, lang_data["script"]):
            en_response = chain.invoke({
                "question": f"Translate to English: {user_input}",
                "history": chat_history
            })
            response += f"\n\n(English): {en_response}"
        
        return response
        
    except Exception as e:
        raise RuntimeError(f"API Error: {str(e)}")

def contains_script(text: str, script: str) -> bool:
    """Verify response contains correct script characters"""
    script_patterns = {
        "Devanagari": r'[\u0900-\u097F]',
        "Tamil": r'[\u0B80-\u0BFF]',
        "Telugu": r'[\u0C00-\u0C7F]',
        "Kannada": r'[\u0C80-\u0CFF]',
        "Malayalam": r'[\u0D00-\u0D7F]',
        "Bengali": r'[\u0980-\u09FF]',
        "Gujarati": r'[\u0A80-\u0AFF]',
        "Gurmukhi": r'[\u0A00-\u0A7F]',
        "Arabic": r'[\u0600-\u06FF]'
    }
    return bool(re.search(script_patterns.get(script, r'[a-zA-Z]'), text))

if __name__ == "__main__":
    main()
    
