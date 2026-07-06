import os
import re
import datetime
import time
import streamlit as st
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

# ==========================================
# ⚙️ MODEL AUTO-INSTALL & INITIALIZATION
# ==========================================
MODEL_DIR = "models"
MODEL_NAME = "qwen1_5-0_5b-chat-q4_k_m.gguf" 
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

@st.cache_resource
def initialize_offline_cores():
    """Checks local storage on first launch; downloads model from Hugging Face if missing."""
    if not LLAMA_AVAILABLE:
        return None
        
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        with st.spinner("📦 First-time launch: Downloading 0.5B model from Hugging Face..."):
            try:
                hf_hub_download(
                    repo_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
                    filename=MODEL_NAME,
                    local_dir=MODEL_DIR,
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                st.error(f"Failed to auto-download model: {e}")
                return None

    try:
        return Llama(model_path=MODEL_PATH, n_ctx=1024, n_threads=4)
    except Exception:
        return None

llm = initialize_offline_cores()

# ==========================================
# 📦 DYNAMIC LOCAL FACT KNOWLEDGE DATABASE
# ==========================================
OFFLINE_RAG_DB = {
    "yellow spots": "Cassava Mosaic Disease (CMD). Spread by whiteflies. Action: Uproot infected plants immediately. Plant resistant stems next season.",
    "brown spot": "Cercospora Leaf Spot or fungal infection. Action: Ensure wider plant spacing for ventilation, remove lower infected foliage, and apply copper-based fungicide if severe.",
    "dried leaves": "Maize Stem Borer damage or severe drought. Action: Check stalk for holes. Apply neem extract solution directly into the funnel.",
    "spots": "General leaf spot infestation. Check for pests underneath the leaves and optimize local crop rotation.",
    "bakin doriya": "Cutar taba ganyen masara (CMD). Mataki: Cire shukan da ya rube. Yi amfani da irin da ke jure cututtuka."
}

CULTURAL_PROVERBS = [
    "Yoruba: Bẹ́ẹ́ni a ṣe gbin gbin, bẹ́ẹ́ni a ó kórè. (As we sow, so shall we reap.)",
    "Hausa: Mai hakuri yakan dafa dutse har ya sha romonsa. (The patient farmer cooks a stone and drinks its soup.)",
    "Swahili: Mvumilivu hula mbivu. (A patient person eats ripe fruit.)",
    "Igbo: Onye gba mbo na ubi, owuwe ihe ubi ga-asacha anya mmiri ya. (He who labors in the field will have his tears wiped by the harvest.)"
]

# Initialize Financial State Variables
if "revenue" not in st.session_state:
    st.session_state.revenue = 0.0
if "expenses" not in st.session_state:
    st.session_state.expenses = 0.0

# 🌟 NEW: Initialize Input State Variables for Clearing Functionality
if "text_input_value" not in st.session_state:
    st.session_state.text_input_value = ""
if "audio_input_key" not in st.session_state:
    st.session_state.audio_input_key = 0

# ==========================================
# 🌐 TRANSLATION DICTIONARIES
# ==========================================
LANG_DICT = {
    "English": {
        "title": "🌾 Offline Smart Farm Assistant",
        "subtitle": "Voice-First Agricultural Advisor & Ledger (Zero-Data Mode)",
        "diagnose_tab": "🤖 AI Advisor",
        "calendar_tab": "📅 Timeline Calculator",
        "finance_tab": "💰 Financial Ledger",
        "symptom_label": "Describe crop symptoms:",
        "audio_label": "Or upload an audio symptom description:",
        "submit_btn": "Ask Assistant",
        "clear_btn": "🗑️ Clear Inputs",
        "crop_select": "Select Your Main Crop:",
        "date_input": "Planting Date:",
        "calc_btn": "Generate Farming Timeline",
        "ledger_input": "Transaction (e.g., 'I sold maize for 45000 Naira'):",
        "log_btn": "Log Transaction",
        "export_btn": "💾 Save Local Text Report to Desktop",
        "proverb_title": "🌟 Traditional Wisdom"
    },
    "Hausa": {
        "title": "🌾 Mataimakin Manomi na Offline",
        "subtitle": "Shirin Ba da Shawara da Kula da Kudi Ba tare da Internet ba",
        "diagnose_tab": "🤖 AI Advisor",
        "calendar_tab": "📅 Tsarin Shuka",
        "finance_tab": "💰 Littafin Kudi",
        "symptom_label": "Kwatanta matsalar amfanin gona:",
        "audio_label": "Ko kuma sanya rikodin muryar ku:",
        "submit_btn": "Tambayi Mataimaki",
        "clear_btn": "🗑️ Goge Bayanai",
        "crop_select": "Zaɓi Irin Shukan Ku:",
        "date_input": "Ranar Shuka:",
        "calc_btn": "Lissafi Lokutan Aiki",
        "ledger_input": "Bayanin Kudi (Misali: 'Na sayar da masara akan Naira 45000'):",
        "log_btn": "Yi Rikodin Kudi",
        "export_btn": "💾 Ajiye Rahoto a Desktop",
        "proverb_title": "🌟 Kararin Magana"
    }
}

# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================
def run_ai_advisory(user_input, lang):
    clean_input = user_input.lower().strip()
    
    matched_fact = None
    for key, value in OFFLINE_RAG_DB.items():
        if key in clean_input:
            matched_fact = value
            break
            
    if matched_fact is None:
        matched_fact = f"General monitoring advised for '{user_input}'. Keep fields cleared of weeds and check irrigation intervals."

    cultural_closing = "\n\n🌍 *May your barns overflow this season! Mandani na gari!*" if lang == "Hausa" else "\n\n🌍 *May your harvest be heavy and rewarding!*"

    if (not LLAMA_AVAILABLE) or (llm is None):
        return f"💡 **Offline RAG Result:** {matched_fact}\n\n*(Note: Running in Cloud Demo Mode. Local 0.5B model optimization triggers natively when launched offline on a farmer's laptop CPU).* {cultural_closing}"

    try:
        prompt = f"System: You are an African farming assistant. Use this factsheet: {matched_fact}\nUser: {user_input}\nAssistant:"
        response = llm(prompt, max_tokens=150, stop=["User:", "System:"], echo=False)
        return f"{response['choices']['text'].strip()}{cultural_closing}"
    except Exception:
        return f"💡 **Offline RAG Result:** {matched_fact}\n\n*(Note: Running in Cloud Demo Mode. Local 0.5B model optimization triggers natively when launched offline on a farmer's laptop CPU).* {cultural_closing}"

def calculate_crop_timeline(crop, start_date):
    if crop == "Maize":
        fert1 = start_date + datetime.timedelta(days=21)
        fert2 = start_date + datetime.timedelta(days=42)
        harvest_start = start_date + datetime.timedelta(days=90)
        harvest_end = start_date + datetime.timedelta(days=120)
    else:  # Cassava
        fert1 = start_date + datetime.timedelta(days=30)
        fert2 = start_date + datetime.timedelta(days=90)
        harvest_start = start_date + datetime.timedelta(days=270)
        harvest_end = start_date + datetime.timedelta(days=360)

    line1 = f"📅 Official {crop} Production Timeline:\n"
    line2 = f"• Planting Date: {start_date.strftime('%d %B %Y')}\n"
    line3 = f"• 🚨 First Fertilizer Window: {fert1.strftime('%d %B %Y')}\n"
    line4 = f"• 🚨 Second Fertilizer Window: {fert2.strftime('%d %B %Y')}\n"
    line5 = f"• 🌾 Optimal Harvest Window: {harvest_start.strftime('%d %B %Y')} to {harvest_end.strftime('%d %B %Y')}"
    return line1 + line2 + line3 + line4 + line5

def parse_financial_statement(statement):
    numbers = [float(s) for s in re.findall(r'\b\d+\b', statement)]
    amount = sum(numbers) if numbers else 0.0
    
    stmt_lower = statement.lower()
    if any(x in stmt_lower for x in ["sell", "sold", "sayar"]):
        st.session_state.revenue += amount
        return f"💰 Logged Revenue: +{amount:,.2f} Naira"
    else:
        st.session_state.expenses += amount
        return f"📉 Logged Expense: -{amount:,.2f} Naira"

# 🌟 NEW: Core function to reset the state values
def clear_inputs():
    """Wipes the text string and increments the file uploader key to force a clean re-render."""
    st.session_state.text_input_value = ""
    st.session_state.audio_input_key += 1

# ==========================================
# 🎨 STREAMLIT GRAPHICAL INTERFACE
# ==========================================
st.set_page_config(page_title="Smart Farm Assistant", layout="wide")

col_lang, col_prov = st.columns(2)
with col_lang:
    selected_lang = st.selectbox("🌐 Language / Yare", ["English", "Hausa"])

labels = LANG_DICT[selected_lang]

with col_prov:
    prov_idx = int(time.time() // 10) % len(CULTURAL_PROVERBS)
    st.info(f"**{labels['proverb_title']}:** {CULTURAL_PROVERBS[prov_idx]}")

st.title(labels["title"])
st.subheader(labels["subtitle"])

# Creating tabs for navigation
tab_advisor, tab_timeline, tab_ledger = st.tabs([labels["diagnose_tab"], labels["calendar_tab"], labels["finance_tab"]])

# --- AI ADVISOR TAB ---
with tab_advisor:
    st.write("---")
    
    # Text input synced directly with st.session_state
    user_text = st.text_input(
        labels["symptom_label"], 
        value=st.session_state.text_input_value, 
        key="text_input_field"
    )
    # Sync visual changes back to the tracked variable
    st.session_state.text_input_value = user_text

    # Audio input element using a dynamic unique key tracker to allow forcing clean resets
    audio_file = st.file_uploader(
        labels["audio_label"], 
        type=["wav", "mp3", "m4a"], 
        key=f"audio_uploader_{st.session_state.audio_input_key}"
    )

    if audio_file is not None:
        st.audio(audio_file, format="audio/wav")
        st.success("📢 Audio input registered locally.")

    # Action layout buttons side-by-side
    btn_col1, btn_col2 = st.columns([1, 5])
    with btn_col1:
