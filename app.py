import os
import re
import datetime
import time
import streamlit as st
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

# =========================================================
# ⚙️ MODEL AUTO-INSTALL & INITIALIZATION
# =========================================================
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
        with st.spinner("First-time launch: Downloading 0.5B model from Hugging Face..."):
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

# =========================================================
# 📊 DYNAMIC LOCAL FACT KNOWLEDGE DATABASE
# =========================================================
OFFLINE_RAG_DB = {
    "yellow spots": "Cassava Mosaic Disease (CMD). Spread by whiteflies. Action: Uproot infected plants immediately. Plant resistant stems next season.",
    "brown spot": "Cercospora Leaf Spot or fungal infection. Action: Ensure wider plant spacing for ventilation, remove lower infected foliage, and apply copper-based fungicide if severe.",
    "dried leaves": "Maize Stem Borer damage or severe drought. Action: Check stalk for holes. Apply neem extract solution directly into the funnel.",
    "spots": "General leaf spot infestation. Check for pests underneath the leaves and optimize local crop rotation.",
    "bakin doriya": "Cutar taba ganyen masara (CMD). Mataki: Cire shukan da ya rube. Yi amfani da irin da ke jure cututtuka."
}

CULTURAL_PROVERBS = [
    "Yoruba: Bééni a șe gbin gbin, bééni a ó kórè. (As we sow, so shall we reap.)",
    "Hausa: Mai hakuri yakan dafa dutse har ya sha romonsa. (The patient farmer cooks a stone and drinks its soup.)",
    "Swahili: Mvumilivu hula mbivu. (A patient person eats ripe fruit.)",
    "Igbo: Onye gba mbo na ubi, owuwe ihe ubi ga-asacha anya mmiri ya. (He who labors in the field will have his tears wiped by the harvest.)"
]

if "revenue" not in st.session_state:
    st.session_state.revenue = 0.0
if "expenses" not in st.session_state:
    st.session_state.expenses = 0.0

# Added counter to easily clear/delete all widget states instantly
if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0

# =========================================================
# 🌐 TRANSLATION DICTIONARIES
# =========================================================
LANG_DICT = {
    "English": {
        "title": "Offline Smart Farm Assistant",
        "subtitle": "Voice-First Agricultural Advisor & Ledger (Zero-Data Mode)",
        "diagnose_tab": "AI Advisor",
        "calendar_tab": "Timeline Calculator",
        "finance_tab": "Financial Ledger",
        "symptom_label": "Describe crop symptoms:",
        "submit_btn": "Ask Assistant",
        "crop_select": "Select Your Main Crop:",
        "date_input": "Planting Date:",
        "calc_btn": "Generate Farming Timeline",
        "ledger_input": "Transaction (e.g., 'I sold maize for 45000 Naira'):",
        "log_btn": "Log Transaction",
        "export_btn": "Save Local Text Report to Desktop",
        "proverb_title": "Traditional Wisdom"
    },
    "Hausa": {
        "title": "Mataimakin Manomi na Offline",
        "subtitle": "Shirin Ba da Shawara da Kula da Kudi Ba tare da Internet ba",
        "diagnose_tab": "Al Advisor",
        "calendar_tab": "Tsarin Shuka",
        "finance_tab": "Littafin Kudi",
        "symptom_label": "Kwatanta matsalar amfanin gona:",
        "submit_btn": "Tambayi Mataimaki",
        "crop_select": "Zabi Irin Shukan Ku:",
        "date_input": "Ranar Shuka:",
        "calc_btn": "Lissafi Lokutan Aiki",
        "ledger_input": "Bayanin Kudi (Misali: 'Na sayar da masara akan Naira 45000'):",
        "log_btn": "Yi Rikodin Kudi",
        "export_btn": "Ajiye Rahoto a Desktop",
        "proverb_title": "Kararin Magana"
    }
}

# =========================================================
# 🛠️ HELPER FUNCTIONS
# =========================================================
def run_ai_advisory(user_input, lang):
    clean_input = user_input.lower().strip()
    matched_fact = None
    for key, value in OFFLINE_RAG_DB.items():
        if key in clean_input:
            matched_fact = value
            break

    if matched_fact is None:
        matched_fact = f"General monitoring advised for '{user_input}'. Keep fields cleared of weeds and check irrigation intervals."

    cultural_closing = "\n\n *May your barns overflow this season! Mandani na gari!*" if lang == "Hausa" else "\n\n *May your harvest be heavy and rewarding!*"

    if (not LLAMA_AVAILABLE) or (llm is None):
        return f"**Offline RAG Result:** {matched_fact}\n\n*(Note: Running in Cloud Demo Mode. Local 0.5B model optimization triggers natively when launched offline on a farmer's laptop CPU).*\n{cultural_closing}"

    try:
        prompt = f"System: You are an African farming assistant. Use this factsheet: {matched_fact}\nUser: {user_input}\nAssistant:"
        response = llm(prompt, max_tokens=150, stop=["User:", "System:"], echo=False)
        return f"{response['choices']['text'].strip()}{cultural_closing}"
    except Exception:
        return f"**Offline RAG Result:** {matched_fact}\n\n*(Note: Running in Cloud Demo Mode. Local 0.5B model optimization triggers natively when launched offline on a farmer's laptop CPU).*\n{cultural_closing}"

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

    line1 = f"Official {crop} Production Timeline:\n"
    line2 = f"• Planting Date: {start_date.strftime('%d %B %Y')}\n"
    line3 = f"• First Fertilizer Window: {fert1.strftime('%d %B %Y')}\n"
    line4 = f"• Second Fertilizer Window: {fert2.strftime('%d %B %Y')}\n"
    line5 = f"• Optimal Harvest Window: {harvest_start.strftime('%d %B %Y')} to {harvest_end.strftime('%d %B %Y')}"
    return line1 + line2 + line3 + line4 + line5

def parse_financial_statement(statement):
    numbers = [float(s) for s in re.findall(r'\b\d+\b', statement)]
    amount = sum(numbers) if numbers else 0.0
    stmt_lower = statement.lower()
    if any(x in stmt_lower for x in ["sell", "sold", "sayar"]):
        st.session_state.revenue += amount
        return f"Logged Revenue: +{amount:,.2f} Naira"
    else:
        st.session_state.expenses += amount
        return f"Logged Expense: -{amount:,.2f} Naira"

# =========================================================
# 🖥️ STREAMLIT GRAPHICAL INTERFACE
# =========================================================
st.set_page_config(page_title="Smart Farm Assistant", layout="wide")

col_lang, col_prov = st.columns(2)
with col_lang:
    selected_lang = st.selectbox("Language / Yare", ["English", "Hausa"])
    labels = LANG_DICT[selected_lang]

with col_prov:
    prov_idx = int(time.time() // 10) % len(CULTURAL_PROVERBS)
    st.info(f"**{labels['proverb_title']}**\n{CULTURAL_PROVERBS[prov_idx]}")

st.title(labels["title"])
st.subheader(labels["subtitle"])

# Create Navigation Tabs
tab1, tab2, tab3 = st.tabs([labels["diagnose_tab"], labels["calendar_tab"], labels["finance_tab"]])

# --- TAB 1: AI ADVISOR & SYMPTOM INPUTS ---
with tab1:
    text_key = f"text_symptom_{st.session_state.input_counter}"
    audio_key = f"audio_symptom_{st.session_state.input_counter}"

    user_text = st.text_input(labels["symptom_label"], key=text_key)
    user_audio = st.audio_input("Record audio symptoms / Rikodin sauti:", key=audio_key)

    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button(labels["submit_btn"], type="primary"):
            if user_text:
                result = run_ai_advisory(user_text, selected_lang)
                st.write(result)
            elif user_audio is not None:
                st.info("Audio received locally. (Audio processing engine pipeline placeholder)")
                result = run_ai_advisory("spots", selected_lang)
                st.write(result)
            else:
                st.warning("Please provide either text or audio input first.")

    with col_btn2:
        if st.button("Delete & Clear Inputs / Goge Bayanai"):
            st.session_state.input_counter += 1
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()

# --- TAB 2: TIMELINE CALCULATOR ---
with tab2:
    selected_crop = st.selectbox(labels["crop_select"], ["Maize", "Cassava"])
    planting_date = st.date_input(labels["date_input"], datetime.date.today())
    if st.button(labels["calc_btn"]):
        timeline_results = calculate_crop_timeline(selected_crop, planting_date)
        st.text(timeline_results)

# --- TAB 3: FINANCIAL LEDGER ---
with tab3:
    ledger_text = st.text_input(labels["ledger_input"])
    if st.button(labels["log_btn"]):
        if ledger_text:
            log_result = parse_financial_statement(ledger_text)
            st.success(log_result)
        else:
            st.warning("Please enter transaction text.")
            
    st.markdown("### Farm Profit & Loss Summary / Bayanin Riba da Asara")
    
    # Calculate Total Accumulated Expenses
    total_costs = (
        st.session_state.labour_cost + 
        st.session_state.fertilizer_cost + 
        st.session_state.equipment_cost + 
        st.session_state.other_expenses
    )
    
    # Calculate Net Margin Profit
    net_profit = st.session_state.revenue - total_costs
    
    # Render Itemized Operating Lines
    st.metric("Total Sales Revenue / Kudin Sayarwa (+)", f"{st.session_state.revenue:,.2f} Naira")
    
    col_costs1, col_costs2 = st.columns(2)
    with col_costs1:
        st.metric("Labour Costs / Kudin Lebur (-)", f"{st.session_state.labour_cost:,.2f} Naira")
        st.metric("Fertilizer & Chemicals / Kudin Taki (-)", f"{st.session_state.fertilizer_cost:,.2f} Naira")
    with col_costs2:
        st.metric("Equipment & Tractor / Kayan Aiki (-)", f"{st.session_state.equipment_cost:,.2f} Naira")
        st.metric("Other Expenses / Kudaden Fitarwa (-)", f"{st.session_state.other_expenses:,.2f} Naira")
        
    st.markdown("---")
    
    # Dynamic net evaluation card layout 
    if net_profit >= 0:
        st.success(f"**Net Profit / Riba Ta Tabbata:** {net_profit:,.2f} Naira 🎉")
    else:
        st.error(f"**Net Operating Loss / Asara Ta Fito:** {abs(net_profit):,.2f} Naira ⚠️")

