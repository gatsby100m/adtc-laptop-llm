# =========================================================
# BATCH 1: GLOBAL IMPORTS & SECURE MULTI-MODEL INITIALIZATION
# =========================================================
import os
import re
import datetime
import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import torch

# Core engine availability flags
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

MODEL_DIR = "models"
MODEL_NAME = "qwen1_5-0_5b-chat-q4_k_m.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

@st.cache_resource
def initialize_offline_cores():
    """
    Loads all offline core engines safely within the laptop's 8GB RAM budget.
    Loads: 1. Qwen 0.5B GGUF, 2. MiniLM Embeddings, 3. NLLB-200 Translator.
    """
    llm_instance = None
    bi_encoder = None
    trans_tokenizer = None
    trans_model = None

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Initialize Llama-CPP Reasoning Core
    if LLAMA_AVAILABLE:
        if not os.path.exists(MODEL_PATH):
            with st.spinner("Downloading Qwen1.5-0.5B-Chat weights..."):
                try:
                    from huggingface_hub import hf_hub_download
                    hf_hub_download(
                        repo_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
                        filename=MODEL_NAME,
                        local_dir=MODEL_DIR,
                        local_dir_use_symlinks=False
                    )
                except Exception as download_error:
                    st.error(f"Weights transmission aborted: {str(download_error)}")

        if os.path.exists(MODEL_PATH):
            try:
                # Optimized thread budget for dual/quad core base laptops
                llm_instance = Llama(model_path=MODEL_PATH, n_ctx=1024, n_threads=4)
            except Exception:
                llm_instance = None

    # 2. Initialize Sentence Transformers & Translation Cores
    if TRANSFORMERS_AVAILABLE:
        with st.spinner("Caching Semantic RAG Vector Map vectors..."):
            try:
                bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            except Exception:
                bi_encoder = None

        with st.spinner("Initializing Local NLLB-200 Translation Engine..."):
            try:
                # Highly optimized 600M parameter multilingual translation node
                trans_model_id = "facebook/nllb-200-distilled-600M"
                trans_tokenizer = AutoTokenizer.from_pretrained(trans_model_id)
                trans_model = AutoModelForSeq2SeqLM.from_pretrained(trans_model_id)
            except Exception as e:
                st.error(f"Translation engine initialization failed: {e}")

    return llm_instance, bi_encoder, trans_tokenizer, trans_model

# Run global systems initialization hook
llm, encoder, t_tokenizer, t_model = initialize_offline_cores()

# =========================================================
# BATCH 2: KNOWLEDGE BASES & TRANSLATION ENGINE KERNEL
# =========================================================

FARM_KNOWLEDGE_BASE = [
    "Maize Fertilizer Schedule: The first fertilizer application for maize should happen exactly 21 days after planting using NPK 15-15-15 compound fertilizer to develop roots. The second application must occur 42 days after planting using Urea to provide a high nitrogen boost for stalk growth.",
    "Cassava Leaf Spot Management: Cercospora Leaf Spot causes brown or dark spots on cassava leaves. This fungal infection thrives in humid conditions. Action: Ensure wide plant row spacing for better air ventilation, remove lower infected foliage, and apply copper-based fungicide if the outbreak is severe.",
    "Maize Stem Borer Pest Control: Maize Stem Borers are insects that tunnel holes into maize stalks, leading to withered or dried leaves in the center funnel. Action: Check stalks for small entry holes. Prepare and apply a natural neem extract solution directly into the top leaf funnel to kill larvae safely.",
    "General Soil and Watering Advice: Always monitor soil moisture before watering crops. Overwatering leads to waterlogged soil, suffocating plant roots and causing leaves to turn yellow or develop fungal spots. Keep farming plots cleared of competitive weeds.",
    "TakidabakidoriyaakasarHausa: Cutartabaganyenmasara(CMD)yanakawobakikodoriyaaganye. Matakimafikyaushinecireshukandayarubedawuridonhanayaduwa,kumaayiamfanida ingantaccenirinshukamaijurecututtuka."
]

if encoder is not None:
    db_embeddings = encoder.encode(FARM_KNOWLEDGE_BASE, convert_to_tensor=True)
else:
    db_embeddings = None

CULTURAL_PROVERBS = [
    "Yoruba: Bí énìyàn bá șe gbingbin, béèni yóò șe kórè. (As we sow, so shall we reap.)",
    "Hausa: Mai hakuri yukan dafa dutse har ya sha romonsa. (The patient farmer cooks a stone and drinks its soup.)",
    "Swahili: Mvumilivu hula mbivu. (A patient person eats ripe fruit.)",
    "Igbo: Onye gbambo na ubi, owu we ihe ubi ga-asacha anya mmiri ya. (He who labors in the field will have his tears wiped by the harvest.)"
]

# Initialize Session UI Tracking States
for key, default in [("revenue", 0.0), ("labour_cost", 0.0), ("fertilizer_cost", 0.0), 
                     ("equipment_cost", 0.0), ("other_expenses", 0.0), ("input_counter", 0)]:
    if key not in st.session_state:
        st.session_state[key] = default

LANG_DICT = {
    "English": {
        "title": "Offline Smart Farm Assistant",
        "subtitle": "Voice-First Agricultural Advisor & Ledger (Zero-Data Mode)",
        "diagnose_tab": "AI Advisor", "calendar_tab": "Timeline Calculator", "finance_tab": "Financial Ledger",
        "text_input_label": "Describe crop symptoms:", "submit_btn": "Ask Assistant",
        "crop_select": "Select Your Main Crop:", "date_input": "Planting Date:", "calc_btn": "Generate Farming Timeline",
        "ledger_input": "Transaction (e.g., 'I sold maize for 45000 Naira'):", "log_btn": "Log Transaction",
        "export_btn": "Save Local Text Report to Desktop", "proverb_title": "Traditional Wisdom"
    },
    "Hausa": {
        "title": "Mataimakin Manomi na Offline",
        "subtitle": "Shirin Bada Shawara da Kula da Kudi Ba tare da Internet ba",
        "diagnose_tab": "AI Advisor", "calendar_tab": "Tsarin Shuka", "finance_tab": "Littafin Kudi",
        "text_input_label": "Kwatanta matsalar amfanin gona:", "submit_btn": "Tambayi Mataimaki",
        "crop_select": "Zabi Irin Shukan Ku:", "date_input": "Ranar Shuka:", "calc_btn": "Lissafi Lokutan Aiki",
        "ledger_input": "Bayanin Kudi (Misali: 'Na sayar da masara akan Naira 45000'):", "log_btn": "Yi Rikodin Kudi",
        "export_btn": "Ajiye Rahoto a Desktop", "proverb_title": "Kararin Magana"
    }
}

def local_translate(text: str, target_lang_code: str) -> str:
    """Helper macro to handle string modifications using NLLB-200 safely."""
    if not t_model or not t_tokenizer or not text.strip():
        return text
    try:
        inputs = t_tokenizer(text, return_tensors="pt")
        forced_id = t_tokenizer.lang_code_to_id[target_lang_code]
        generated_tokens = t_model.generate(**inputs, forced_bos_token_id=forced_id, max_length=256)
        return t_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    except Exception:
        return text

# =========================================================
# BATCH 3: UPGRADED SANDWICH HYBRID RAG ENGINE WRAPPER
# =========================================================
def run_ai_advisory(user_input, lang):
    cultural_closing = "\n\n*May your barns overflow this season! Mandani na gari!*" if lang == "Hausa" else "\n\n*May your harvest be heavy and rewarding!*"
    matched_fact = "Advise general monitoring, checking soil moisture, clearing competitive weeds, and maintaining row spacing layout protocols."
    
    # Store the original prompt to process potential translation layers
    working_prompt = user_input
    HAUSA_CODE = "hau_Latn"
    ENGLISH_CODE = "eng_Latn"
    
    # STEP 1: If input is Hausa, translate to English for semantic search and Qwen compatibility
    if lang == "Hausa":
        working_prompt = local_translate(user_input, target_lang_code=ENGLISH_CODE)

    # STEP 2: Execute Semantic Matching over the English Document Store
    if encoder is not None and db_embeddings is not None:
        try:
            query_embedding = encoder.encode(working_prompt, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, db_embeddings)[0]
            best_match_idx = int(np.argmax(cos_scores.cpu().numpy()))
            matched_fact = FARM_KNOWLEDGE_BASE[best_match_idx]
        except Exception:
            pass

    # Safe exit path if Qwen is not loaded
    if (not LLAMA_AVAILABLE) or (llm is None):
        fallback_msg = f"**Offline Semantic Match:** {matched_fact}"
        if lang == "Hausa":
            fallback_msg = local_translate(fallback_msg, target_lang_code=HAUSA_CODE)
        return f"{fallback_msg}\n\n*(Note: Running in high-performance lookup fallback mode).*\n{cultural_closing}"

    # STEP 3: Query the main Qwen core using clean English prompts
    try:
        system_instruction = (
            "You are an expert African agricultural advisor. "
            "CRITICAL: Use the provided Factsheet Context to answer the user's question accurately. "
            "Elaborate on details to sound friendly, but your facts MUST stay completely anchored to context. "
            "Do NOT invent unrelated facts, and write ONLY in clear English text without Chinese characters."
        )
        
        prompt = (
            f"<|im_start|>system\n{system_instruction}\nFactsheet Context: {matched_fact}<|im_end|>\n"
            f"<|im_start|>user\n{working_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        response = llm(
            prompt,
            max_tokens=250,
            temperature=0.0,  # Forces facts to lock tightly onto context references
            top_p=0.1,
            stop=["<|im_end|>", "<|im_start|>", "User:", "System:"],
            echo=False
        )
        
        ai_response = response['choices'][0]['text'].strip()
        ai_response = re.sub(r'[\u4e00-\u9fff]+', '', ai_response)  # Strips rogue characters
        
        if len(ai_response) < 3:
            ai_response = f"Farming Truth Block: {matched_fact}"

        # STEP 4: If the interface language is Hausa, translate the final English answer back to Hausa
        if lang == "Hausa":
            ai_response = local_translate(ai_response, target_lang_code=HAUSA_CODE)
            
        return f"{ai_response}{cultural_closing}"
        
    except Exception as e:
        fallback_err = f"Offline Semantic Fallback: {matched_fact}"
        if lang == "Hausa":
            fallback_err = local_translate(fallback_err, target_lang_code=HAUSA_CODE)
        return f"**{fallback_err}**{cultural_closing}"

# =========================================================
# BATCH 4: TIMELINE CALCULATORS & STREAMLIT UI MOUNT
# =========================================================
def calculate_crop_timeline(crop, start_date):
    if crop == "Maize":
        fert1 = start_date + datetime.timedelta(days=21)
        fert2 = start_date + datetime.timedelta(days=42)
        harvest_start = start_date + datetime.timedelta(days=90)
        harvest_end = start_date + datetime.timedelta(days=120)
        return f"Maize Timeline:\n- 1st Fertilizer (NPK): {fert1}\n- 2nd Fertilizer (Urea): {fert2}\n- Harvest Window: {harvest_start} to {harvest_end}"
    else:
        fert1 = start_date + datetime.timedelta(days=30)
        fert2 = start_date + datetime.timedelta(days=90)
        harvest_start = start_date + datetime.timedelta(days=270)
        return f"Cassava Timeline:\n- Maintenance Window: {fert1}\n- Root Bulking Boost: {fert2}\n- Harvest Begins: {harvest_start}"

def parse_financial_statement(stmt):
    """Placeholder accounting interpreter matching your current system schema."""
    stmt_lower = stmt.lower()
    try:
        amount = float(re.findall(r'\d+', stmt)[0])
    except IndexError:
        return "Could not extract numerical value from entry."
        
    if any(x in stmt_lower for x in ["sold", "sales", "nasayar"]):
        st.session_state.revenue += amount
        return f"Logged Revenue: +{amount:,.2f} Naira"
    elif any(x in stmt_lower for x in ["labour", "worker", "lebur"]):
        st.session_state.labour_cost += amount
        return f"Logged Labour Cost: -{amount:,.2f} Naira"
    else:
        st.session_state.other_expenses += amount
        return f"Logged Miscellaneous Expense: -{amount:,.2f} Naira"

# --- RENDER WEB GRAPHICAL INTERFACE VIEWS ---
st.set_page_config(page_title="SmartFarmAssistant", layout="wide")

if llm is None:
    st.warning("Application running in dummy mode. AI vector features require active weights storage paths.")
else:
    st.success("AI Core, Translation Pipeline, and Semantic Vector Engine loaded successfully in offline mode!")

col_lang, col_prov = st.columns(2)
with col_lang:
    selected_lang = st.selectbox("Language / Yare", ["English", "Hausa"])

labels = LANG_DICT[selected_lang]

with col_prov:
    prov_idx = int(time.time() // 10) % len(CULTURAL_PROVERBS)
    st.info(f"**{labels['proverb_title']}**\n{CULTURAL_PROVERBS[prov_idx]}")

st.title(labels["title"])
st.subheader(labels["subtitle"])

tab1, tab2, tab3 = st.tabs([labels.get("diagnose_tab", "AI Advisor"), labels.get("calendar_tab", "Timeline Calculator"), labels.get("finance_tab", "Financial Ledger")])

# --- TAB 1: INTERACTIVE AI ADVISOR ---
with tab1:
    text_key = f"text_symptom_{st.session_state.get('input_counter', 0)}"
    user_text = st.text_input(labels.get("text_input_label", "Describe crop symptoms:"), key=text_key)
    
    if st.button(labels["submit_btn"], type="primary"):
        if user_text:
            with st.spinner("Processing local multi-stage pipeline translation..."):
                result = run_ai_advisory(user_text, selected_lang)
            st.write(result)
        else:
            st.warning("Please provide text input first.")

# --- TAB 2: TIMELINE CALCULATOR ---
with tab2:
    selected_crop = st.selectbox(labels["crop_select"], ["Maize", "Cassava"])
    planting_date = st.date_input(labels["date_input"], datetime.date.today())
    if st.button(labels["calc_btn"]):
        st.text(calculate_crop_timeline(selected_crop, planting_date))

# --- TAB 3: FINANCIAL ACCOUNTING MANAGER ---
with tab3:
    st.markdown("### Enter New Transactions / Shigar da Kudi")
    nlp_statement = st.text_input(labels["ledger_input"], key=f"nlp_stmt_{st.session_state.get('input_counter', 0)}")
    if st.button(labels["log_btn"]):
        if nlp_statement:
            st.info(parse_financial_statement(nlp_statement))
            st.rerun()
