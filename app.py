import os
import re
import datetime
import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Try importing local AI and vector libraries
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Try importing Hugging Face translation tools
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    HF_TRANSFORMERS_AVAILABLE = True
except ImportError:
    HF_TRANSFORMERS_AVAILABLE = False

MODEL_DIR = "models"
MODEL_NAME = "qwen1_5-0_5b-chat-q4_k_m.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

@st.cache_resource
def initialize_offline_cores():
    llm_instance = None
    bi_encoder = None
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
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
                llm_instance = Llama(model_path=MODEL_PATH, n_ctx=1024, n_threads=4)
            except Exception:
                llm_instance = None

    if TRANSFORMERS_AVAILABLE:
        with st.spinner("Caching Semantic RAG Vector Map vectors..."):
            try:
                bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            except Exception:
                bi_encoder = None
                
    return llm_instance, bi_encoder

@st.cache_resource
def load_hausa_translator_engine():
    """
    Directly loads and caches tokenizer and Seq2Seq structures for NLLB execution.
    """
    if HF_TRANSFORMERS_AVAILABLE:
        try:
            model_name = "mide6x/english-hausa-nllb"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            return tokenizer, model
        except Exception as e:
            st.error(f"Could not load Hugging Face Hausa Translator: {e}")
            return None, None
    return None, None

# Global initializations
llm, encoder = initialize_offline_cores()
hf_tokenizer, hf_model = load_hausa_translator_engine()

# =========================================================
# ADVANCED HYBRID VECTOR RAG ENGINE WITH STRICT HF TRANSLATION
# =========================================================
def run_ai_advisory(user_input, lang):
    cultural_closing = "\n\n*May your barns overflow this season! Mandani na gari!*" if lang == "Hausa" else "\n\n*May your harvest be heavy and rewarding!*"
    matched_fact = "Advise general monitoring, checking soil moisture, clearing competitive weeds, and maintaining row spacing layout protocols."
    
    # 1. Execute Semantic Math Vector Search
    if encoder is not None and db_embeddings is not None:
        try:
            query_embedding = encoder.encode(user_input, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, db_embeddings)
            best_match_idx = int(np.argmax(cos_scores.cpu().numpy()))
            matched_fact = FARM_KNOWLEDGE_BASE[best_match_idx]
        except Exception:
            pass

    # 2. Enforce English Response Prompt Generation on Qwen Core 
    system_instruction = (
        "You are an expert African agricultural advisor."
        "CRITICAL: Use the provided Factsheet Context to answer the user's question accurately."
        "Elaborate on the details to sound friendly and encouraging, but your facts MUST stay completely anchored to the factsheet context."
        "Do NOT invent unrelated facts, and write ONLY in clear English text without complex symbols."
    )
        
    prompt = (
        f"<|im_start|>system\n{system_instruction}\nFactsheet Context: {matched_fact}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    # Generate coherent core answer text using English token layout pathing
    if LLAMA_AVAILABLE and llm is not None:
        try:
            response = llm(
                prompt,
                max_tokens=200,
                temperature=0.1,  # Kept minimal to focus factual extraction explicitly
                stop=["<|im_end|>", "<|im_start|>", "User:", "System:"],
                echo=False
            )
            ai_response = response['choices'][0]['text'].strip()
            ai_response = re.sub(r'[\u4e00-\u9fff]+', '', ai_response)  # Erase unexpected artifacts
        except Exception:
            ai_response = matched_fact
    else:
        ai_response = matched_fact

    # 3. Mandated Exclusive Hugging Face Hausa Translation Loop (No String Fallbacks)
    if lang == "Hausa":
        if hf_tokenizer is not None and hf_model is not None:
            try:
                # Prepare text sequence arrays natively
                inputs = hf_tokenizer(ai_response, return_tensors="pt", padding=True)
                
                # Tag target tracking registers directly for Hausa latin matrices
                hf_tokenizer.src_lang = "eng_Latn"
                hf_tokenizer.tgt_lang = "hau_Latn"
                hausa_token_id = hf_tokenizer.convert_tokens_to_ids(hf_tokenizer.tgt_lang)
                
                # Execute strict sequence-to-sequence translation processing
                outputs = hf_model.generate(
                    **inputs,
                    forced_bos_token_id=hausa_token_id,
                    max_length=256,
                    num_beams=4
                )
                
                # Decode the tensor arrays exclusively into translated Hausa text
                ai_response = hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                return f"**Translation Error Code:** Exception inside HuggingFace model engine loop execution: {str(e)}"
        else:
            return "**System Error:** HuggingFace translator environment tools are uninitialized or missing dependencies."

    return f"{ai_response}{cultural_closing}"
    
# =========================================================
# TRANSLATION DICTIONARIES
# =========================================================
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

# =========================================================
# TIMELINE AND FINANCIAL LEDGER PARSERS
# =========================================================
def calculate_crop_timeline(crop, start_date):
    if crop == "Maize":
        fert1 = start_date + datetime.timedelta(days=21)
        fert2 = start_date + datetime.timedelta(days=42)
        harvest_start = start_date + datetime.timedelta(days=90)
        harvest_end = start_date + datetime.timedelta(days=120)
        return f"🌽 Maize Timeline:\n- Apply NPK: {fert1}\n- Apply Urea: {fert2}\n- Harvest Windows: {harvest_start} to {harvest_end}"
    else:
        fert1 = start_date + datetime.timedelta(days=30)
        fert2 = start_date + datetime.timedelta(days=90)
        harvest_start = start_date + datetime.timedelta(days=270)
        return f"🌿 Cassava Timeline:\n- Weed/Fertilizer 1: {fert1}\n- Fertilizer 2: {fert2}\n- Ready to Harvest Around: {harvest_start}"

def parse_financial_statement(statement):
    """
    Parses numeric text and records transactions contextually.
    """
    stmt_lower = statement.lower()
    
    # Extract numbers from text statement
    numbers = [float(s) for s in re.findall(r'\b\d+\b', statement)]
    amount = numbers[0] if numbers else 0.0
    
    sales_keywords = ["sell", "sold", "sayar", "revenue", "sales", "riba"]
    labour_keywords = ["labour", "worker", "lebur", "pay", "biya"]
    fert_keywords = ["fertilizer", "chemical", "taki", "seed", "iri"]
    equip_keywords = ["equipment", "tractor", "rental", "kayan", "traktò"]

    if any(x in stmt_lower for x in sales_keywords):
        st.session_state.revenue += amount
        return f"Logged Cost of Sale (Revenue): +{amount:,.2f} Naira"
    elif any(x in stmt_lower for x in labour_keywords):
        st.session_state.labour_cost += amount
        return f"Logged Labour Cost: -{amount:,.2f} Naira"
    elif any(x in stmt_lower for x in fert_keywords):
        st.session_state.fertilizer_cost += amount
        return f"Logged Fertilizer/Input Cost: -{amount:,.2f} Naira"
    elif any(x in stmt_lower for x in equip_keywords):
        st.session_state.equipment_cost += amount
        return f"Logged Equipment Cost: -{amount:,.2f} Naira"
    else:
        st.session_state.other_expenses += amount
        return f"Logged Miscellaneous Expense: -{amount:,.2f} Naira"

# =========================================================
# STREAMLIT GRAPHICAL INTERFACE
# =========================================================
st.set_page_config(page_title="SmartFarmAssistant", layout="wide")

if llm is None:
    st.warning("Application running in dummy mode. AI vector features require active weights storage paths.")
else:
    st.success("AI Core and Semantic Vector Engine loaded successfully in offline mode!")

col_lang, col_prov = st.columns(2)
with col_lang:
    selected_lang = st.selectbox("Language / Yare", ["English", "Hausa"])

labels = LANG_DICT[selected_lang]

with col_prov:
    prov_idx = int(time.time() // 10) % len(CULTURAL_PROVERBS)
    st.info(f"**{labels['proverb_title']}**\n{CULTURAL_PROVERBS[prov_idx]}")

st.title(labels["title"])
st.subheader(labels["subtitle"])

tab1, tab2, tab3 = st.tabs([
    labels.get("diagnose_tab", "AI Advisor"),
    labels.get("calendar_tab", "Timeline Calculator"),
    labels.get("finance_tab", "Financial Ledger")
])

# --- TAB 1: AI ADVISOR & SYMPTOM INPUTS ---
with tab1:
    text_key = f"text_symptom_{st.session_state.get('input_counter', 0)}"
    audio_key = f"audio_symptom_{st.session_state.get('input_counter', 0)}"
    user_text = st.text_input(labels.get("text_input_label", "Describe crop symptoms:"), key=text_key)
    
    col_aud1, col_aud2 = st.columns(2)
    with col_aud1:
        user_audio = st.audio_input("Record audio symptoms / Rikodin sauti:", key=audio_key)
    with col_aud2:
        uploaded_audio = st.file_uploader(
            "Upload audio file / Dorawa sauti:",
            type=["wav", "mp3", "m4a", "ogg"],
            key=f"audio_file_uploader_{st.session_state.get('input_counter', 0)}"
        )
        
    if uploaded_audio is not None and user_audio is None:
        user_audio = uploaded_audio
        
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
            st.rerun()

# --- TAB 2: TIMELINE CALCULATOR ---
with tab2:
    selected_crop = st.selectbox(labels["crop_select"], ["Maize", "Cassava"])
    planting_date = st.date_input(labels["date_input"], datetime.date.today())
    if st.button(labels["calc_btn"]):
        timeline_results = calculate_crop_timeline(selected_crop, planting_date)
        st.text(timeline_results)

# --- TAB 3: FINANCIAL LEDGER ---
with tab3:
    st.markdown("### Enter New Transactions / Shigar da Kudi")
    nlp_statement = st.text_input(labels["ledger_input"], key=f"nlp_stmt_{st.session_state.get('input_counter', 0)}")
    
    if st.button(labels["log_btn"]):
        if nlp_statement:
            parse_result = parse_financial_statement(nlp_statement)
            st.info(parse_result)
            st.rerun()
            
    st.markdown("---")
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        sale_input = st.number_input("Crop Sales Revenue (Naira):", min_value=0.0, step=500.0, key="sale_in")
        if st.button("Add to Sales / Kara Kudin Sayarwa"):
            st.session_state.revenue += sale_input
            st.success(f"Added +{sale_input:,.2f} Naira to Sales!")
            st.rerun()
            
        labour_input = st.number_input("Labour & Worker Cost (Naira):", min_value=0.0, step=500.0, key="labour_in")
        if st.button("Add to Labour / Kara Kudin Lebur"):
            st.session_state.labour_cost += labour_input
            st.success(f"Added -{labour_input:,.2f} Naira to Labour!")
            st.rerun()
            
    with col_in2:
        fert_input = st.number_input("Fertilizer & Chemicals Cost (Naira):", min_value=0.0, step=500.0, key="fert_in")
        if st.button("Add to Fertilizer / Kara Kudin Taki"):
            st.session_state.fertilizer_cost += fert_input
            st.success(f"Added -{fert_input:,.2f} Naira to Fertilizer!")
            st.rerun()
            
        equip_input = st.number_input("Equipment & Tractor Rental (Naira):", min_value=0.0, step=500.0, key="equip_in")
        if st.button("Add to Equipment / Kara Kudin Kayan Aiki"):
            st.session_state.equipment_cost += equip_input
            st.success(f"Added -{equip_input:,.2f} Naira to Equipment!")
            st.rerun()

    st.markdown("---")
    st.markdown("### Farm Profit & Loss Summary / Bayanin Riba da Asara")
    total_costs = (
        st.session_state.labour_cost +
        st.session_state.fertilizer_cost +
        st.session_state.equipment_cost +
        st.session_state.other_expenses
    )
    net_profit = st.session_state.revenue - total_costs
    
    st.metric("Total Sales Revenue / Kudin Sayarwa (+)", f"{st.session_state.revenue:,.2f} Naira")
    col_metrics1, col_metrics2 = st.columns(2)
    with col_metrics1:
        st.metric("Labour Costs / Kudin Lebur (-)", f"{st.session_state.labour_cost:,.2f} Naira")
        st.metric("Fertilizer & Chemicals / Kudin Taki (-)", f"{st.session_state.fertilizer_cost:,.2f} Naira")
    with col_metrics2:
        st.metric("Equipment & Tractor / Kayan Aiki (-)", f"{st.session_state.equipment_cost:,.2f} Naira")
        st.metric("Other Expenses / Kudaden Fitarwa (-)", f"{st.session_state.other_expenses:,.2f} Naira")
        
    st.markdown("---")
    if net_profit >= 0:
        st.success(f"**Net Profit / Riba Ta Tabbata:** {net_profit:,.2f} Naira")
    else:
        st.error(f"**Net Operating Loss / Asara Ta Fito:** {abs(net_profit):,.2f} Naira")
        
    if st.button("Reset Ledger / Goge Dukan Bayanan Kudi", type="secondary"):
        st.session_state.revenue = 0.0
    
    st.markdown("---")
    st.markdown("### Farm Profit & Loss Summary / Bayanin Riba da Asara")
    total_costs = (
        st.session_state.labour_cost +
        st.session_state.fertilizer_cost +
        st.session_state.equipment_cost +
        st.session_state.other_expenses
    )
    net_profit = st.session_state.revenue - total_costs
    
    st.metric("Total Sales Revenue / Kudin Sayarwa (+)", f"{st.session_state.revenue:,.2f} Naira")
    col_metrics1, col_metrics2 = st.columns(2)
    with col_metrics1:
        st.metric("Labour Costs / Kudin Lebur (-)", f"{st.session_state.labour_cost:,.2f} Naira")
        st.metric("Fertilizer & Chemicals / Kudin Taki (-)", f"{st.session_state.fertilizer_cost:,.2f} Naira")
    with col_metrics2:
        st.metric("Equipment & Tractor / Kayan Aiki (-)", f"{st.session_state.equipment_cost:,.2f} Naira")
        st.metric("Other Expenses / Kudaden Fitarwa (-)", f"{st.session_state.other_expenses:,.2f} Naira")
        
    st.markdown("---")
    if net_profit >= 0:
        st.success(f"**Net Profit / Riba Ta Tabbata:** {net_profit:,.2f} Naira")
    else:
        st.error(f"**Net Operating Loss / Asara Ta Fito:** {abs(net_profit):,.2f} Naira")
        
    if st.button("Reset Ledger / Goge Dukan Bayanan Kudi", type="secondary"):
        st.session_state.revenue = 0.0
        st.session_state.labour_cost = 0.0
        st.session_state.fertilizer_cost = 0.0
        st.session_state.equipment_cost = 0.0
        st.session_state.other_expenses = 0.0
        st.success("Ledger cleared successfully!")
        st.rerun()
        
    st.subheader("Save Records Locally")
    current_ledger_data = {
        "Revenue": [st.session_state.get('revenue', 0.0)],
        "LabourCost": [st.session_state.get('labour_cost', 0.0)],
        "FertilizerCost": [st.session_state.get('fertilizer_cost', 0.0)],
        "EquipmentCost": [st.session_state.get('equipment_cost', 0.0)],
        "OtherExpenses": [st.session_state.get('other_expenses', 0.0)]
    }
    
    if st.button("Save Ledger to Laptop", key="save_ledger_tab3_btn"):
        try:
            import pandas as pd
            df = pd.DataFrame(current_ledger_data)
            file_name = "ledger_backup.csv"
            df.to_csv(file_name, index=False)
            absolute_path = os.path.abspath(file_name)
            st.success(f"Saved successfully to your laptop at:\n`{absolute_path}`")
        except Exception as e:
            st.error(f"Failed to save: {e}")
            
    st.markdown("---")
    st.subheader("Download Ledger File")
    st.write("Download the current ledger data directly through your web browser.")
    try:
        import pandas as pd
        df = pd.DataFrame(current_ledger_data)
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇ Download Ledger as CSV",
            data=csv_data,
            file_name="ledger_download.csv",
            mime="text/csv",
            key="download_ledger_tab3_btn"
        )
    except Exception as download_error:
        st.info("Please fill in or save your ledger data above to enable downloading.")
