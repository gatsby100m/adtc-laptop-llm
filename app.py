import os
import datetime
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# =====================================================================
# ⚙️ CONFIGURATION & HACKATHON FLAGS
# =====================================================================
# Set to True to test quickly on the cloud. 
# Set to False for Devpost submission to download and run 100% offline!
RUN_IN_CLOUD_FIRST = True  

MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
LOCAL_MODELS_DIR = "models"
MODEL_PATH = os.path.join(LOCAL_MODELS_DIR, MODEL_FILE)

# Serverless inference URL routing matching standard open-source API specifications
CLOUD_API_URL = f"https://huggingface.co"

# =====================================================================
# 📂 COMPACT RAG ENGINE (LOCAL GROUNDING)
# =====================================================================
def load_local_knowledge(crop_type):
    """Reads context directly from local text files to prevent model hallucination."""
    file_path = f"knowledge/{crop_type.lower()}_guide.txt"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return "No local specific agricultural document found for this item."

# =====================================================================
# 📥 AUTO-DOWNLOAD ROUTINE
# =====================================================================
def ensure_local_model_exists():
    """Checks for local model file; downloads automatically if missing."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📦 First boot detected! Downloading ultra-lightweight 0.5B LLM weights..."):
            os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
            hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILE,
                local_dir=LOCAL_MODELS_DIR,
                local_dir_use_symlinks=False
            )
        st.success("🎉 Download complete! Model saved completely offline.")

# =====================================================================
# 🤖 DUAL-MODE INFERENCE ENGINE (WITH PROVERBS & CULTURAL WRAP)
# =====================================================================
def generate_response(prompt_text, context=""):
    """Executes prompt via cloud endpoints or local engine based on mode."""
    
    # Ground the prompt securely with our localized context files
    system_prompt = (
        "You are an expert African agricultural specialist. Use the following context documents "
        "to answer accurately. End your response with a brief, warm, encouraging traditional greeting "
        f"or proverb suitable for an African farmer.\n\nContext:\n{context}"
    )
    
    full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

    if RUN_IN_CLOUD_FIRST:
        # SECURE CHECK: Safely extract the secret from Streamlit Cloud dashboard settings
        if "HF_TOKEN" in st.secrets:
            hf_token = st.secrets["HF_TOKEN"]
        else:
            return "Configuration Error: 'HF_TOKEN' is missing in Streamlit Advanced Settings."

        headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json"
        }
        
        # Formatted using OpenAI/HF-compatible ChatCompletion schema for maximum system compatibility
        payload = {
            "model": MODEL_REPO.replace("-GGUF", ""), # Strip extension if needed by cloud serverless engine
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ],
            "max_tokens": 300
        }
        
        try:
            response = requests.post(CLOUD_API_URL, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            res_json = response.json()
            return res_json['choices'][0]['message']['content']
        except Exception as e:
            return f"Cloud API Connection Error: {str(e)}"
    else:
        ensure_local_model_exists()
        # Keep context footprint clamped to 1024 to respect memory limits
        llm = Llama(model_path=MODEL_PATH, n_ctx=1024, verbose=False)
        output = llm(
            full_prompt,
            max_tokens=300,
            stop=["<|im_end|>"]
        )
        return output["choices"]["text"]

# =====================================================================
# 🎨 STREAMLIT GRAPHICAL INTERFACE
# =====================================================================
st.set_page_config(page_title="Smart Farm Assistant", page_icon="🌾", layout="centered")

# --- 1. SYSTEM CONTROLS & LANGUAGE TOGGLE ---
st.title("🌾 Smart Farm Assistant")
language = st.selectbox("🌐 Choose Language / Zabi Yare", ["English", "Hausa"])

# Cultural Daily Motivating Proverbs Dictionary
proverbs = {
    "English": "🍀 Proverbs of the day: 'Rain does not fall on one roof alone.' (An Ashanti Proverb) — Keep working hard alongside your community!",
    "Hausa": "🍀 Karin magana na ranar: 'Yau da gobe mai sa al'amura su daidaitu.' — Taimakon juna yana kawo babban amfanin gona!"
}
st.info(proverbs[language])

# --- Translation Copy Framework ---
if language == "English":
    lbl_tab1, lbl_tab2, lbl_tab3 = "📋 Advisory Core", "📅 Crop Scheduler", "💰 Ledger Financials"
    lbl_query = "Ask your farming question or simulate voice text:"
    lbl_btn = "Process Inquiry"
    lbl_crop = "Select Targeted Crop Context"
    lbl_date = "Enter Planting Date"
    lbl_calc = "Calculate Milestones"
    lbl_txt_voice = "🎙️ Voice Input Simulation (Type what you would speak naturally into the app)"
else:
    lbl_tab1, lbl_tab2, lbl_tab3 = "📋 Shawarwari", "📅 Tsarin Lokaci", "💰 Lissafin Kudi"
    lbl_query = "Tambayi tambayar ku na harkar noma:"
    lbl_btn = "Sami Amsa"
    lbl_crop = "Zabi Amfanin Gona"
    lbl_date = "Shigar da Ranar Shuka"
    lbl_calc = "Kirga Ranaku"
    lbl_txt_voice = "🎙️ Kwaikwayon Muryar (Rubuta abinda zaka fada da baki)"

tab1, tab2, tab3 = st.tabs([lbl_tab1, lbl_tab2, lbl_tab3])

# =====================================================================
# 📋 TAB 1: LOCALIZED ADVISORY (RAG & VOICE SIMULATION)
# =====================================================================
with tab1:
    st.subheader(lbl_tab1)
    
    selected_crop = st.radio(lbl_crop, ["Maize", "Cassava"])
    input_mode = st.radio("Input Method", ["Text Input", "Voice Simulation Input"])
    
    if input_mode == "Text Input":
        user_query = st.text_input(lbl_query, placeholder="e.g., my leaves have spots")
    else:
        user_query = st.text_area(lbl_txt_voice, placeholder="e.g., 'I notice brown rings on my stalks' or 'I spent 5000 Naira on fertilizer'")
        
    if st.button(lbl_btn):
        if user_query:
            with st.spinner("Processing... / Yana kan aiki..."):
                # Pull local contextual records down to pass into LLM context window safely
                context_data = load_local_knowledge(selected_crop)
                ai_answer = generate_response(user_query, context_data)
                st.write("### 🤖 Response / Amsa:")
                st.write(ai_answer)
        else:
            st.warning("Please provide input text first!")

# =====================================================================
# 📅 TAB 2: MATHEMATICAL CROP MILESTONE TRACKING
# =====================================================================
with tab2:
    st.subheader(lbl_tab2)
    planting_date = st.date_input(lbl_date, datetime.date.today())
    
    if st.button(lbl_calc):
        # Precise calculations using clean Python datetime logic (zero RAM penalty)
        fert1_date = planting_date + datetime.timedelta(days=21)
        fert2_date = planting_date + datetime.timedelta(days=56)
        harvest_start = planting_date + datetime.timedelta(days=90)
        harvest_end = planting_date + datetime.timedelta(days=120)
        
        st.success("🗓️ Crop Development Milestones Calculated Successfully!")
        
        schedule_data = {
            "Milestone Stage": ["First Fertilizer Pass", "Second Fertilizer Pass", "Harvest Window Opens", "Harvest Window Closes"],
            "Target Date": [fert1_date.strftime('%B %d, %Y'), fert2_date.strftime('%B %d, %Y'), harvest_start.strftime('%B %d, %Y'), harvest_end.strftime('%B %d, %Y')]
        }
        df_schedule = pd.DataFrame(schedule_data)
        st.table(df_schedule)
        
        # Export functionality directly to standard clean system files
        report_text = f"Smart Farm Assistant - Schedule Report\nPlanting Date: {planting_date}\n" + df_schedule.to_string()
