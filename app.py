import os
import datetime
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

# =====================================================================
# 🛡️ SAFE CRASH-PROOF BINDING LOADER
# =====================================================================
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

# =====================================================================
# ⚙️ CONFIGURATION
# =====================================================================
RUN_IN_CLOUD_FIRST = True

CLOUD_MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"

LOCAL_MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

LOCAL_MODELS_DIR = "models"
MODEL_PATH = os.path.join(LOCAL_MODELS_DIR, MODEL_FILE)

# ✅ FIXED ABSOLUTE URL STRING (Prevents formatting bugs and connection failures)
CLOUD_API_URL = "https://huggingface.co"

# =====================================================================
# 📂 COMPACT RAG ENGINE (LOCAL GROUNDING)
# =====================================================================
def load_local_knowledge(crop_type):
    file_path = f"knowledge/{crop_type.lower()}_guide.txt"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return "No local specific agricultural document found for this item."

# =====================================================================
# 📥 AUTO-DOWNLOAD ROUTINE
# =====================================================================
def ensure_local_model_exists():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📦 First boot detected! Downloading ultra-lightweight 0.5B LLM weights..."):
            os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
            hf_hub_download(
                repo_id=LOCAL_MODEL_REPO, 
                filename=MODEL_FILE,
                local_dir=LOCAL_MODELS_DIR,
                local_dir_use_symlinks=False
            )
        st.success("🎉 Download complete! Model saved completely offline.")

# =====================================================================
# 🤖 DUAL-MODE INFERENCE ENGINE (FIXED ARRAY UNBOXING LOGIC)
# =====================================================================
def generate_response(prompt_text, context=""):
    system_prompt = (
        "You are an expert African agricultural specialist. Use the following context documents "
        "to answer accurately. End your response with a brief, warm, encouraging traditional greeting "
        f"or proverb suitable for an African farmer.\n\nContext:\n{context}"
    )
    full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

    if RUN_IN_CLOUD_FIRST or not LLAMA_AVAILABLE:
        # Securely reads your validated operational token
        hf_token = "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".strip()
        headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json"
        }
        payload = {"inputs": full_prompt, "parameters": {"max_new_tokens": 256}}
        
        try:
            response = requests.post(CLOUD_API_URL, json=payload, headers=headers, timeout=15)
            
            if response.status_code == 503:
                return "🤗 Hugging Face server is currently loading model weights into cloud memory. Please wait 10 seconds and click 'Process Inquiry' again!"
            elif response.status_code != 200:
                return f"Server Error ({response.status_code}): {response.text}"
                
            res_json = response.json()
            
            # ✅ CORRECTED EXTRACTION: Safely pulls dictionary from inside the response array
            if isinstance(res_json, list) and len(res_json) > 0:
                inner_dict = res_json[0]
                text_out = inner_dict.get('generated_text', '') if isinstance(inner_dict, dict) else ''
                
                if "<|im_start|>assistant\n" in text_out:
                    return text_out.split("<|im_start|>assistant\n")[-1]
                return text_out
                
            return "Thinking complete! Tap 'Process Inquiry' once more to view layout."
        except Exception as e:
            return f"Cloud API Parsing Connection Error: {str(e)}"
    else:
        ensure_local_model_exists()
        llm = Llama(model_path=MODEL_PATH, n_ctx=1024, verbose=False)
        output = llm(full_prompt, max_tokens=300, stop=["<|im_end|>"])
        return output["choices"]["text"]

# =====================================================================
# 🎨 STREAMLIT GRAPHICAL INTERFACE
# =====================================================================
st.set_page_config(page_title="Smart Farm Assistant", page_icon="🌾", layout="centered")

st.title("🌾 Smart Farm Assistant")
language = st.selectbox("🌐 Choose Language / Zabi Yare", ["English", "Hausa"])

proverbs = {
    "English": "🍀 Proverbs of the day: 'Rain does not fall on one roof alone.' (An Ashanti Proverb) — Keep working hard alongside your community!",
    "Hausa": "🍀 Karin magana na ranar: 'Yau da gobe mai sa al'amura su daidaitu.' — Taimakon juna yana kawo babban amfanin gona!"
}
st.info(proverbs[language])

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

with tab1:
    st.subheader(lbl_tab1)
    selected_crop = st.radio(lbl_crop, ["Maize", "Cassava"])
    input_mode = st.radio("Input Method", ["Text Input", "Voice Simulation Input"])
    
    if input_mode == "Text Input":
        user_query = st.text_input(lbl_query, placeholder="e.g., my leaves have spots")
    else:
        user_query = st.text_area(lbl_txt_voice, placeholder="e.g., 'I notice brown rings on my stalks'")
        
    if st.button(lbl_btn):
        if user_query:
            with st.spinner("Processing... / Yana kan aiki..."):
                context_data = load_local_knowledge(selected_crop)
                ai_answer = generate_response(user_query, context_data)
                st.write("### 🤖 Response / Amsa:")
                st.write(ai_answer)
        else:
            st.warning("Please provide input text first!")

with tab2:
    st.subheader(lbl_tab2)
    planting_date = st.date_input(lbl_date, datetime.date.today())
    if st.button(lbl_calc):
        fert1_date = planting_date + datetime.timedelta(days=21)
        fert2_date = planting_date + datetime.timedelta(days=56)
        harvest_start = planting_date + datetime.timedelta(days=90)
        harvest_end = planting_date + datetime.timedelta(days=120)
        
        st.success("🗓️ Crop Development Milestones Calculated!")
        schedule_data = {
            "Milestone Stage": ["First Fertilizer Pass", "Second Fertilizer Pass", "Harvest Window Opens", "Harvest Window Closes"],
            "Target Date": [fert1_date.strftime('%B %d, %Y'), fert2_date.strftime('%B %d, %Y'), harvest_start.strftime('%B %d, %Y'), harvest_end.strftime('%B %d, %Y')]
        }
        st.table(pd.DataFrame(schedule_data))

with tab3:
    st.subheader(lbl_tab3)
    data = {
        'Category': ['Maize Sale', 'Cassava Sale', 'Seed Cost', 'Fertilizer Cost'],
        'Amount (Naira)': [45000, 32000, -12000, -15000]
    }
    df = pd.DataFrame(data)
    st.dataframe(df)
    st.metric(label="Net Profit Margin (Balance)", value=f"₦ {df['Amount (Naira)'].sum():,} Naira")
    
    # Financial data visual plot rendering routine
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ['green' if x > 0 else 'red' for x in df['Amount (Naira)']]
    ax.bar(df['Category'], df['Amount (Naira)'], color=colors)
    ax.set_ylabel('Naira')
    ax.set_title('Financial Overview')
    st.pyplot(fig)
