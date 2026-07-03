# =====================================================================
# ⚙️ CONFIGURATION
# =====================================================================
RUN_IN_CLOUD_FIRST = True

CLOUD_MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"

LOCAL_MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

LOCAL_MODELS_DIR = "models"
MODEL_PATH = os.path.join(LOCAL_MODELS_DIR, MODEL_FILE)

# Correct Hugging Face Inference API endpoint
CLOUD_API_URL = f"https://api-inference.huggingface.co/models/{CLOUD_MODEL_REPO}"
