"""
Streamlit App to render the full retro-tropical Command Center UI (web_ui/index.html)
with base64 embedded assets and backend FastAPI support.
"""
import base64
from pathlib import Path
import threading
import os
import uvicorn
import streamlit as st
import streamlit.components.v1 as components

# Streamlit Page Config
st.set_page_config(
    page_title="⚡ VECTOR 2026 — Voice Indic RAG Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit header/footer padding for full-screen UI experience
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 0rem;
        padding-right: 0rem;
        max-width: 100%;
    }
    iframe {
        border: none !important;
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper to read file as base64
def get_base64_image(file_path: Path) -> str:
    if file_path.exists():
        with open(file_path, "rb") as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
    return ""

# Load and prepare full HTML string
@st.cache_data
def get_prepared_html() -> str:
    web_ui_dir = Path(__file__).parent / "web_ui"
    index_file = web_ui_dir / "index.html"
    
    if not index_file.exists():
        return "<h1>Error: web_ui/index.html not found</h1>"
        
    with open(index_file, "r", encoding="utf-8") as f:
        html = f.read()
        
    # Replace relative PNG images with inline base64
    for img_name in ["ironman.png", "thor.png", "cap.png", "spider_gwen.png", "spider_gwen_clean.png", "spider_gwen_cutout.png", "bg_beach.png"]:
        b64_data = get_base64_image(web_ui_dir / img_name)
        if b64_data:
            html = html.replace(f'"{img_name}"', f'"{b64_data}"')
            html = html.replace(f"'{img_name}'", f"'{b64_data}'")
            html = html.replace(f'src="/{img_name}"', f'src="{b64_data}"')
            
    return html

# Start background FastAPI server for /query, /languages, /health endpoints
@st.cache_resource
def start_fastapi_backend():
    def _run():
        import main
        # Disable heavy startup preloads in background thread
        os.environ["ENABLE_PROMPT_GUARD"] = "false"
        uvicorn.run(main.app, host="0.0.0.0", port=7860, log_level="warning")
        
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return True

# Initialize backend
try:
    start_fastapi_backend()
except Exception:
    pass

# Render Full Command Center UI
prepared_html = get_prepared_html()
components.html(prepared_html, height=1200, scrolling=True)
