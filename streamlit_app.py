"""
Streamlit Community Cloud entry point for VECTOR: Voice-Enabled Indic RAG.
Renders the full Command Center UI.
"""
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(
    page_title="⚡ VECTOR — Voice Indic RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Render full Command Center retro UI from web_ui/index.html
html_path = Path(__file__).parent / "web_ui" / "index.html"
if html_path.exists():
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=920, scrolling=True)
else:
    st.error("web_ui/index.html file not found.")
