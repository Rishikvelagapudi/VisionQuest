"""
Streamlit Native UI for VECTOR: Voice-Enabled Indic RAG.
Optimized for instant Streamlit Cloud health check (<1s boot time).
"""
import asyncio
import json
import os
import tempfile
import streamlit as st

# Disable heavy network/ONNX downloads at boot to prevent Streamlit health check timeout
os.environ["ENABLE_PROMPT_GUARD"] = "false"
os.environ["ENABLE_CONTEXT_CHUNK_SCAN"] = "false"

# Page setup
st.set_page_config(
    page_title="⚡ VECTOR — Voice Indic RAG",
    page_icon="⚡",
    layout="wide",
)

# Custom retro-tropical styling
st.markdown("""
<style>
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .main-title {
        font-family: monospace;
        font-size: 2.2rem;
        font-weight: bold;
        color: #38bdf8;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚡ VECTOR Command Center</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Voice-Enabled Indic Multilingual Retrieval-Augmented Generation</div>', unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔍 Query Input")
    
    input_mode = st.radio("Select Input Mode", ["Text Query", "Audio File Upload"], horizontal=True)
    
    language_hint = st.selectbox(
        "Language Hint",
        options=["auto", "en", "hi", "mr"],
        format_func=lambda x: {"auto": "Auto-Detect", "en": "English (en)", "hi": "Hindi (hi)", "mr": "Marathi (mr)"}[x]
    )
    
    cross_lingual = st.checkbox("Enable Cross-Lingual Vector Search", value=False)
    bypass_cache = st.checkbox("Bypass Semantic Answer Cache", value=False)
    
    query_text = ""
    audio_path = None
    
    if input_mode == "Text Query":
        query_text = st.text_area(
            "Enter your question:",
            placeholder="e.g., What are the key features of the Manhattan Project?",
            height=120
        )
    else:
        uploaded_audio = st.file_uploader("Upload Audio File (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a"])
        if uploaded_audio:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_audio.read())
                audio_path = tmp.name
            st.audio(uploaded_audio)

    submit_btn = st.button("🚀 Run RAG Pipeline", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Output & JSON Telemetry")
    
    if submit_btn:
        if not query_text and not audio_path:
            st.warning("Please enter a text query or upload an audio file.")
        else:
            with st.spinner("Loading models & processing RAG pipeline..."):
                try:
                    from rag_pipeline.orchestrator import get_orchestrator
                    from rag_pipeline.schemas import QueryRequest
                    
                    orchestrator = get_orchestrator()
                    
                    req = QueryRequest(
                        text=query_text.strip() if query_text else None,
                        audio_path=audio_path,
                        language_hint=language_hint,
                        cross_lingual=cross_lingual,
                        bypass_cache=bypass_cache
                    )
                    
                    response = asyncio.run(orchestrator.execute(req))
                    
                    # Display Answer
                    st.markdown("### 💬 Answer")
                    if response.answer_source == "declined":
                        st.error(response.answer)
                    else:
                        st.success(response.answer)
                    
                    # Telemetry Metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Target Language", response.language_detected.upper())
                    m2.metric("Answer Source", response.answer_source)
                    m3.metric("Total Latency", f"{response.total_ms:.1f} ms")
                    
                    # Retrieved Chunks
                    with st.expander("📚 Retrieved Knowledge Chunks", expanded=True):
                        for i, chunk in enumerate(response.retrieved_chunks):
                            st.markdown(f"**Chunk #{i+1}** `[{chunk.source_lang.upper()}]` (Score: `{chunk.final_score:.4f}`)")
                            st.info(chunk.text)
                    
                    # Full JSON Response
                    with st.expander("🛠️ Full Raw JSON Response", expanded=True):
                        st.json(response.model_dump())
                        
                except Exception as ex:
                    st.error(f"Error executing pipeline: {ex}")
                finally:
                    if audio_path and os.path.exists(audio_path):
                        try:
                            os.remove(audio_path)
                        except Exception:
                            pass
