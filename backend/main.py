"""
Hugging Face Space Application for VECTOR: Voice-Enabled Indic RAG.
Renders the full retro-tropical Command Center UI and exposes FastAPI endpoints.
ZeroGPU compatible.
"""

import asyncio
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

# Compatibility shim for older packages importing HfFolder from huggingface_hub
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "HfFolder"):
        class DummyHfFolder:
            @staticmethod
            def get_token():
                import os
                return os.environ.get("HF_TOKEN") or None
            @staticmethod
            def save_token(token):
                pass
            @staticmethod
            def delete_token():
                pass
        huggingface_hub.HfFolder = DummyHfFolder
except Exception:
    pass

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import gradio as gr
import uvicorn

# ZeroGPU decorator shim
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, **kwargs):
            if func is None:
                def decorator(f):
                    return f
                return decorator
            return func

import config
from rag_pipeline.orchestrator import get_orchestrator
from rag_pipeline.schemas import QueryRequest, QueryResponse
from vector_search.embed import get_embedder
from vector_search import get_index_manager


# Read the full custom HTML Command Center UI
def get_custom_html() -> str:
    demo_file = config.WEB_UI_DIR / "index.html"
    if demo_file.exists():
        with open(demo_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Voice-Enabled Multilingual Indic RAG</h1>"


@spaces.GPU
def _dummy_zerogpu():
    """ZeroGPU requirement: at least one function registered to event scan."""
    return True


# Create core FastAPI application
app = FastAPI(title="⚡ VECTOR — Voice-Enabled Indic RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Preload models and perform full pipeline warmup asynchronously at startup
@app.on_event("startup")
async def startup_event():
    if os.getenv("SKIP_WARMUP", "true").lower() == "true" or os.getenv("RENDER") or os.getenv("RENDER_SERVICE_ID"):
        print("[Startup] Cloud PaaS environment detected: Skipping eager warmup to preserve RAM under 150MB.")
        return
    async def run_warmup():
        print("[Space Startup] Preloading embedding model, FAISS indexes, and warming up pipeline...")
        orchestrator = get_orchestrator()
        await asyncio.to_thread(orchestrator.warmup_pipeline)
        print("[Space Startup] Full RAG pipeline preloaded and warmed up successfully.")
    
    asyncio.create_task(run_warmup())


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> HTMLResponse:
    """Serve full retro-tropical Command Center UI directly to browser."""
    demo_file = config.WEB_UI_DIR / "index.html"
    if demo_file.exists():
        with open(demo_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>VECTOR 2026 Command Center</h1>")


@app.get("/ironman.png")
async def serve_ironman() -> FileResponse:
    """Serve Iron Man sprite image."""
    img_path = config.WEB_UI_DIR / "ironman.png"
    if img_path.exists():
        return FileResponse(img_path)
    raise HTTPException(status_code=404, detail="Iron Man image missing")


@app.get("/thor.png")
async def serve_thor() -> FileResponse:
    """Serve Thor sprite image."""
    img_path = config.WEB_UI_DIR / "thor.png"
    if img_path.exists():
        return FileResponse(img_path)
    raise HTTPException(status_code=404, detail="Thor image missing")


@app.get("/cap.png")
async def serve_cap() -> FileResponse:
    """Serve Captain America sprite image."""
    img_path = config.WEB_UI_DIR / "cap.png"
    if img_path.exists():
        return FileResponse(img_path)
    raise HTTPException(status_code=404, detail="Captain America image missing")


@app.get("/health", response_class=JSONResponse)
async def health_check() -> Dict[str, Any]:
    """Health check reporting system and index readiness."""
    index_mgr = get_index_manager()
    index_stats = {
        name: idx.index.ntotal if hasattr(idx, "index") and idx.index else "qdrant_cloud" for name, idx in index_mgr.indexes.items()
    }
    return {
        "status": "healthy",
        "configured_languages": config.LANGUAGES,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "indexes_loaded": index_stats,
        "centroids_available": list(index_mgr.centroids.keys()),
        "sarvam_stt_configured": bool(config.SARVAM_API_KEY),
        "llm_fallback_configured": bool(config.LLM_API_KEY),
    }


@app.get("/languages", response_class=JSONResponse)
async def get_supported_languages() -> Dict[str, Any]:
    """Returns metadata for all currently configured active languages."""
    lang_details = [
        {"code": l, **config.get_language_info(l)} for l in config.LANGUAGES
    ]
    return {
        "active_languages": config.LANGUAGES,
        "language_details": lang_details,
    }


def _parse_bool(val: Any, default: bool = True) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "on")
    return bool(val)


@app.post("/query", response_model=QueryResponse)
async def query_pipeline(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    language_hint: Optional[str] = Form(None),
    cross_lingual: Optional[Any] = Form(None),
    bypass_cache: Optional[Any] = Form(None),
    request_body: Optional[QueryRequest] = None,
) -> QueryResponse:
    """
    Execute end-to-end Voice RAG query for the Command Center UI.
    """
    orchestrator = get_orchestrator()
    temp_audio_path = None
    is_cross_lingual = _parse_bool(cross_lingual, default=False)
    is_bypass_cache = _parse_bool(bypass_cache, default=False)
    
    try:
        if request_body and (request_body.text or request_body.audio_path):
            return await orchestrator.execute(request_body)
            
        if file and file.filename:
            suffix = Path(file.filename).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_audio_path = tmp.name
                
            req = QueryRequest(
                audio_path=temp_audio_path,
                language_hint=language_hint,
                cross_lingual=is_cross_lingual,
                bypass_cache=is_bypass_cache,
            )
            return await orchestrator.execute(req)
            
        if text and text.strip():
            req = QueryRequest(
                text=text.strip(),
                language_hint=language_hint,
                cross_lingual=is_cross_lingual,
                bypass_cache=is_bypass_cache,
            )
            return await orchestrator.execute(req)
            
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'file' audio upload or 'text' query must be provided.",
        )
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass


# Create Gradio interface block and mount on FastAPI app
with gr.Blocks(title="⚡ VECTOR — Voice Indic RAG") as demo:
    gr.HTML(get_custom_html())
    dummy_btn = gr.Button("zero_gpu_anchor", visible=False)
    dummy_btn.click(fn=_dummy_zerogpu)

app = gr.mount_gradio_app(app, demo, path="/gradio")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"[Space Startup] Starting VECTOR Command Center UI on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
