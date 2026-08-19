---
title: VECTOR - Voice Indic RAG Engine
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Voice Indic RAG with Sub-10ms FAISS Retrieval & 4-Tier Guardrails
---

# ⚡ VECTOR: Voice-Enabled Multilingual Indic RAG Engine

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FAISS Sub-10ms](https://img.shields.io/badge/FAISS-Sub--10ms_Latency-brightgreen?style=flat)](https://github.com/facebookresearch/faiss)
[![SLA Pass Rate](https://img.shields.io/badge/SLA_Pass_Rate-100%25-success?style=flat)](#2-cold-start-multilingual-sla-benchmark-15-languages)
[![Indic Languages](https://img.shields.io/badge/Languages-15_Indic_%2B_EN-blue?style=flat)](#-language-extensibility-matrix)
[![Docker SDK](https://img.shields.io/badge/Docker-HF_Spaces-2496ED?style=flat&logo=docker&logoColor=white)](#-hugging-face-space-deployment)

**An instrumented, ultra-low-latency, voice-enabled Retrieval-Augmented Generation (RAG) engine built from scratch for 14 Indic languages + English.**

[Features](#-key-features) • [Architecture](#-system-architecture) • [Language Matrix](#-language-extensibility-matrix) • [Benchmarks](#-benchmark-results) • [Deep-Dive](#-technical-deep-dive--engineering-rationales) • [Quickstart](#-quickstart--local-setup) • [Tests](#-test-suite--verification-5050-passing) • [Structure](#-repository-structure)

</div>

---

## 📌 Executive Summary

**VECTOR** is an open-source, high-throughput, sub-10ms Retrieval-Augmented Generation (RAG) engine engineered specifically for the linguistic diversity of the Indian subcontinent. Operating on low-cost CPU environments, VECTOR delivers end-to-end voice and text question answering across **14 Indic languages** (*Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Nepali, Odia, Punjabi, Sanskrit, Tamil, Telugu, Urdu*) plus **English** (15 languages total, **~743,000 deduplicated passages**).

The active runtime deployment loads **148,854 in-memory FAISS vectors** across 3 core active languages (**English [`en`]**, **Hindi [`hi`]**, and **Marathi [`mr`]**), achieving an average retrieval latency of **~7.04 ms** (p95: 7.97 ms vs 50.0 ms budget SLA).

### Key Architectural Advantages
- ⚡ **Sub-10ms Vector Retrieval**: In-memory FAISS HNSW graph traversal ($0.73\text{ ms}$) + INT8 ONNX vectorized embedding ($6.31\text{ ms}$) on CPU.
- 🛡️ **Cascaded 4-Tier Guardrails**: Stem regex with variable word-gap sliding, Meta Prompt-Guard 86M neural DPI/IPI shield, 6-class intent filter, and own-language centroid distance gate.
- 🔀 **Script-Aware BM25 + Dense Fusion**: Automatic cross-script detection bypassing lexical penalties for cross-lingual queries.
- 🧮 **Deterministic Context Synthesis**: TextRank graph centrality + SVD singular energy matrix reduction delivering factual answers in $<10\text{ ms}$ on CPU with zero LLM API cost or latency.
- 🌴 **Command Center UI**: Retro-tropical Web Audio frequency visualizer with real-time 9-stage telemetry waterfall breakdown.

---

## 🏛️ Horizontal Swimlane Architecture

```mermaid
graph LR
    %% Custom Styling Classes
    classDef inputStyle fill:#1E293B,stroke:#38BDF8,stroke-width:2px,color:#F8FAFC;
    classDef sttStyle fill:#0F172A,stroke:#818CF8,stroke-width:2px,color:#F8FAFC;
    classDef guardStyle fill:#311B92,stroke:#B388FF,stroke-width:2px,color:#FFFFFF;
    classDef cacheStyle fill:#064E3B,stroke:#34D399,stroke-width:2px,color:#F8FAFC;
    classDef faissStyle fill:#164E63,stroke:#22D3EE,stroke-width:2px,color:#F8FAFC;
    classDef synthStyle fill:#4C1D95,stroke:#C084FC,stroke-width:2px,color:#F8FAFC;
    classDef outStyle fill:#065F46,stroke:#10B981,stroke-width:2px,color:#FFFFFF;
    classDef blockStyle fill:#881337,stroke:#F43F5E,stroke-width:2px,color:#FFFFFF;

    subgraph PATH1 ["🎙️ Path 1: Audio & Text Ingestion"]
        A[Audio Upload / Microphone Stream] ::: inputStyle --> STT[Sarvam Saaras STT + ffmpeg 16kHz] ::: sttStyle
        T[Raw Text Input Bypass] ::: inputStyle --> ROUTER[Language Resolution Router] ::: sttStyle
        STT --> ROUTER
    end

    subgraph PATH2 ["🛡️ Path 2: 4-Tier Security Shield"]
        ROUTER --> G1[Tier-1 Stem Regex + Obfuscation Decoder] ::: guardStyle
        G1 -- Safe --> G2[Tier-2 Meta Prompt-Guard 86M DPI] ::: guardStyle
        G2 -- Safe --> G3[Tier-3 6-Class Query Intent Gate] ::: guardStyle
        G3 -- Factual --> G4[Tier-4 Own-Lang Centroid Distance Gate] ::: guardStyle
    end

    subgraph PATH3 ["⚡ Path 3: Sub-0.5ms Hot Cache Fast-Path"]
        G4 -- On-Topic --> CACHE{Hot Cache Lookup} ::: cacheStyle
        CACHE -- "Hit (<0.5ms)" --> FAST_OUT[Zero-Latency Response] ::: outStyle
    end

    subgraph PATH4 ["🔎 Path 4: Hybrid Vector Retrieval Engine"]
        CACHE -- "Miss" --> EMB[multilingual-e5-small INT8 ONNX] ::: faissStyle
        EMB --> FAISS[Parallel FAISS HNSW Native & LongDoc Search] ::: faissStyle
        FAISS --> RRF[Reciprocal Rank Fusion k=60] ::: faissStyle
        RRF --> BM25[Adaptive Script-Aware BM25 Fusion] ::: faissStyle
        BM25 --> GATE{Disqualification Gate} ::: faissStyle
    end

    subgraph PATH5 ["🧠 Path 5: Deterministic Synthesis & Grounding"]
        GATE -- High Relevance --> IPI[Batched Prompt-Guard IPI Context Scan] ::: synthStyle
        IPI -- Clean Chunks --> SYNTH[Continuous TextRank + SVD Energy Synthesis] ::: synthStyle
        SYNTH --> GROUND[Post-Gen Grounding Overlap Verifier] ::: synthStyle
        GROUND -- Grounded --> FINAL_OUT[JSON Response + 9-Stage Telemetry] ::: outStyle
    end

    %% Rejection Routing
    G1 -- Blocked --> REJECT[Declined Response: Safety Violation] ::: blockStyle
    G2 -- Injected --> REJECT
    G3 -- Non-Factual --> REJECT
    G4 -- Off-Topic --> REJECT
    GATE -- Score < 0.35 --> DECLINE[Declined Response: Insufficient Info] ::: blockStyle
    IPI -- Poisoned --> REJECT
    GROUND -- Ungrounded --> DECLINE
```

### 🛣️ Swimlane Pipeline Execution Breakdown

| Swimlane / Path | Key Components & Models | Latency Budget | Action on Failure / Edge Case |
| :--- | :--- | :---: | :--- |
| **🎙️ Path 1: Ingestion & STT** | Sarvam Saaras `saaras:v3` + `ffmpeg` 16kHz mono normalizer | `< 150 ms` (Audio) / `< 0.1 ms` (Text) | Fallback to default `language_hint` or auto-detect |
| **🛡️ Path 2: 4-Tier Security Shield** | Tier-1 Regex Stem Gap=4, Tier-2 Prompt-Guard 86M ONNX, Tier-3 6-Class Intent, Tier-4 Centroid Distance | `< 2.5 ms` | Fail-Safe-by-Category (`model_failed=True`), block immediately |
| **⚡ Path 3: Cache Fast-Path** | Gold QA Pairs + Dynamic In-Memory Vector LRU Cache ($N=2048$) | **`< 0.5 ms`** | Fallback to full retrieval pipeline on cache miss |
| **🔎 Path 4: Hybrid Search Engine** | `multilingual-e5-small` INT8 ONNX + FAISS HNSW ($M=32$) + RRF ($k=60$) + Script-Aware BM25 | **`< 8.0 ms`** | Candidate disqualification gate if composite score $< 0.35$ |
| **🧠 Path 5: Synthesis & Grounding** | Batched IPI Prompt-Guard + Continuous TextRank + SVD Singular Energy + Token Overlap | **`< 10.0 ms`** | Return standard non-hallucinating template on grounding fail |

---

## 🌐 Language Extensibility Matrix

VECTOR treats `config.LANGUAGES` as the single source of truth for active runtime languages. Adding or removing languages from active memory requires changing only this configuration list. The underlying processed dataset contains **~743,000 deduplicated passages** across all 15 languages:

| Code | Language | Script Family | Active Status | MS MARCO Dataset Source | Deduplicated Passages |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **`en`** | English | Latin (`Latn`) | 🟢 **Active Loaded** | MS MARCO English Stream | 49,507 |
| **`hi`** | Hindi | Devanagari (`Deva`) | 🟢 **Active Loaded** | `train/hintrain.parquet` & `val` | 49,509 |
| **`mr`** | Marathi | Devanagari (`Deva`) | 🟢 **Active Loaded** | `train/martrain.parquet` | 49,529 |
| **`as`** | Assamese | Bengali/Assamese (`Beng`) | 🟡 Zero-Code Extensible | `train/asmtrain.parquet` | 49,550 |
| **`bn`** | Bengali | Bengali (`Beng`) | 🟡 Zero-Code Extensible | `train/bentrain.parquet` | 49,531 |
| **`gu`** | Gujarati | Gujarati (`Gujr`) | 🟡 Zero-Code Extensible | `train/gujtrain.parquet` | 49,550 |
| **`kn`** | Kannada | Kannada (`Knda`) | 🟡 Zero-Code Extensible | `train/kantrain.parquet` | 49,545 |
| **`ml`** | Malayalam | Malayalam (`Mlym`) | 🟡 Zero-Code Extensible | `train/maltrain.parquet` | 49,542 |
| **`ne`** | Nepali | Devanagari (`Deva`) | 🟡 Zero-Code Extensible | `train/neptrain.parquet` | 49,520 |
| **`or`** | Odia | Odia (`Orya`) | 🟡 Zero-Code Extensible | `train/oritrain.parquet` | 49,560 |
| **`pa`** | Punjabi | Gurmukhi (`Guru`) | 🟡 Zero-Code Extensible | `train/pantrain.parquet` | 49,534 |
| **`sa`** | Sanskrit | Devanagari (`Deva`) | 🟡 Zero-Code Extensible | `validation/sanval.parquet` | 49,633 |
| **`ta`** | Tamil | Tamil (`Taml`) | 🟡 Zero-Code Extensible | `train/tamtrain.parquet` & `val` | 49,581 |
| **`te`** | Telugu | Telugu (`Telu`) | 🟡 Zero-Code Extensible | `validation/telval.parquet` | 49,604 |
| **`ur`** | Urdu | Perso-Arabic (`Arab`) | 🟡 Zero-Code Extensible | `validation/urdval.parquet` | 49,576 |

> [!NOTE]
> Active Loaded In-Memory Vectors: **148,545 Native Passages** + **309 LongDoc Chunks** = **148,854 In-Memory Vectors**. Total available corpus across all 15 languages: **~743,000 Passages**.

---

## ⚡ Benchmark Results

### 1. 🏎️ End-to-End Retrieval Latency (50ms Budget SLA)
*Benchmarked on CPU: 8 vCPUs | 15.78 GB RAM | Windows 11 AMD64*

| Pipeline Retrieval Stage | Avg Latency | P50 (Median) | P95 Latency | P99 Latency | Budget SLA | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Query Embedding (`multilingual-e5-small` ONNX)** | 6.31 ms | 6.21 ms | 7.28 ms | 7.94 ms | — | ⚡ Sub-8ms |
| **FAISS HNSW Search (`148,545 vectors`)** | 0.73 ms | 0.71 ms | 0.93 ms | 1.16 ms | — | ⚡ Sub-1ms |
| **Total Retrieval Latency (Embed + Search)** | **7.04 ms** | **6.96 ms** | **7.97 ms** | **8.95 ms** | **50.00 ms** | ✅ **PASS (84% Faster)** |

---

### 2. ❄️ Cold-Start Multilingual SLA Benchmark (15 Languages, Cache-Bypassed)
*Evaluates uncached performance across all 15 languages (`bypass_cache=True`). SLA Budget: `< 200 ms`.*

| Language | Code | Context Guard | Cross-Encoder Rerank | Generation | Total Cold Latency | SLA Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **English** | `en` | 0.99 ms | 53.40 ms | 0.92 ms | **120.48 ms** | ✅ **PASS** |
| **Hindi** | `hi` | 1.57 ms | 88.12 ms | 0.25 ms | **175.05 ms** | ✅ **PASS** |
| **Tamil** | `ta` | 1.35 ms | 79.50 ms | 0.19 ms | **173.49 ms** | ✅ **PASS** |
| **Telugu** | `te` | 1.21 ms | 90.72 ms | 0.18 ms | **164.77 ms** | ✅ **PASS** |
| **Bengali** | `bn` | 1.96 ms | 93.57 ms | 0.17 ms | **162.78 ms** | ✅ **PASS** |
| **Urdu** | `ur` | 1.58 ms | 103.12 ms | 0.21 ms | **169.99 ms** | ✅ **PASS** |
| **Marathi** | `mr` | 1.59 ms | 103.31 ms | 0.23 ms | **177.35 ms** | ✅ **PASS** |
| **Gujarati** | `gu` | 1.82 ms | 89.76 ms | 0.25 ms | **161.61 ms** | ✅ **PASS** |
| **Kannada** | `kn` | 1.82 ms | 76.86 ms | 0.27 ms | **167.62 ms** | ✅ **PASS** |
| **Malayalam** | `ml` | 1.82 ms | 83.41 ms | 0.16 ms | **170.34 ms** | ✅ **PASS** |
| **Punjabi** | `pa` | 2.21 ms | 100.67 ms | 0.19 ms | **181.81 ms** | ✅ **PASS** |
| **Assamese** | `as` | 1.83 ms | 110.64 ms | 0.23 ms | **194.22 ms** | ✅ **PASS** |
| **Odia** | `or` | 1.99 ms | 86.99 ms | 0.23 ms | **178.73 ms** | ✅ **PASS** |
| **Nepali** | `ne` | 1.31 ms | 109.84 ms | 0.23 ms | **184.45 ms** | ✅ **PASS** |
| **Sanskrit** | `sa` | 0.00 ms | 66.14 ms | 0.00 ms | **145.40 ms** | ✅ **PASS** |
| **Out-of-Domain Control** | `en` | 0.00 ms | 97.39 ms | 0.00 ms | **168.86 ms** | ✅ **PASS (Declined)** |
| **Safety Control** | `en` | 0.00 ms | 0.00 ms | 0.00 ms | **0.24 ms** | ✅ **PASS (Blocked)** |

---

### 3. 🚀 High-Throughput Speed Benchmark (750 Queries Total)
*Throughput: `51.7 Queries/sec` (14.50 seconds total benchmark runtime across 15 languages)*

| Pipeline Stage / Metric | P50 (Median) | P70 | P90 | P99 | Mean | Speedup Mechanism |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Query Embedding** | **15.18 ms** | 17.01 ms | 22.14 ms | 46.44 ms | 16.82 ms | ONNX Dynamic Shapes INT8 |
| **FAISS Graph Search** | **< 0.90 ms** | < 0.90 ms | < 0.90 ms | 0.91 ms | 0.86 ms | HNSW Index + search_k |
| **Cross-Encoder Rerank** | **26.70 ms** | 108.49 ms | 147.18 ms | 203.29 ms | 108.50 ms | ONNX MiniLM + Bounding |
| **Context Synthesis** | **8.50 ms** | 8.80 ms | 9.20 ms | 12.40 ms | 8.80 ms | TextRank + SVD Energy |
| **Cache Fast-Path** | **0.23 ms** | 0.28 ms | 0.35 ms | 0.70 ms | 0.35 ms | Dynamic LRU Vector Cache |
| **Full Pipeline Latency** | **16.45 ms** | **18.27 ms** | **23.78 ms** | **57.71 ms** | **19.22 ms** | ⚡ **Sub-20ms Median** |

---

## 🌟 Technical Deep-Dive & Engineering Rationales

<details>
<summary><b>1. 🛡️ Cascaded 4-Tier Pre-Retrieval Safety Guardrails</b></summary>

- **Tier-1: Stem + Flexible-Gap Regex (<0.1 ms)**: Replaces rigid phrase-literal matching with verb/object root stems and variable word-gap matching (`max_gap=4`). Handles gerunds (*"stealing"*, *"fabricating"*), irregular past conjugations (*"stole"*, *"hid"*), and unlisted adjectives across 15 languages.
- **Tier-2: Meta Prompt-Guard 86M Neural Safety (~1.5 ms)**: ONNX-accelerated Direct Prompt Injection (DPI) and Jailbreak classifier. Includes Unicode confusable unrolling and Base64 decoders. Unhandled exceptions strictly fail safe (`is_safe=False`, `model_failed=True`).
- **Tier-3: Pre-Retrieval Intent Filter**: 6-class intent taxonomy filtering creative writing, suggestion requests, personal advice, planning tasks, roleplay chat, and naming prompt categories before vector search.
- **Tier-4: Own-Language Centroid Gate**: Computes cosine distance from query embeddings to corpus centroids, requiring `own_lang_dist <= threshold * 1.5` to prevent cross-language cluster false positives.
</details>

<details>
<summary><b>2. ⚡ Script-Aware BM25 + FAISS Vector Search</b></summary>

- **In-Memory FAISS HNSW**: Built with $M=32$, $efConstruction=200$, $efSearch=64$, delivering 0.73ms CPU search over 148,545 passage vectors.
- **Script-Aware Score Fusion**: Monolingual queries combine BM25 + dense cosine similarity (`HYBRID_BM25_WEIGHT = 0.35`). Cross-script queries (e.g. English -> Hindi) automatically detect script mismatch and bypass BM25 lexical penalties.
- **Disqualification Gate**: Rejects candidate matches under score threshold 0.35 with standard non-hallucinating template.
</details>

<details>
<summary><b>3. 🧠 Deterministic TextRank + SVD Context Synthesis</b></summary>

- **Continuous TextRank Graph Centrality**: Computes sentence adjacency matrix $W_{ij} = \max(0, \vec{s}_i \cdot \vec{s}_j)$ with query relevance prior power iterations.
- **SVD Matrix Energy Filtering**: Retains principal components reaching $\ge 95\%$ cumulative singular energy to extract salient facts in $<10\text{ ms}$ on CPU with zero LLM API cost.
- **Swappable LLM / SLM Adapter**: Optional fallback to Groq / Cerebras APIs (`llama-3.3-70b-versatile`) or local Qwen SLMs.
</details>

---

## 🚀 Quickstart & Local Setup

### 1. Clone & Install
```bash
git clone https://github.com/Rishikvelagapudi/VECTOR.git
cd VECTOR
pip install -r requirements.txt
cp .env.example .env
```

### 2. Configure Environment Variables (`.env`)
```env
SARVAM_API_KEY=your_sarvam_api_key_here
LLM_API_KEY=your_groq_api_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile
```

### 3. Run Benchmark Suite
```bash
# 1. Run 50ms retrieval latency budget check
python -m app.benchmark 50

# 2. Run cold-start 15-language SLA benchmark
python benchmark/run_cold_start_bench.py

# 3. Run high-throughput 750-query speed benchmark
python benchmark/run_speed_bench_50.py
```

### 4. Launch REST API & Web Command Center UI
```bash
# Run FastAPI server directly
uvicorn api.main:app --host 0.0.0.0 --port 7860 --reload

# Or launch Gradio Space entrypoint:
python app.py
```
Open **[http://localhost:7860](http://localhost:7860)** in your browser.

---

## 🧪 Test Suite & Verification (50/50 Passing)

Run the full automated test suite:
```bash
pytest tests/ -v
```

| Test File | Count | Coverage & Scope |
| :--- | :---: | :--- |
| `tests/test_eval_fixes.py` | 17 | Adversarial safety, intent classification, Prompt-Guard fail-safe, centroid weighting |
| `tests/test_pipeline.py` | 27 | Passage/window/semantic chunking, BM25 script fusion, grounding overlap, 15-lang routing |
| `tests/test_prompt_guard.py` | 6 | DPI injection, IPI context filtering, confusable unpacker, sub-20ms latency check |

---

## 📁 Repository Structure

```
VECTOR/
├── api/                  # FastAPI web server (/query, /health, /languages)
├── app/                  # Fast ONNX retriever & 50ms benchmark runner
├── benchmark/            # Latency, cold-start, & throughput evaluation scripts
│   └── results/          # JSON & Markdown benchmark reports
├── chunking/             # Native, sentence-window, semantic, & RRF splitters
├── data/                 # FAISS HNSW indexes, centroids, JSONL corpora scripts
├── demo/                 # VECTOR Web Audio UI & visual assets
├── generation/           # TextRank + SVD non-LLM synthesis & LLM fallback
├── guardrails/           # 4-tier cascaded pre-retrieval & post-gen grounding
├── pipeline/             # Async 9-stage pipeline state machine & Pydantic schemas
├── retrieval/            # multilingual-e5-small INT8 ONNX & FAISS engine
├── stt/                  # Sarvam Saaras STT & ffmpeg 16kHz audio pipeline
├── tests/                # 50/50 unit & integration test suite
├── training/             # SFT dataset generator & Qwen Colab training notebooks
├── app.py                # VECTOR Space entrypoint application
├── config.py             # Single source of truth configuration
├── Dockerfile            # Container definition for Hugging Face Spaces
└── requirements.txt      # Python dependencies
```

---

## 🚀 Hugging Face Space Deployment

Deployed via **Docker SDK** on Hugging Face Spaces:
- **Live Space URL**: [https://ansh123456789-ragingoa.hf.space](https://ansh123456789-ragingoa.hf.space)

> [!TIP]
> Free `cpu-basic` Spaces sleep after 48h inactivity. Initial container spin-up takes **30–90s**. Warm runtime operates at **~7–16 ms**.

---

## 📜 License
MIT License. **VECTOR Multilingual Indic RAG Engine**.
