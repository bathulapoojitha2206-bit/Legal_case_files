"""
Legal Case Files - RAG Knowledge Assistant
Streamlit Web Application with Glassmorphic Legal Theme.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Optional

import streamlit as st

from rag_assistant.config import RAGConfig
from rag_assistant.pipeline import LegalRAGPipeline, PipelineResult
from rag_assistant.loader import DocumentLoader

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="Legal Case Files – RAG Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphic Dark Slate Design System
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --primary-accent: #38bdf8;
    --primary-glow: rgba(56, 189, 248, 0.25);
    --secondary-accent: #818cf8;
    --bg-dark: #0b0f19;
    --card-bg: rgba(17, 24, 39, 0.7);
    --card-border: rgba(255, 255, 255, 0.08);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --success-badge: #10b981;
    --warning-badge: #f59e0b;
    --danger-badge: #ef4444;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    color: var(--text-primary);
}

h1, h2, h3, h4, .stTitle {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

/* Glassmorphic Cards */
.glass-card {
    background: var(--card-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.glass-card:hover {
    border-color: rgba(56, 189, 248, 0.3);
}

/* Hero Banner */
.hero-container {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 50%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6);
    position: relative;
    overflow: hidden;
}

.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, var(--primary-glow) 0%, transparent 70%);
    pointer-events: none;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, #ffffff 0%, #bae6fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: var(--text-secondary);
    font-size: 1.05rem;
    line-height: 1.5;
    max-width: 800px;
}

/* Badges */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.status-online {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.status-offline {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.status-active-index {
    background: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
}

/* Citations & Source Excerpts */
.citation-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 600;
    font-size: 0.95rem;
    color: #bae6fd;
    margin-bottom: 0.4rem;
}

.score-badge {
    background: rgba(56, 189, 248, 0.12);
    color: #38bdf8;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.78rem;
    font-weight: 700;
}

.source-box {
    background: rgba(15, 23, 42, 0.6);
    border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 0.85rem 1.1rem;
    margin-top: 0.5rem;
    font-size: 0.92rem;
    line-height: 1.6;
    color: #cbd5e1;
}

/* Quick prompt pills */
.quick-prompt-btn {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.85rem;
    color: #94a3b8;
    cursor: pointer;
    transition: all 0.2s ease;
    text-align: left;
    margin-bottom: 0.5rem;
}

.quick-prompt-btn:hover {
    border-color: #38bdf8;
    color: #f8fafc;
    background: rgba(56, 189, 248, 0.1);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. APPLICATION STATE & PIPELINE SINGLETON
# ------------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_pipeline(
    model_name: str,
    retriever_k: int,
    use_reranker: bool,
    chunk_size: int,
    chunk_overlap: int
) -> LegalRAGPipeline:
    """Instantiate and cache the RAG pipeline."""
    config = RAGConfig(
        ollama_model=model_name,
        retriever_k=retriever_k,
        use_reranker=use_reranker,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    pipeline = LegalRAGPipeline(config=config)
    # Automatically load existing index if present on disk
    if pipeline.vector_store_manager.index_exists():
        try:
            pipeline.vector_store_manager.load_index()
        except Exception:
            pass
    return pipeline


# ------------------------------------------------------------------------------
# 3. SIDEBAR: CONTROLS & DIAGNOSTICS
# ------------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚖️ Legal RAG Control Center")

    # A. System Health Card
    temp_gen = LegalRAGPipeline().generator
    is_ollama_alive, ollama_msg, available_models = temp_gen.check_ollama_status()

    st.markdown("#### System Diagnostics")
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        if is_ollama_alive:
            st.markdown('<span class="status-pill status-online">● Ollama Online</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pill status-offline">● Ollama Offline</span>', unsafe_allow_html=True)

    # Temporary check for index
    index_check_mgr = LegalRAGPipeline().vector_store_manager
    has_index = index_check_mgr.index_exists()
    with col_stat2:
        if has_index:
            st.markdown('<span class="status-pill status-active-index">● Index Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pill status-offline">● No Index</span>', unsafe_allow_html=True)

    st.divider()

    # B. Model & Provider Settings
    st.markdown("#### LLM & Generation")
    default_models = ["llama3.2", "llama3.2:latest", "llama3", "mistral", "llava:7b"]
    selectable_models = sorted(list(set(available_models + default_models)))

    selected_model = st.selectbox(
        "Local Ollama Model",
        options=selectable_models,
        index=selectable_models.index("llama3.2") if "llama3.2" in selectable_models else 0,
        help="Select any local model pulled in Ollama."
    )

    retrieval_k = st.slider(
        "Retrieval Top-K Passages",
        min_value=1,
        max_value=8,
        value=4,
        help="Number of most relevant legal excerpts to retrieve."
    )

    use_reranker = st.toggle(
        "Neural Cross-Encoder Reranker",
        value=False,
        help="Enable second-stage neural reranking (ms-marco-MiniLM-L-6-v2) for refined relevance."
    )

    st.divider()

    # C. Ingestion Settings Expander
    with st.expander("⚙️ Chunking & Storage Parameters", expanded=False):
        chunk_size = st.number_input("Chunk Size (characters)", min_value=200, max_value=1500, value=600, step=50)
        chunk_overlap = st.number_input("Chunk Overlap (characters)", min_value=0, max_value=300, value=100, step=25)
        st.caption(f"Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)")
        st.caption(f"Vector Store: FAISS Index Flat/L2")

    st.divider()

    # D. Document Ingestion Card
    st.markdown("#### Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload Case Files (.pdf, .txt, .docx)",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Add custom case briefs, contracts, or court transcripts."
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        ingest_btn = st.button("📥 Index Uploads", use_container_width=True, type="primary")
    with col_btn2:
        index_samples_btn = st.button("📚 Load 4 Samples", use_container_width=True)


# Initialize active pipeline
pipeline = get_pipeline(
    model_name=selected_model,
    retriever_k=retrieval_k,
    use_reranker=use_reranker,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

# Handle Ingestion actions
if index_samples_btn:
    with st.spinner("Indexing synthetic sample case files..."):
        sample_dir = Path("data/sample_documents").resolve()
        report = pipeline.ingest_directory(sample_dir, save_index=True)
        st.success(
            f"Successfully indexed {report.total_chunks} chunks across {len(report.files_indexed)} sample files in {report.duration_seconds}s!"
        )
        st.rerun()

if ingest_btn:
    if not uploaded_files:
        st.warning("Please upload one or more legal case documents (.pdf, .txt, .docx) first.")
    else:
        with st.spinner(f"Ingesting and vectorizing {len(uploaded_files)} document(s)..."):
            extracted_docs = []
            for up_file in uploaded_files:
                file_bytes = up_file.read()
                res = pipeline.loader.load_bytes(file_bytes, up_file.name)
                if res.success and res.documents:
                    extracted_docs.extend(res.documents)
                else:
                    st.error(f"Failed to parse {up_file.name}: {res.error_message or res.warning_message}")

            if extracted_docs:
                chunks = pipeline.splitter.split_documents(extracted_docs)
                pipeline.vector_store_manager.add_documents(chunks)
                pipeline.vector_store_manager.save_index()
                st.success(f"Indexed {len(chunks)} chunks into FAISS vector store!")
                st.rerun()


# ------------------------------------------------------------------------------
# 4. MAIN WORKSPACE HEADER
# ------------------------------------------------------------------------------

st.markdown("""
<div class="hero-container">
    <div class="hero-title">Legal Case Files – RAG Knowledge Assistant</div>
    <div class="hero-subtitle">
        Enterprise-grade factual question answering for legal case documents. Grounded strictly in retrieved case transcripts, deeds, contracts, and court filings with zero hallucination.
    </div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 5. TABS INTERFACE
# ------------------------------------------------------------------------------

tab_qa, tab_browser, tab_eval, tab_arch = st.tabs([
    "💬 Case Intelligence & QA",
    "📑 Document Browser & Inspector",
    "📊 Benchmark Evaluation",
    "ℹ️ Architecture & Diagnostics"
])


# ------------------------------------------------------------------------------
# TAB 1: CASE QA & INQUIRY
# ------------------------------------------------------------------------------
with tab_qa:
    col_input, col_preset = st.columns([2.8, 1.2])

    with col_preset:
        st.markdown("##### Quick Legal Inquiries")
        preset_q1 = "What penal sections is Vikram Chowdary charged with?"
        preset_q2 = "What are the delay claims in BlueWave Constructions?"
        preset_q3 = "On what legal instrument does Lakshmi Devi base title?"
        preset_q4 = "What was the reason for Anil Kumar's termination?"

        q_select = st.radio(
            "Select sample question:",
            [
                "Custom Question",
                preset_q1,
                preset_q2,
                preset_q3,
                preset_q4
            ],
            index=0,
            label_visibility="collapsed"
        )

    with col_input:
        default_val = "" if q_select == "Custom Question" else q_select
        user_query = st.text_area(
            "Enter your legal question:",
            value=default_val,
            placeholder="e.g. Which court issued the injunction in the property dispute, and what were the terms?",
            height=95
        )
        ask_btn = st.button("⚖️ Analyze Case Files", type="primary", use_container_width=False)

    if ask_btn or (user_query and q_select != "Custom Question" and st.session_state.get("last_auto_q") != user_query):
        st.session_state["last_auto_q"] = user_query
        if not user_query.strip():
            st.warning("Please enter a legal query.")
        else:
            with st.spinner("Analyzing case filings and generating grounded answer..."):
                result: PipelineResult = pipeline.ask(user_query, top_k=retrieval_k)

            # Display Status Notification if non-success
            if result.status == "warning":
                st.warning(result.answer)
            elif result.status == "error":
                st.error(result.answer)
            else:
                # Main Answer Presentation
                st.markdown("### 📋 Case Analysis & Factual Findings")
                st.markdown(f"""
                <div class="glass-card">
                    <div style="font-size: 1.05rem; line-height: 1.7; color: #f1f5f9; white-space: pre-wrap;">
{result.answer}
                    </div>
                    <div style="margin-top: 1.2rem; display: flex; gap: 1rem; flex-wrap: wrap; font-size: 0.8rem; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 0.8rem;">
                        <span>🤖 <b>Model:</b> {result.model_used}</span>
                        <span>⚡ <b>Retrieval:</b> {result.retrieval_time_sec}s</span>
                        <span>🧠 <b>Generation:</b> {result.generation_time_sec}s</span>
                        <span>⏱️ <b>Total Latency:</b> {result.total_time_sec}s</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Sources & Evidence Citations
                st.markdown(f"### 📑 Grounding Sources & Passages ({len(result.passages)} retrieved)")
                for idx, passage in enumerate(result.passages, start=1):
                    page_label = f"Page {passage.page}" if passage.page else "Full Document"
                    chunk_label = f"Chunk #{passage.chunk_id}" if passage.chunk_id is not None else ""
                    expander_label = f"Source {idx}: {passage.filename}  ({page_label} | {chunk_label} | Match Score: {passage.score:.3f})"

                    with st.expander(expander_label, expanded=(idx == 1)):
                        st.markdown(f"""
                        <div class="source-box">
                            {passage.content}
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.78rem; color: #64748b;">
                            <b>Origin File Path:</b> {passage.source}
                        </div>
                        """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 2: DOCUMENT BROWSER & INSPECTOR
# ------------------------------------------------------------------------------
with tab_browser:
    st.markdown("### 📂 Case Document Repository")
    st.write("Browse existing synthetic case documents, preview contents, and inspect metadata.")

    sample_files = list(Path("data/sample_documents").glob("*.*"))
    if not sample_files:
        st.info("No documents found in `data/sample_documents`. Use the sidebar to load sample cases.")
    else:
        col_list, col_preview = st.columns([1, 2])
        with col_list:
            file_names = [f.name for f in sample_files if f.suffix in [".pdf", ".txt", ".docx"]]
            selected_file_name = st.radio("Select Document:", file_names)

        with col_preview:
            if selected_file_name:
                selected_path = Path("data/sample_documents") / selected_file_name
                st.markdown(f"#### Viewing: `{selected_file_name}`")
                st.caption(f"File Size: {selected_path.stat().st_size:,} bytes | Type: {selected_path.suffix.upper()}")

                loader = DocumentLoader()
                doc_res = loader.load_file(selected_path)
                if doc_res.success:
                    st.metric("Total Extracted Pages / Sections", len(doc_res.documents))
                    for i, page_doc in enumerate(doc_res.documents, start=1):
                        with st.expander(f"Section / Page {i}", expanded=(i == 1)):
                            st.text_area(
                                f"Content (Page {i})",
                                value=page_doc.page_content,
                                height=200,
                                key=f"preview_{selected_file_name}_{i}"
                            )
                else:
                    st.error(f"Could not load preview: {doc_res.error_message}")


# ------------------------------------------------------------------------------
# TAB 3: BENCHMARK & EVALUATION
# ------------------------------------------------------------------------------
with tab_eval:
    st.markdown("### 📊 Empirical RAG Evaluation Benchmark")
    st.write(
        "Evaluate retrieval accuracy (Recall@k, Precision@k, MRR), answer keyword grounding, and hallucination rejection."
    )

    eval_json_path = Path("evaluation_results.json")
    if eval_json_path.exists():
        with open(eval_json_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Recall@k (Hit Rate)", f"{eval_data.get('mean_recall_at_k', 0):.1%}")
        with m_col2:
            st.metric("Precision@k", f"{eval_data.get('mean_precision_at_k', 0):.1%}")
        with m_col3:
            st.metric("Mean Reciprocal Rank (MRR)", f"{eval_data.get('mean_reciprocal_rank', 0):.3f}")
        with m_col4:
            st.metric("Rejection Rate (Negative)", f"{eval_data.get('hallucination_rejection_accuracy', 0):.1%}")

        st.markdown("#### Test Case Breakdown")
        cases_list = eval_data.get("cases", [])
        if cases_list:
            table_rows = []
            for c in cases_list:
                table_rows.append({
                    "ID": c["id"],
                    "Case / Topic": c["case"],
                    "Question": c["question"][:55] + "..." if len(c["question"]) > 55 else c["question"],
                    "Hit (Recall)": "✅ Yes" if c["retrieval_hit"] else ("—" if c["is_negative_test"] else "❌ No"),
                    "Precision@k": f"{c['precision_at_k']:.2f}" if not c["is_negative_test"] else "—",
                    "KW Coverage": f"{c['keyword_coverage']:.0%}" if not c["is_negative_test"] else "—",
                    "Grounded Status": "Passed Rejection" if c["is_negative_test"] and c["correctly_rejected"] else "Answered"
                })
            st.dataframe(table_rows, use_container_width=True)
    else:
        st.info("Evaluation results have not yet been generated. Run `python evaluate.py` in the terminal to execute benchmarks.")


# ------------------------------------------------------------------------------
# TAB 4: ARCHITECTURE & DIAGNOSTICS
# ------------------------------------------------------------------------------
with tab_arch:
    st.markdown("### 🏗️ Technical Architecture & Workflow")

    st.markdown("""
```mermaid
graph TD
    A[Legal Case Documents: PDF / TXT / DOCX] --> B[DocumentLoader with Robust Fallbacks]
    B --> C[DocumentSplitter: Recursive 600-char chunks, 100-char overlap]
    C --> D[EmbeddingManager: sentence-transformers/all-MiniLM-L6-v2]
    D --> E[(FAISS Vector Store: Flat L2 / Cosine Norm)]
    
    Q[User Legal Question] --> F[LegalRetriever: Similarity Search Top-K]
    E --> F
    F --> G[Optional: CrossEncoder ms-marco Reranker]
    G --> H[Strict Grounded Legal Prompt Template]
    H --> I[Ollama Local LLM: llama3.2]
    I --> J[Grounded Factual Answer + Case File Citations]
```
    """, unsafe_allow_html=True)

    st.markdown("#### Deployment & Local Execution Guide")
    st.markdown("""
    - **Local Ollama Daemon**: Ensure Ollama is running via `ollama serve`.
    - **Required Model**: Pull the lightweight Llama 3.2 model: `ollama pull llama3.2`.
    - **Cloud Deployment Note**: Standard serverless cloud hosts do not provide GPU or local Ollama instances. For cloud hosting (e.g. Streamlit Cloud / HuggingFace Spaces), configure `LLM_PROVIDER=groq` or `openai` in `.env`.
    """)
