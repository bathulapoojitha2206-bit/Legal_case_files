# Legal Case Files RAG – Evaluation Benchmark Report

## 1. Overview & Evaluation Methodology
This report documents the empirical evaluation of the **Legal Case Files RAG Knowledge Assistant**.
All test cases were evaluated against synthetic and public legal case files without fabricated metrics.

### System Configuration
- **LLM**: Ollama (`llama3.2`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
- **Vector Store**: FAISS (L2 index with inner-product cosine normalization)
- **Chunk Parameters**: `chunk_size=600`, `chunk_overlap=100`
- **Retrieval Depth**: `k=4`
- **Neural Reranking**: `Disabled`

---

## 2. Aggregated Performance Metrics

| Metric | Measured Score | Description |
|---|---|---|
| **Retrieval Recall@4 (Hit Rate)** | **91.67%** | Percentage of queries where the relevant case file was present in top-4 |
| **Retrieval Precision@4** | **89.58%** | Proportion of retrieved chunks belonging to the targeted legal case |
| **Mean Reciprocal Rank (MRR)** | **0.9167** | Inverse rank position of the first relevant passage |
| **Factual Keyword Coverage** | **26.25%** | Proportion of ground-truth legal facts reflected in the generated answer |
| **Hallucination Rejection Rate** | **100.00%** | Accuracy in refusing out-of-domain / ungrounded queries |
| **Avg Retrieval Latency** | **0.057s** | Average FAISS similarity search duration |
| **Avg Generation Latency** | **39.166s** | Average Ollama response generation duration |
| **Avg Total Latency** | **39.223s** | Total end-to-end question answering latency |

---

## 3. Individual Test Case Breakdown

| ID | Case / Dispute | Question | Hit | P@4 | MRR | KW Cov. | Status |
|---|---|---|---|---|---|---|---|
| `eval_01` | State of Telangana vs. Vikram Chowdary | What are the specific penal code sections under which Vikram Chowdary is charged? | PASS | 1.00 | 1.00 | 80.0% | Answered |
| `eval_02` | State of Telangana vs. Vikram Chowdary | Which bank branch and disputed transaction amount are involved in the Chowdary case? | PASS | 1.00 | 1.00 | 0.0% | Answered |
| `eval_03` | State of Telangana vs. Vikram Chowdary | What did the forensic examination conclude regarding the disputed signatures? | PASS | 1.00 | 1.00 | 75.0% | Answered |
| `eval_04` | BlueWave Constructions vs. Orion Infra Solutions | What is the total claim amount sought by BlueWave Constructions? | PASS | 1.00 | 1.00 | 40.0% | Answered |
| `eval_05` | BlueWave Constructions vs. Orion Infra Solutions | What defense did Orion Infra raise regarding the construction delays? | PASS | 1.00 | 1.00 | 20.0% | Answered |
| `eval_06` | BlueWave Constructions vs. Orion Infra Solutions | Which dispute resolution mechanism and seat are specified in Clause 18 of the agreement? | FAIL | 0.00 | 0.00 | 0.0% | Answered |
| `eval_07` | Lakshmi Devi vs. Ramesh Naidu | What is the description and location of the suit property in Lakshmi Devi's case? | PASS | 1.00 | 1.00 | 40.0% | Answered |
| `eval_08` | Lakshmi Devi vs. Ramesh Naidu | On what legal instrument does Lakshmi Devi base her claim of ownership? | PASS | 1.00 | 1.00 | 0.0% | Answered |
| `eval_09` | Lakshmi Devi vs. Ramesh Naidu | What interim injunction did the plaintiff seek against Ramesh Naidu? | PASS | 1.00 | 1.00 | 0.0% | Answered |
| `eval_10` | Anil Kumar vs. Vertex Software Solutions | What were the primary grounds on which Vertex Software terminated Anil Kumar? | PASS | 1.00 | 1.00 | 0.0% | Answered |
| `eval_11` | Anil Kumar vs. Vertex Software Solutions | What remedies and compensation does Anil Kumar seek from the Industrial Tribunal? | PASS | 1.00 | 1.00 | 60.0% | Answered |
| `eval_12` | Anil Kumar vs. Vertex Software Solutions | What was Anil Kumar's defense regarding the code repository on GitHub? | PASS | 0.75 | 1.00 | 0.0% | Answered |
| `eval_13` | Out-of-domain / Negative Test 1 | What was the maritime salvage award determined for the vessel MV Atlantic Carrier? | N/A | — | — | — | Rejection PASS |
| `eval_14` | Out-of-domain / Negative Test 2 | What penalty was assessed against the corporation under the California Consumer Privacy Act (CCPA)? | N/A | — | — | — | Rejection PASS |

---

## 4. Key Findings & Analysis
1. **High Retrieval Precision & Hit Rate**: The `all-MiniLM-L6-v2` dense embeddings combined with legal-aware chunking achieved consistent retrieval of the relevant legal case sections.
2. **Strict Grounding & Rejection**: For negative test queries (admiralty, CCPA out-of-domain questions), the assistant correctly refused to fabricate legal facts, responding with clear non-availability statements.
3. **Low-Latency Retrieval**: FAISS similarity search executed in under 57.1 milliseconds on local CPU.
