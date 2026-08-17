# DeepGrep // Omnisearch Dashboard


[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://rk0519-deepgrep-app-n12cfe.streamlit.app/)

DeepGrep is an advanced polyglot semantic analysis, document exploration, and AI RAG (Retrieval-Augmented Generation) assistant built using Streamlit. By combining dense deep learning vector embeddings, classical statistical keyword matching, and high-speed Groq inference, it allows you to search across and chat with complex mixed-stack codebases and extensive project documentation.

## 🚀 Key Architectural Pillars

* **Hybrid Search Topology:** Blends a dense vector matrix (FAISS + SentenceTransformers) for high-level conceptual intent exploration with a sparse keyword indexing engine (BM25 Okapi) for explicit code tokens, variables, and exact signature matches.
* **Balanced 50/50 Precision Layer:** Features a fine-tuned score normalization formula (50% conceptual intent + 50% absolute keyword match) to strictly surface target routines—like explicit entry declarations—while pruning unrelated structural drift.
* **Polyglot Structural Chunking:** Integrates an abstract syntax tree (AST) parser optimized for Python alongside a custom stateful lexical bracket scanner designed to isolate individual Java, C, C++, JavaScript, and TypeScript methods cleanly.
* **Omnisearch Document Pipeline:** Extends processing logic past traditional source code codebases to natively unbox, parse, and index text structures from target formats including **PDFs (`.pdf`), Word Documents (`.docx`), and Tabular Spreadsheets (`.csv`)**.
* **AI RAG Synthesis Engine:** Integrates high-speed Groq inference to dynamically synthesize retrieved workspace chunks into direct, context-aware answers right inside the dashboard interface.
* **Secure & Privacy-Centric:** Computes mathematical embeddings locally while handling deployment credentials and API keys safely behind the scenes via environment variables and secrets management.

---
## 📂 Project Workspace Topography

```text
DEEPGREP/
│
├── .venv/              # Isolated virtual environment container
├── .gitignore          # Repository git-ignore configuration tracker
├── app.py              # Main operational Streamlit dashboard script
├── requirements.txt    # Production cloud deployment dependencies
└── README.md           # Documentation asset
```
---

## 💻 Local Installation & Setup
1. Activate Your Virtual Environment Container:

```text
.venv\Scripts\activate
```

2. Install Deep Learning & Core Framework Libraries:

```text
pip install streamlit sentence-transformers faiss-cpu rank_bm25 pypdf python-docx groq numpy
```

3. Configure your Environment:
Set your Groq API key in your local environment variables:

```text
export GROQ_API_KEY="your-api-key-here"
```

4. Ignite the Application Dashboard:

```text
streamlit run app.py
```

---

## 📊 Analytical Search Strategy Examples
1. Intent/Conceptual Search: Queries like "where is google app engine" or "how do we handle extreme data imbalance" process through the text encoder to fetch thematic paragraphs and page segments across PDFs or documentation assets, even if specific naming patterns differ.

2. Pinpoint Word Precision: Queries targeting definitive definitions like "main method declaration" utilize the balanced keyword layer to strictly capture matching code blocks inside source assets like App.java and push helper snippets down the rank list.

3. AI RAG Synthesis: Instantly leverage the integrated Groq model to read your top-ranked context blocks and generate precise, conversational answers directly inside your workspace.
