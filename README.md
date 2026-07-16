# DeepGrep // Omnisearch Dashboard


🚀 **Live App Link:** [https://deepgrep.streamlit.app/](https://deepgrep.streamlit.app/)

DeepGrep is a local-first, polyglot semantic analysis and document exploration tool built using Streamlit. By combining dense deep learning vector embeddings with classical statistical keyword matching, it allows you to search across complex mixed-stack codebases and extensive project documentation entirely inside local RAM.

## 🚀 Key Architectural Pillars

* **Hybrid Search Topology:** Blends a dense vector matrix (FAISS + SentenceTransformers) for high-level conceptual intent exploration with a sparse keyword indexing engine (BM25 Okapi) for explicit code tokens, variables, and exact signature matches.
* **Balanced 50/50 Precision Layer:** Features a fine-tuned score normalization formula (50% conceptual intent + 50% absolute keyword match) to strictly surface target routines—like explicit entry declarations—while pruning unrelated structural drift.
* **Polyglot Structural Chunking:** Integrates an abstract syntax tree (AST) parser optimized for Python alongside a custom stateful lexical bracket scanner designed to isolate individual Java, C, C++, JavaScript, and TypeScript methods cleanly.
* **Omnisearch Document Pipeline:** Extends processing logic past traditional source code codebases to natively unbox, parse, and index text structures from target formats including **PDFs (`.pdf`), Word Documents (`.docx`), and Tabular Spreadsheets (`.csv`)**.
* **Local & Privacy-Centric:** Computes mathematical embeddings, multi-dimensional array comparisons, and data parsing loops completely inside your machine's memory cache, eliminating external third-party cloud data exposure.

---

## 📂 Project Workspace Topography

Based on your repository workspace configuration:
```text
DEEPGREP/
│
├── .venv/              # Isolated virtual environment container
├── .gitignore          # Repository git-ignore configuration tracker
├── app.py              # Main operational Streamlit dashboard script
├── requirements.txt    # Production cloud deployment dependencies
└── README.md           # Documentation asset

💻 Local Installation & Setup
1. Activate Your Virtual Environment Container:

.venv\Scripts\activate

2. Install Deep Learning & Core Framework Libraries:

pip install streamlit sentence-transformers faiss-cpu rank_bm25 pypdf python-docx numpy

3. Ignite the Application Dashboard:

streamlit run app.py


📊 Analytical Search Strategy Examples
1. Intent/Conceptual Search: Queries like "where is google app engine" or "how do we handle extreme data imbalance" process through the text encoder to fetch thematic paragraphs and page segments across PDFs or documentation assets, even if specific naming patterns differ.

2. Pinpoint Word Precision: Queries targeting definitive definitions like "main method declaration" utilize the balanced keyword layer to strictly capture matching code blocks inside source assets like App.java and push helper snippets down the rank list.
