import streamlit as st
import ast
import re
import csv
import io
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# External document reader libraries
from pypdf import PdfReader
from docx import Document

# --- 1. SETUP & INTERFACE STYLING ---
st.set_page_config(
    page_title="DeepGrep // Omnisearch Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    code, pre, [data-testid="stCodeBlock"] {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00ffcc 0%, #0077ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #8892b0;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .sidebar-card {
        background-color: #11192e;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .match-tag {
        background-color: rgba(0, 255, 204, 0.1);
        color: #00ffcc;
        border: 1px solid rgba(0, 255, 204, 0.2);
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .idle-tag {
        color: #64748b;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. LOAD AI ENCODER ---
@st.cache_resource
def get_ai_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

ai_model = get_ai_model()

# --- 3. UNIFIED POLYGLOT & DOCUMENT PROCESSING UTILITY ---
def split_file_into_chunks(filename, raw_bytes):
    """
    Inspects extensions and strategically parses text contents from 
    both code assets and production document data channels.
    """
    chunks = []
    extension = filename.split('.')[-1].lower() if '.' in filename else ''

    # --- CATEGORY 1: DOCUMENT PROCESSING ---
    
    # A. PDF Ingestion
    if extension == 'pdf':
        try:
            pdf_stream = io.BytesIO(raw_bytes)
            reader = PdfReader(pdf_stream)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    chunks.append({
                        "filename": filename,
                        "type": f"PDF Page {page_num + 1}",
                        "line_number": page_num + 1,
                        "code": text.strip()
                    })
        except Exception as e:
            pass

    # B. Word Document Ingestion (.docx)
    elif extension == 'docx':
        try:
            docx_stream = io.BytesIO(raw_bytes)
            doc = Document(docx_stream)
            full_text = [para.text for para in doc.paragraphs if para.text.strip()]
            
            # Group text blocks every 8 paragraphs to preserve vector context
            step = 8
            for i in range(0, len(full_text), step):
                combined_block = "\n".join(full_text[i:i+step])
                chunks.append({
                    "filename": filename,
                    "type": "Word Document Segment",
                    "line_number": i + 1,
                    "code": combined_block
                })
        except Exception as e:
            pass

    # C. CSV Spreadsheet Ingestion
    elif extension == 'csv':
        try:
            decoded_text = raw_bytes.decode("utf-8", errors="ignore")
            csv_reader = csv.reader(io.StringIO(decoded_text))
            rows = list(csv_reader)
            
            # Format row matrices into clean descriptive sentence summaries
            step = 10
            for i in range(0, len(rows), step):
                slice_rows = rows[i:i+step]
                formatted_lines = [", ".join(row) for row in slice_rows if row]
                if formatted_lines:
                    chunks.append({
                        "filename": filename,
                        "type": f"CSV Rows {i+1}-{i+len(formatted_lines)}",
                        "line_number": i + 1,
                        "code": "\n".join(formatted_lines)
                    })
        except Exception as e:
            pass

    # --- CATEGORY 2: CODE & PLAIN TEXT PROCESSING ---
    else:
        try:
            raw_content = raw_bytes.decode("utf-8", errors="ignore")
            lines = raw_content.splitlines()

            # Python AST Parser
            if extension == 'py':
                try:
                    parsed_tree = ast.parse(raw_content)
                    for node in ast.walk(parsed_tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            start = node.lineno
                            end = max(getattr(n, 'lineno', start) for n in ast.walk(node))
                            code_block = "\n".join(lines[start-1:end])
                            
                            if len(code_block.strip()) > 2:
                                node_label = "Function" if isinstance(node, ast.FunctionDef) else "Class"
                                chunks.append({
                                    "filename": filename,
                                    "type": f"Python {node_label} ({node.name})",
                                    "line_number": start,
                                    "code": code_block
                                })
                    
                    top_level_lines = [l for l in lines if l.strip() and not l.startswith(('def ', 'class ', ' ', '\t'))]
                    if top_level_lines:
                        chunks.append({
                            "filename": filename,
                            "type": "Python Script Setup",
                            "line_number": 1,
                            "code": "\n".join(top_level_lines[:25])
                        })
                except SyntaxError:
                    pass

            # C, Java, JS, and TS Structural Bracket Scanner
            elif extension in ['java', 'c', 'js', 'ts', 'cpp', 'h']:
                block_signature = re.compile(r'(?:(?:public|private|protected|static|final|async|export|function|class)\s+)*[\w\<\>\[\]]+\s+\w+\s*\([^\)]*\)\s*\{|class\s+\w+.*\{')
                
                for match in block_signature.finditer(raw_content):
                    start_index = match.start()
                    line_count = raw_content[:start_index].count('\n') + 1
                    
                    bracket_depth = 0
                    end_index = -1
                    for i in range(start_index, len(raw_content)):
                        if raw_content[i] == '{':
                            bracket_depth += 1
                        elif raw_content[i] == '}':
                            bracket_depth -= 1
                            if bracket_depth == 0:
                                end_index = i + 1
                                break
                    
                    if end_index != -1:
                        code_block = raw_content[start_index:end_index]
                        if len(code_block.strip()) > 2:
                            first_line = code_block.splitlines()[0]
                            kind = "Class" if "class " in first_line else "Function/Method"
                            chunks.append({
                                "filename": filename,
                                "type": f"{extension.upper()} {kind}",
                                "line_number": line_count,
                                "code": code_block
                            })

            # Flat File Sequential Splitter Fallback
            if not chunks:
                window_size = 15
                for i in range(0, len(lines), window_size):
                    code_block = "\n".join(lines[i:i+window_size])
                    if code_block.strip():
                        chunks.append({
                            "filename": filename,
                            "type": "Text Segment / Config Block",
                            "line_number": i + 1,
                            "code": code_block
                        })
        except Exception as e:
            pass
                
    return chunks

# --- 4. ENGINE STATE MEMORY ---
if "code_database" not in st.session_state:
    st.session_state.code_database = []
if "semantic_index" not in st.session_state:
    st.session_state.semantic_index = None
if "keyword_finder" not in st.session_state:
    st.session_state.keyword_finder = None
if "tracked_filenames" not in st.session_state:
    st.session_state.tracked_filenames = set()

# --- SIDEBAR: CONTROLS ---
with st.sidebar:
    st.markdown("### ⚙️ Engine Control Room")
    uploaded_files = st.file_uploader(
        "Drop code or documentation files here",
        type=["py", "js", "ts", "java", "c", "cpp", "txt", "md", "sql", "pdf", "docx", "csv"],
        accept_multiple_files=True
    )
    
    if st.button("Index Workspace", use_container_width=True, type="primary"):
        if uploaded_files:
            extracted_chunks = []
            files_processed = set()
            
            for file in uploaded_files:
                file_bytes = file.read()
                files_processed.add(file.name)
                extracted_chunks.extend(split_file_into_chunks(file.name, file_bytes))
                
            if extracted_chunks:
                corpus_strings = [c["code"] for c in extracted_chunks]
                embeddings = ai_model.encode(corpus_strings, convert_to_numpy=True)
                
                vector_dimension = embeddings.shape[1]
                faiss_matrix = faiss.IndexFlatL2(vector_dimension)
                faiss_matrix.add(embeddings)
                
                tokenized_words = [text.lower().split() for text in corpus_strings]
                bm25_matcher = BM25Okapi(tokenized_words)
                
                st.session_state.code_database = extracted_chunks
                st.session_state.semantic_index = faiss_matrix
                st.session_state.keyword_finder = bm25_matcher
                st.session_state.tracked_filenames = files_processed
                
                st.sidebar.success(f"Successfully indexed {len(extracted_chunks)} matrix elements!")
        else:
            st.sidebar.warning("Drop files in first.")

# --- MAIN WORKSPACE INTERFACE ---
st.markdown('<div class="main-title">DeepGrep // Omnisearch Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">A polyglot semantic workspace search dashboard running vector operations on documentation, datasets, and codebases.</div>', unsafe_allow_html=True)

if st.session_state.code_database:
    user_query = st.text_input(
        "Search your indexed files",
        placeholder="Type a logic concept, document project requirements, variable signature, or cell data parameter...",
        key="workspace_search"
    )
    
    if user_query:
        query_vector = ai_model.encode([user_query], convert_to_numpy=True)
        total_chunks = len(st.session_state.code_database)
        distances, space_indices = st.session_state.semantic_index.search(query_vector, total_chunks)
        
        query_tokens = user_query.lower().split()
        keyword_scores = st.session_state.keyword_finder.get_scores(query_tokens)
        
        def scale_scores(values, invert=False):
            v_min, v_max = values.min(), values.max()
            if v_max == v_min:
                return np.zeros_like(values)
            scaled = (values - v_min) / (v_max - v_min)
            return (1.0 - scaled) if invert else scaled  # ✅ Changed to match the parameter name

        semantic_scores_mapped = np.zeros(total_chunks)
        for placement, idx in enumerate(space_indices[0]):
            semantic_scores_mapped[idx] = distances[0][placement]
            
        scaled_semantic = scale_scores(semantic_scores_mapped, invert=True)
        scaled_keywords = scale_scores(keyword_scores)
        
        # FIXED USER RATIO: Sharp 50% Intent Concept + 50% Word Precision matching distribution
        final_scores = (0.5 * scaled_semantic) + (0.5 * scaled_keywords)
        
        sorted_indices = np.argsort(final_scores)[::-1]
        best_matches = [idx for idx in sorted_indices if final_scores[idx] > 0.05][:5]
        
        hits_found_in_files = set(st.session_state.code_database[idx]["filename"] for idx in best_matches)
        
        left_panel, right_panel = st.columns([1, 2], gap="large")
        
        with left_panel:
            st.markdown("### Workspace Structure")
            st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
            
            for file_name in sorted(list(st.session_state.tracked_filenames)):
                if file_name in hits_found_in_files:
                    st.markdown(f"🟢 **{file_name}** <span class=\"match-tag\">Match</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span class=\"idle-tag\">⚫ {file_name}</span>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with right_panel:
            st.markdown("### Top Ranked Snippets")
            if best_matches:
                for idx in best_matches:
                    match_item = st.session_state.code_database[idx]
                    percentage = int(final_scores[idx] * 100)
                    
                    # Deduce display formatting frame
                    f_name = match_item['filename']
                    if f_name.endswith(('.py', '.js', '.ts', '.java', '.c', '.cpp', '.sql')):
                        lang_theme = "python" if f_name.endswith('.py') else ("javascript" if f_name.endswith(('.js', '.ts')) else "java")
                        code_display = match_item["code"]
                    else:
                        # For text blocks, PDFs, or CSV rows, display as clean formatting markdown text
                        lang_theme = "markdown"
                        code_display = match_item["code"]
                    
                    header_line = f"📄 {match_item['filename']} | {match_item['type']} — Match Score: {percentage}%"
                    with st.expander(header_line, expanded=(idx == best_matches[0])):
                        st.code(code_display, language=lang_theme)
            else:
                st.info("No relevant file content matches found.")
else:
    st.info("👋 DeepGrep is idle. Drop source code or documentation assets on the left and click index.")