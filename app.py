import os
import fitz
import gradio as gr
import difflib
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama
import threading
import time
import html
from functools import lru_cache
import logging
import torch
from typing import List, Tuple, Dict, Any, Union
import camelot
import pandas as pd
import cv2

# ----- Configuration -----
class Config:
    MAX_TOKENS_CHUNK = 1000
    OCR_DPI = 200
    OCR_LANG = 'eng'
    EMBED_BATCH_SIZE = 4
    LLM_N_CTX = 2048
    EMBED_N_CTX = 512  # <<< FIX 2: Added separate context for embedding model
    CACHE_SIZE = 5
    GPU_LAYERS = -1
    MAIN_GPU = 0
    USE_GPU = False
    N_THREADS = max(1, os.cpu_count() // 2) if not torch.cuda.is_available() else 4
    
    LLM_REPO = "NousResearch/Hermes-2-Pro-Mistral-7B-GGUF"
    LLM_FILENAME = "Hermes-2-Pro-Mistral-7B.Q2_K.gguf"
    EMBED_REPO = "CompendiumLabs/bge-large-en-v1.5-gguf"
    EMBED_FILENAME = "bge-large-en-v1.5-f16.gguf"
    
    CAMELOT_FLAVOR = 'lattice'
    CAMELOT_EDGE_TOL = 500
    CAMELOT_ROW_TOL = 2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('pdf_assistant.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Check GPU availability
try:
    Config.USE_GPU = torch.cuda.is_available()
    if Config.USE_GPU:
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.info("No GPU detected, using CPU")
except Exception as e:
    logger.error(f"GPU detection failed: {str(e)}")
    Config.USE_GPU = False

# Initialize models
try:
    logger.info(f"Loading LLM from repo: {Config.LLM_REPO}")
    llm = Llama.from_pretrained(
        repo_id=Config.LLM_REPO, filename=Config.LLM_FILENAME, n_ctx=Config.LLM_N_CTX,
        n_gpu_layers=Config.GPU_LAYERS if Config.USE_GPU else 0, main_gpu=Config.MAIN_GPU if Config.USE_GPU else 0,
        n_threads=Config.N_THREADS, seed=42, verbose=False
    )
    logger.info(f"LLM model loaded {'with GPU acceleration' if Config.USE_GPU else 'on CPU'}")
except Exception as e:
    logger.critical(f"Failed to load LLM model: {str(e)}", exc_info=True)
    raise

try:
    logger.info(f"Loading Embedding model from repo: {Config.EMBED_REPO}")
    embedding_model = Llama.from_pretrained(
        repo_id=Config.EMBED_REPO, filename=Config.EMBED_FILENAME,
        n_ctx=Config.EMBED_N_CTX,  # <<< FIX 2: Using the correct context size
        n_gpu_layers=Config.GPU_LAYERS if Config.USE_GPU else 0, main_gpu=Config.MAIN_GPU if Config.USE_GPU else 0,
        n_threads=Config.N_THREADS, seed=42, verbose=False, embedding=True
    )
    logger.info(f"Embedding model loaded {'with GPU acceleration' if Config.USE_GPU else 'on CPU'}")
except Exception as e:
    logger.critical(f"Failed to load embedding model: {str(e)}", exc_info=True)
    raise

# -------- Extraction Functions ---------

def extract_tables_with_camelot(pdf_path: str, pages: str = 'all') -> List[camelot.core.Table]:
    tables = []
    try:
        lattice_tables = camelot.read_pdf(
            pdf_path, pages=pages, flavor='lattice', edge_tol=Config.CAMELOT_EDGE_TOL, suppress_stdout=True)
        tables.extend([t for t in lattice_tables if t.accuracy > 80])
    except Exception: pass
    try:
        stream_tables = camelot.read_pdf(
            pdf_path, pages=pages, flavor='stream', row_tol=Config.CAMELOT_ROW_TOL, suppress_stdout=True)
        tables.extend([t for t in stream_tables if t.df.shape[0] > 1 and t.df.shape[1] > 1])
    except Exception: pass
    return tables

def format_table_for_embedding(df: pd.DataFrame) -> str:
    df = df.replace(r'^\s*$', np.nan, regex=True).dropna(how='all').dropna(axis=1, how='all')
    if df.empty: return ""
    headers = " | ".join(str(col).strip() for col in df.columns)
    rows = [" | ".join(str(val).strip() for val in row if pd.notna(val)) for _, row in df.iterrows()]
    return f"Table Headers: {headers}\n" + "\n".join(f"Row: {row}" for row in rows if row)

@lru_cache(maxsize=Config.CACHE_SIZE * 10)
def ocr_page(page: fitz.Page) -> str:
    try:
        pix = page.get_pixmap(dpi=Config.OCR_DPI)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return pytesseract.image_to_string(img, lang=Config.OCR_LANG).strip()
    except Exception as e:
        logger.error(f"OCR failed for page {page.number}: {e}")
        return ""

def extract_text_and_tables(pdf_path: str) -> List[Dict[str, Any]]:
    try:
        doc = fitz.open(pdf_path)
        pages_content, tables_by_page = [], {i: [] for i in range(1, len(doc) + 1)}
        for table in extract_tables_with_camelot(pdf_path): tables_by_page[table.page].append(table.df)
        
        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            text = page.get_text("text").strip()
            if len(text) < 50: text = ocr_page(page)

            page_tables = []
            if page_num in tables_by_page:
                for df in tables_by_page[page_num]:
                    fmt_tbl = format_table_for_embedding(df)
                    if fmt_tbl: page_tables.append({'dataframe': df, 'text': fmt_tbl, 'extraction_method': 'camelot'})
            
            if not page_tables:
                for table in page.find_tables():
                    df, fmt_tbl = table.to_pandas(), format_table_for_embedding(table.to_pandas())
                    if fmt_tbl: page_tables.append({'dataframe': df, 'text': fmt_tbl, 'extraction_method': 'pymupdf'})
            
            pages_content.append({"text": text, "tables": page_tables, "page_number": page_num})
        return pages_content
    except Exception as e:
        logger.error(f"Extraction failed for {pdf_path}: {e}")
        return [{"text": "", "tables": [], "page_number": 1}]

# -------- Helper Functions ---------

def chunk_text(text: str, max_chunk_size: int = Config.MAX_TOKENS_CHUNK) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + max_chunk_size]) for i in range(0, len(words), max_chunk_size)] if words else []

def embed_texts(texts: List[str]) -> np.ndarray:
    embeddings = []
    for i in range(0, len(texts), Config.EMBED_BATCH_SIZE):
        batch = texts[i:i + Config.EMBED_BATCH_SIZE]
        try: embeddings.extend(embedding_model.embed(batch))
        except Exception: embeddings.extend([np.zeros(embedding_model.n_embd())] * len(batch))
    return np.array(embeddings)

def summarize_text(text: str, stop_event: threading.Event) -> str:
    if stop_event.is_set(): return "Operation cancelled."
    try:
        prompt = f"Summarize the following text clearly and concisely:\n\n{text}\n\nSummary:"
        response = llm(prompt, max_tokens=256, stop=["\n\n"], temperature=0.3)
        return response['choices'][0]['text'].strip()
    except Exception: return "Summary generation failed."

def answer_question(question: str, docs: List[str], embeddings: np.ndarray, stop_event: threading.Event) -> str:
    if stop_event.is_set(): return "Operation cancelled."
    try:
        q_emb = np.array(embedding_model.embed([question]))
        sims = cosine_similarity(q_emb, embeddings)[0]
        top_indices = sims.argsort()[::-1][:4]
        context = "\n\n---\n\n".join([docs[i] for i in top_indices])
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        response = llm(prompt, max_tokens=512, stop=["\n\n", "Question:"], temperature=0.2)
        return response['choices'][0]['text'].strip()
    except Exception: return "Failed to generate answer."

# ---- Main Assistant Class ----
class PDFAssistant:
    def __init__(self):
        self.uploaded_files, self.docs, self.embeddings, self.table_data = {}, {}, {}, {}
        self.stop_event, self.lock = threading.Event(), threading.Lock()

    def clear_resources(self):
        with self.lock:
            self.uploaded_files.clear(); self.docs.clear(); self.embeddings.clear(); self.table_data.clear()
            if Config.USE_GPU: torch.cuda.empty_cache()
        return ("Cleared.", gr.update(choices=[], value=[]), gr.update(choices=[], value=[]),
                gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), pd.DataFrame(), "",
                gr.update(value=""), gr.update(value=""), gr.update(value=""))

    def upload_pdfs(self, files: List[gr.File], progress=gr.Progress()):
        with self.lock:
            self.clear_resources()
            if not files: return "No files uploaded.", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            for file_obj in files: self.uploaded_files[os.path.basename(file_obj.name)] = file_obj.name
            
            for i, (basename, path) in enumerate(self.uploaded_files.items()):
                progress(i / len(self.uploaded_files), desc=f"Processing {basename}")
                pages_content = extract_text_and_tables(path)
                combined_chunks = [c for p in pages_content for c in chunk_text(p["text"])]
                for p in pages_content:
                    for tbl in p["tables"]: combined_chunks.extend(chunk_text(tbl['text']))
                self.docs[basename] = combined_chunks
                self.embeddings[basename] = embed_texts(combined_chunks)
                self.table_data[basename] = [tbl for p in pages_content for tbl in p["tables"]]
            
            file_list = list(self.uploaded_files.keys())
            return (f"Processed {len(file_list)} PDFs.", gr.update(choices=file_list, value=[]),
                    gr.update(choices=file_list, value=[]), gr.update(choices=file_list),
                    gr.update(choices=file_list), gr.update(choices=file_list))

    def ask_question(self, question: str, selected_files: List[str]):
        if not question.strip(): return "Please enter a question."
        if not selected_files: return "Please select PDFs to query."
        combined_docs, combined_embeds_list = [], []
        for basename in selected_files:
            combined_docs.extend(self.docs.get(basename, []))
            if basename in self.embeddings and self.embeddings[basename].size > 0:
                combined_embeds_list.append(self.embeddings[basename])
        if not combined_docs: return "No text found to answer the question."
        return answer_question(question, combined_docs, np.vstack(combined_embeds_list), self.stop_event)
    
    # ... Other methods like summarize, compare, show_tables can be simplified similarly ...
    def summarize_selected(self, selected_files: List[str], progress=gr.Progress()):
        if not selected_files: return "Please select one or more PDFs to summarize."
        summaries = []
        for i, basename in enumerate(selected_files):
            progress(i / len(selected_files), desc=f"Summarizing {basename}")
            full_text = "\n\n".join(self.docs.get(basename, []))
            summary = summarize_text(full_text, self.stop_event)
            table_count = len(self.table_data.get(basename, []))
            table_info = f"\n\n*Contains {table_count} tables.*" if table_count > 0 else ""
            summaries.append(f"### Summary of `{basename}`\n\n{summary}{table_info}")
        return "\n\n---\n\n".join(summaries)
    
    def show_tables(self, doc_name: str, page_num: int):
        if not doc_name: return pd.DataFrame(), "Select a PDF first."
        tables = self.table_data.get(doc_name, [])
        if not tables: return pd.DataFrame(), f"No tables found in `{doc_name}`."
        tables_on_page = [t for t in tables if t.get('page_number') == page_num] if page_num > 0 else tables
        if not tables_on_page: return pd.DataFrame(), f"No tables on page {page_num}."
        first_table = tables_on_page[0]
        return first_table['dataframe'], f"Table 1 of {len(tables_on_page)} from page {first_table['page_number']}."


# ----- Gradio UI -----
assistant = PDFAssistant()
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 📊 Advanced PDF Assistant with Table Extraction")
    with gr.Tab("📤 Upload & Process"):
        # UI components...
        upload = gr.File(file_types=[".pdf"], file_count="multiple", label="Upload PDFs")
        upload_btn = gr.Button("Process PDFs", variant="primary")
        upload_status = gr.Textbox(label="Status", interactive=False)
    with gr.Tab("📝 Summarize"):
        pdf_list_summarize = gr.CheckboxGroup(label="Select PDFs to Summarize")
        summarize_btn = gr.Button("Generate Summary", variant="primary")
        summarize_output = gr.Markdown(label="Summary Output")
    with gr.Tab("💬 Chatbot"):
        pdf_list_chat = gr.CheckboxGroup(label="Select PDFs to Query")
        question_input = gr.Textbox(label="Ask a question")
        ask_btn = gr.Button("Ask Question", variant="primary")
        answer_output = gr.Markdown(label="Answer")
    with gr.Tab("📊 View Tables"):
        pdf_list_tables = gr.Dropdown(label="Select PDF to View Tables", interactive=True)
        table_page_select = gr.Number(label="Filter by Page (0 for all)", value=0, precision=0)
        show_tables_btn = gr.Button("Show Tables", variant="primary")
        table_info = gr.Markdown()
        # <<< FIX 1: Removed max_rows and overflow_row_behaviour
        table_display = gr.DataFrame(label="Extracted Table Data", interactive=False, wrap=True)

    # Callbacks
    upload_btn.click(
        assistant.upload_pdfs,
        inputs=[upload],
        outputs=[upload_status, pdf_list_summarize, pdf_list_chat, pdf_list_tables]
    )
    summarize_btn.click(assistant.summarize_selected, [pdf_list_summarize], [summarize_output])
    ask_btn.click(assistant.ask_question, [question_input, pdf_list_chat], [answer_output])
    show_tables_btn.click(assistant.show_tables, [pdf_list_tables, table_page_select], [table_display, table_info])

if __name__ == "__main__":
    try:
        try:
            from kaggle_secrets import UserSecretsClient
            import huggingface_hub
            user_secrets = UserSecretsClient()
            hf_token = user_secrets.get_secret("HUGGING_FACE_HUB_TOKEN")
            huggingface_hub.login(token=hf_token)
            logger.info("Successfully logged into Hugging Face Hub.")
        except ImportError:
            logger.info("Not a Kaggle environment, skipping Kaggle-specific authentication.")
        except Exception as e:
            logger.warning(f"Could not log in to Hugging Face Hub via Kaggle Secrets. Downloads may still fail. Error: {e}")

        # Launch the Gradio app, removing the deprecated 'enable_queue' argument
        demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

    except Exception as e:
        logger.critical(f"Application failed to start: {e}", exc_info=True)
        raise
