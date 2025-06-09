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
import camelot  # for advanced table extraction
import pandas as pd  # for better table handling
import cv2      # for image processing in OCR

# ----- Configuration -----
class Config:
    MAX_TOKENS_CHUNK = 1000
    OCR_DPI = 200
    OCR_LANG = 'eng'
    EMBED_BATCH_SIZE = 4
    LLM_N_CTX = 2048
    CACHE_SIZE = 5
    GPU_LAYERS = -1  # -1 means use all available layers
    MAIN_GPU = 0
    USE_GPU = False
    N_THREADS = max(1, os.cpu_count() // 2) if not torch.cuda.is_available() else 4
    
    # --- Model Identifiers for Llama.from_pretrained ---
    LLM_REPO = "NousResearch/Hermes-2-Pro-Mistral-7B-GGUF"
    LLM_FILENAME = "Hermes-2-Pro-Mistral-7B.Q2_K.gguf"
    EMBED_REPO = "CompendiumLabs/bge-large-en-v1.5-gguf"
    EMBED_FILENAME = "bge-large-en-v1.5-f16.gguf"
    
    # Camelot table extraction settings
    CAMELOT_FLAVOR = 'lattice'
    CAMELOT_EDGE_TOL = 500
    CAMELOT_ROW_TOL = 2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_assistant.log'),
        logging.StreamHandler()
    ]
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

# Initialize models using the new, simplified method
try:
    logger.info(f"Loading LLM from repo: {Config.LLM_REPO}")
    llm = Llama.from_pretrained(
        repo_id=Config.LLM_REPO,
        filename=Config.LLM_FILENAME,
        n_ctx=Config.LLM_N_CTX,
        n_gpu_layers=Config.GPU_LAYERS if Config.USE_GPU else 0,
        main_gpu=Config.MAIN_GPU if Config.USE_GPU else 0,
        n_threads=Config.N_THREADS,
        seed=42,
        verbose=False
    )
    logger.info(f"LLM model loaded {'with GPU acceleration' if Config.USE_GPU else 'on CPU'}")
except Exception as e:
    logger.critical(f"Failed to load LLM model: {str(e)}", exc_info=True)
    raise

try:
    logger.info(f"Loading Embedding model from repo: {Config.EMBED_REPO}")
    embedding_model = Llama.from_pretrained(
        repo_id=Config.EMBED_REPO,
        filename=Config.EMBED_FILENAME,
        n_ctx=Config.LLM_N_CTX,
        n_gpu_layers=Config.GPU_LAYERS if Config.USE_GPU else 0,
        main_gpu=Config.MAIN_GPU if Config.USE_GPU else 0,
        n_threads=Config.N_THREADS,
        seed=42,
        verbose=False,
        embedding=True
    )
    logger.info(f"Embedding model loaded {'with GPU acceleration' if Config.USE_GPU else 'on CPU'}")
except Exception as e:
    logger.critical(f"Failed to load embedding model: {str(e)}", exc_info=True)
    raise

# -------- Enhanced Table Extraction with Camelot ---------

def extract_tables_with_camelot(pdf_path: str, pages: str = 'all') -> List[camelot.core.Table]:
    """Extract tables using Camelot, trying both lattice and stream methods."""
    tables = []
    logger.info(f"Running Camelot on {os.path.basename(pdf_path)} for pages: {pages}")
    
    try:
        lattice_tables = camelot.read_pdf(
            pdf_path, pages=pages, flavor='lattice', edge_tol=Config.CAMELOT_EDGE_TOL, suppress_stdout=True
        )
        tables.extend([t for t in lattice_tables if t.accuracy > 80])
        logger.info(f"Camelot (lattice) found {len(tables)} tables.")
    except Exception as e:
        logger.warning(f"Camelot lattice method failed: {e}")

    try:
        stream_tables = camelot.read_pdf(
            pdf_path, pages=pages, flavor='stream', row_tol=Config.CAMELOT_ROW_TOL, suppress_stdout=True
        )
        for t in stream_tables:
            if t.df.shape[0] > 1 and t.df.shape[1] > 1:
                tables.append(t)
        logger.info(f"Camelot (stream) found {len(stream_tables)} additional tables.")
    except Exception as e:
        logger.warning(f"Camelot stream method failed: {e}")
        
    return tables

def format_table_for_embedding(df: pd.DataFrame) -> str:
    """Convert DataFrame to a text format suitable for embedding."""
    df = df.replace(r'^\s*$', np.nan, regex=True).dropna(how='all').dropna(axis=1, how='all')
    if df.empty:
        return ""
    
    headers = " | ".join(str(col).strip() for col in df.columns)
    rows = [" | ".join(str(val).strip() for val in row if pd.notna(val)) for _, row in df.iterrows()]
    return f"Table Headers: {headers}\n" + "\n".join(f"Row: {row}" for row in rows if row)

@lru_cache(maxsize=Config.CACHE_SIZE * 10)
def ocr_page(page: fitz.Page) -> str:
    """Performs OCR on a single PyMuPDF page object."""
    try:
        pix = page.get_pixmap(dpi=Config.OCR_DPI)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
        text = pytesseract.image_to_string(img, lang=Config.OCR_LANG)
        return text.strip()
    except Exception as e:
        logger.error(f"OCR failed for page {page.number}: {str(e)}")
        return ""

def extract_text_and_tables(pdf_path: str) -> List[Dict[str, Any]]:
    """Extracts text and tables from a PDF using a hybrid approach."""
    try:
        doc = fitz.open(pdf_path)
        pages_content = []

        all_camelot_tables = extract_tables_with_camelot(pdf_path)
        tables_by_page = {i: [] for i in range(1, len(doc) + 1)}
        for table in all_camelot_tables:
            tables_by_page[table.page].append(table.df)
        
        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            text = page.get_text("text")
            
            if len(text.strip()) < 50:
                logger.info(f"Page {page_num} has sparse text, falling back to OCR.")
                text = ocr_page(page)

            page_tables = []
            if page_num in tables_by_page:
                for df in tables_by_page[page_num]:
                    formatted_table = format_table_for_embedding(df)
                    if formatted_table:
                        page_tables.append({'dataframe': df, 'text': formatted_table, 'extraction_method': 'camelot'})
            
            if not page_tables:
                for table in page.find_tables():
                    df = table.to_pandas()
                    formatted_table = format_table_for_embedding(df)
                    if formatted_table:
                        page_tables.append({'dataframe': df, 'text': formatted_table, 'extraction_method': 'pymupdf'})
            
            pages_content.append({"text": text.strip(), "tables": page_tables, "page_number": page_num})
        
        logger.info(f"Extracted content from {len(doc)} pages.")
        return pages_content
        
    except Exception as e:
        logger.error(f"Error extracting from {pdf_path}: {str(e)}")
        return [{"text": "", "tables": [], "page_number": 1}]

# -------- Helper Functions ---------

def chunk_text(text: str, max_chunk_size: int = Config.MAX_TOKENS_CHUNK) -> List[str]:
    words = text.split()
    if not words: return []
    return [" ".join(words[i:i + max_chunk_size]) for i in range(0, len(words), max_chunk_size)]

def embed_texts(texts: List[str]) -> np.ndarray:
    embeddings = []
    for i in range(0, len(texts), Config.EMBED_BATCH_SIZE):
        batch = texts[i:i + Config.EMBED_BATCH_SIZE]
        try:
            embs = embedding_model.embed(batch)
            embeddings.extend(embs)
        except Exception as e:
            logger.error(f"Embedding failed for batch, adding zero vectors: {e}")
            embeddings.extend([np.zeros(embedding_model.n_embd())] * len(batch))
    return np.array(embeddings)

def summarize_text(text: str, stop_event: threading.Event) -> str:
    if stop_event.is_set(): return "Operation cancelled."
    try:
        prompt = f"Summarize the following text clearly and concisely:\n\n{text}\n\nSummary:"
        response = llm(prompt, max_tokens=256, stop=["\n\n"], temperature=0.3)
        return response['choices'][0]['text'].strip()
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return "Summary generation failed."

def answer_question(question: str, docs: List[str], embeddings: np.ndarray, stop_event: threading.Event, top_k: int = 4) -> str:
    if stop_event.is_set(): return "Operation cancelled."
    try:
        q_emb = np.array(embedding_model.embed([question]))
        sims = cosine_similarity(q_emb, embeddings)[0]
        top_indices = sims.argsort()[::-1][:top_k]
        context = "\n\n---\n\n".join([docs[i] for i in top_indices])
        
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        response = llm(prompt, max_tokens=512, stop=["\n\n", "Question:"], temperature=0.2)
        return response['choices'][0]['text'].strip()
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        return "Failed to generate answer."

def compare_pdfs(pdf1_path: str, pdf2_path: str) -> str:
    try:
        doc1 = fitz.open(pdf1_path)
        doc2 = fitz.open(pdf2_path)
        diff_html = []
        for i in range(max(len(doc1), len(doc2))):
            text1, text2 = (doc1[i].get_text() if i < len(doc1) else ""), (doc2[i].get_text() if i < len(doc2) else "")
            diff_lines = list(difflib.ndiff(text1.splitlines(), text2.splitlines()))
            page_diff, has_diff = [], False
            for line in diff_lines:
                if line.startswith("? "): continue
                has_diff = True
                line_esc = html.escape(line[2:])
                if line.startswith("+ "): page_diff.append(f'<div style="background-color:#d4edda;">+ {line_esc}</div>')
                elif line.startswith("- "): page_diff.append(f'<div style="background-color:#f8d7da;">- {line_esc}</div>')
            if has_diff: diff_html.append(f"<h3>Page {i+1} Differences</h3>{''.join(page_diff)}")
        return "<hr>".join(diff_html) if diff_html else "<h3>No textual differences found.</h3>"
    except Exception as e:
        logger.error(f"PDF comparison failed: {e}")
        return f"Comparison error: {str(e)}"

# ---- Main Assistant Class ----
class PDFAssistant:
    def __init__(self):
        self.uploaded_files: Dict[str, str] = {}
        self.docs: Dict[str, List[str]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.table_data: Dict[str, List[Dict]] = {}
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def _get_path_from_basename(self, basename: str) -> str:
        return self.uploaded_files.get(basename)

    def clear_resources(self):
        with self.lock:
            self.uploaded_files.clear(); self.docs.clear(); self.embeddings.clear(); self.table_data.clear()
            if Config.USE_GPU: torch.cuda.empty_cache()
            logger.info("All resources cleared.")
        return ("All resources cleared.", gr.update(choices=[], value=[]), gr.update(choices=[], value=[]),
                gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), pd.DataFrame(), "",
                gr.update(value=""), gr.update(value=""), gr.update(value=""))

    def upload_pdfs(self, files: List[gr.File], progress=gr.Progress()):
        self.stop_event.clear()
        with self.lock:
            self.clear_resources()
            if not files: return "No files uploaded.", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

            for file_obj in files: self.uploaded_files[os.path.basename(file_obj.name)] = file_obj.name
            
            for i, (basename, path) in enumerate(self.uploaded_files.items()):
                progress(i / len(self.uploaded_files), desc=f"Processing {basename}")
                if self.stop_event.is_set(): return "Processing cancelled.", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                
                pages_content = extract_text_and_tables(path)
                combined_chunks, page_tables = [], []
                for page in pages_content:
                    combined_chunks.extend(chunk_text(page["text"]))
                    for tbl in page["tables"]: combined_chunks.extend(chunk_text(tbl['text']))
                    page_tables.extend(page["tables"])
                
                self.docs[basename] = combined_chunks
                self.embeddings[basename] = embed_texts(combined_chunks)
                self.table_data[basename] = page_tables
            
            file_list = list(self.uploaded_files.keys())
            msg = f"Successfully processed {len(self.uploaded_files)} PDFs."
            logger.info(msg)
            return (msg, gr.update(choices=file_list, value=[]), gr.update(choices=file_list, value=[]),
                    gr.update(choices=file_list), gr.update(choices=file_list), gr.update(choices=file_list))

    def summarize_selected(self, selected_files: List[str], progress=gr.Progress()):
        if not selected_files: return "Please select one or more PDFs to summarize."
        self.stop_event.clear()
        summaries = []
        for i, basename in enumerate(selected_files):
            progress(i / len(selected_files), desc=f"Summarizing {basename}")
            full_text = "\n\n".join(self.docs.get(basename, []))
            summary = summarize_text(full_text, self.stop_event)
            if self.stop_event.is_set(): return "Summarization cancelled."
            table_count = len(self.table_data.get(basename, []))
            table_info = f"\n\n*This document contains {table_count} extracted tables.*" if table_count > 0 else ""
            summaries.append(f"### Summary of `{basename}`\n\n{summary}{table_info}")
        return "\n\n---\n\n".join(summaries)

    def ask_question(self, question: str, selected_files: List[str]):
        if not question.strip(): return "Please enter a question."
        if not selected_files: return "Please select PDFs to query."
        self.stop_event.clear()

        combined_docs, combined_embeds_list = [], []
        for basename in selected_files:
            combined_docs.extend(self.docs.get(basename, []))
            if basename in self.embeddings and self.embeddings[basename].size > 0:
                combined_embeds_list.append(self.embeddings[basename])
        if not combined_docs or not combined_embeds_list: return "No text content found to answer the question."
        
        return answer_question(question, combined_docs, np.vstack(combined_embeds_list), self.stop_event)

    def compare_selected(self, file1_name: str, file2_name: str):
        if not file1_name or not file2_name: return "Please select two PDFs to compare."
        if file1_name == file2_name: return "Please select two different PDFs."
        path1, path2 = self._get_path_from_basename(file1_name), self._get_path_from_basename(file2_name)
        if not path1 or not path2: return "Error: Could not find file paths."
        return compare_pdfs(path1, path2)

    def show_tables(self, doc_name: str, page_num: int):
        if not doc_name: return pd.DataFrame(), "Select a PDF first."
        tables = self.table_data.get(doc_name, [])
        if not tables: return pd.DataFrame(), f"No tables found in `{doc_name}`."
        
        tables_on_page = [t for t in tables if t.get('page_number') == page_num] if page_num > 0 else tables
        if not tables_on_page: return pd.DataFrame(), f"No tables found on page {page_num} of `{doc_name}`."
        
        first_table = tables_on_page[0]
        return first_table['dataframe'], f"Displaying table 1 of {len(tables_on_page)} from page {first_table['page_number']}. Method: `{first_table['extraction_method']}`."

    def cancel_long_task(self):
        self.stop_event.set()
        logger.info("Operation cancelled by user.")
        return "Operation cancelled."

# ----- Gradio UI -----
assistant = PDFAssistant()

with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 📊 Advanced PDF Assistant with Table Extraction")

    with gr.Tab("📤 Upload & Process"):
        upload = gr.File(file_types=[".pdf"], file_count="multiple", label="Upload PDFs")
        with gr.Row():
            upload_btn = gr.Button("Process PDFs", variant="primary")
            clear_btn = gr.Button("Clear All Data", variant="stop")
        upload_status = gr.Textbox(label="Status", interactive=False)

    with gr.Tab("📝 Summarize"):
        pdf_list_summarize = gr.CheckboxGroup(label="Select PDFs to Summarize")
        with gr.Row():
            summarize_btn = gr.Button("Generate Summary", variant="primary")
            cancel_sum = gr.Button("Cancel", variant="secondary")
        summarize_output = gr.Markdown(label="Summary Output")

    with gr.Tab("💬 Chatbot"):
        pdf_list_chat = gr.CheckboxGroup(label="Select PDFs to Query")
        question_input = gr.Textbox(label="Ask a question about the selected PDF(s)", placeholder="e.g., What were the total revenues in 2023?")
        with gr.Row():
            ask_btn = gr.Button("Ask Question", variant="primary")
            cancel_qa = gr.Button("Cancel", variant="secondary")
        answer_output = gr.Markdown(label="Answer")

    with gr.Tab("🔍 Compare PDFs"):
        with gr.Row():
            pdf_list_compare1 = gr.Dropdown(label="Select First PDF")
            pdf_list_compare2 = gr.Dropdown(label="Select Second PDF")
        compare_btn = gr.Button("Compare Selected PDFs", variant="primary")
        compare_output = gr.HTML(label="Comparison Results")

    with gr.Tab("📊 View Tables"):
        pdf_list_tables = gr.Dropdown(label="Select PDF to View Tables From", interactive=True)
        with gr.Row():
            show_tables_btn = gr.Button("Show Tables", variant="primary")
            table_page_select = gr.Number(label="Filter by Page (0 for all)", value=0, precision=0)
        table_info = gr.Markdown()
        table_display = gr.DataFrame(label="Extracted Table Data", interactive=False, wrap=True, max_rows=20, overflow_row_behaviour="paginate")

    # Callbacks
    upload_btn.click(
        assistant.upload_pdfs,
        inputs=[upload],
        outputs=[upload_status, pdf_list_summarize, pdf_list_chat, pdf_list_compare1, pdf_list_compare2, pdf_list_tables]
    )
    
    clear_btn.click(
        assistant.clear_resources,
        outputs=[upload_status, pdf_list_summarize, pdf_list_chat, pdf_list_compare1, pdf_list_compare2, pdf_list_tables, table_display, table_info, summarize_output, answer_output, compare_output]
    )
    
    summarize_click_event = summarize_btn.click(assistant.summarize_selected, [pdf_list_summarize], [summarize_output])
    cancel_sum.click(assistant.cancel_long_task, cancels=[summarize_click_event])
    
    ask_click_event = ask_btn.click(assistant.ask_question, [question_input, pdf_list_chat], [answer_output])
    cancel_qa.click(assistant.cancel_long_task, cancels=[ask_click_event])
    
    compare_btn.click(assistant.compare_selected, [pdf_list_compare1, pdf_list_compare2], [compare_output])
    show_tables_btn.click(assistant.show_tables, [pdf_list_tables, table_page_select], [table_display, table_info])
    
    demo.unload(assistant.clear_resources)

if __name__ == "__main__":
    try:
        # --- Hugging Face Authentication for Kaggle/Cloud Environments ---
        # This block should be at the start of the execution scope
        try:
            from kaggle_secrets import UserSecretsClient
            import huggingface_hub
            user_secrets = UserSecretsClient()
            hf_token = user_secrets.get_secret("HUGGING_FACE_HUB_TOKEN")
            logger.info("Logging into Hugging Face Hub using Kaggle Secret...")
            huggingface_hub.login(token=hf_token)
            logger.info("Successfully logged in.")
        except ImportError:
            logger.info("Not in a Kaggle environment, skipping Kaggle-specific authentication.")
        except Exception as e:
            logger.warning(f"Could not log in to Hugging Face Hub via Kaggle Secrets. Downloads may fail. Error: {e}")

        # Launch the Gradio app
        demo.launch(server_name="0.0.0.0", server_port=7860, enable_queue=True, share=True)

    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}", exc_info=True)
        raise
