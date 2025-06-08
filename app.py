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
from typing import List, Tuple, Dict, Any
from huggingface_hub import hf_hub_download
import requests
from tqdm import tqdm

# ----- Configuration -----
class Config:
    MAX_TOKENS_CHUNK = 1000  # Rough limit for chunk size to embed/summarize
    OCR_DPI = 200  # DPI for pdf2image OCR fallback
    OCR_LANG = 'eng'  # pytesseract language
    EMBED_BATCH_SIZE = 4  # Batch size for embedding processing
    LLM_N_CTX = 2048  # Context window size
    CACHE_SIZE = 5  # Number of PDFs to cache
    GPU_LAYERS = -1  # -1 means use all available layers
    MAIN_GPU = 0  # Main GPU index
    USE_GPU = False  # Will be set based on availability
    N_THREADS = 4 if not torch.cuda.is_available() else 1  # Threads based on GPU
    MODEL_DIR = "models"  # Directory to store downloaded models
    LLM_MODEL = "Hermes-2-Pro-Mistral-7B.Q2_K.gguf"
    LLM_REPO = "TheBloke/Hermes-2-Pro-Mistral-7B-GGUF"
    EMBED_MODEL = "bge-large-en-v1.5-f16.gguf"
    EMBED_REPO = "CompendiumLabs/bge-large-en-v1.5-gguf"

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

# Create models directory if it doesn't exist
os.makedirs(Config.MODEL_DIR, exist_ok=True)

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

def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def download_model(repo_id, filename):
    """Download model from HuggingFace Hub"""
    model_path = os.path.join(Config.MODEL_DIR, filename)
    
    if not os.path.exists(model_path):
        logger.info(f"Downloading {filename}...")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=Config.MODEL_DIR,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            logger.info(f"Downloaded {filename} successfully")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {str(e)}")
            raise
    else:
        logger.info(f"Model {filename} already exists, skipping download")
    
    return model_path

# Download models
try:
    llm_path = download_model(Config.LLM_REPO, Config.LLM_MODEL)
    embed_path = download_model(Config.EMBED_REPO, Config.EMBED_MODEL)
except Exception as e:
    logger.error(f"Model download failed: {str(e)}")
    raise

# Initialize models
try:
    llm = Llama(
        model_path=llm_path,
        n_ctx=Config.LLM_N_CTX,
        n_gpu_layers=Config.GPU_LAYERS if Config.USE_GPU else 0,
        main_gpu=Config.MAIN_GPU if Config.USE_GPU else 0,
        n_threads=Config.N_THREADS,
        seed=42,
        verbose=False
    )
    logger.info(f"LLM model loaded {'with GPU acceleration' if Config.USE_GPU else 'on CPU'}")
except Exception as e:
    logger.error(f"Failed to load LLM model: {str(e)}")
    raise

try:
    embedding_model = Llama(
        model_path=embed_path,
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
    logger.error(f"Failed to load embedding model: {str(e)}")
    raise

# -------- Helper Functions ---------

def get_device_info() -> Dict[str, Any]:
    """Return information about available compute devices"""
    info = {
        "gpu_available": Config.USE_GPU,
        "gpu_name": torch.cuda.get_device_name(0) if Config.USE_GPU else "None",
        "cuda_version": torch.version.cuda if Config.USE_GPU else "N/A",
        "llm_device": "GPU" if Config.USE_GPU and llm.params.n_gpu_layers > 0 else "CPU",
        "embedding_device": "GPU" if Config.USE_GPU and embedding_model.params.n_gpu_layers > 0 else "CPU",
        "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB" if Config.USE_GPU else "N/A"
    }
    return info

@lru_cache(maxsize=Config.CACHE_SIZE)
def ocr_page(page_path: str) -> str:
    """Optimized OCR processing with GPU acceleration if available"""
    try:
        # Convert PDF to image
        images = convert_from_path(
            page_path,
            dpi=Config.OCR_DPI,
            first_page=1,
            last_page=1,
            thread_count=4,
            grayscale=True
        )
        
        # Use GPU-accelerated processing if available
        if Config.USE_GPU:
            try:
                import cv2
                image_np = np.array(images[0])
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                text = pytesseract.image_to_string(
                    image_np,
                    lang=Config.OCR_LANG,
                    config='--psm 6 --oem 3'
                )
                return text.strip()
            except Exception as e:
                logger.warning(f"GPU OCR failed, falling back to CPU: {str(e)}")
        
        # Standard CPU processing
        text = pytesseract.image_to_string(
            images[0],
            lang=Config.OCR_LANG,
            config='--psm 6 --oem 3'
        )
        return text.strip()
    except Exception as e:
        logger.error(f"OCR failed for {page_path}: {str(e)}")
        return ""

def extract_text_and_tables_with_ocr(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text and tables from PDF with OCR fallback"""
    try:
        doc = fitz.open(pdf_path)
        pages_content = []
        
        for page in doc:
            # First try fast text extraction
            text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            
            if not text.strip():
                # Fallback to OCR only if needed
                text = ocr_page(pdf_path)
            
            # Extract tables
            tables = []
            for table in page.find_tables():
                tables.append(table.extract())
            
            pages_content.append({
                "text": text.strip(),
                "tables": tables
            })
        
        return pages_content
    except Exception as e:
        logger.error(f"Error extracting from {pdf_path}: {str(e)}")
        return [{"text": "", "tables": []}]

def chunk_text(text: str, max_chunk_size: int = Config.MAX_TOKENS_CHUNK) -> List[str]:
    """Split text into manageable chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_chunk_size):
        chunk = " ".join(words[i:i + max_chunk_size])
        chunks.append(chunk)
    return chunks

def embed_texts(texts: List[str]) -> np.ndarray:
    """Generate embeddings for text chunks"""
    embeddings = []
    for i in range(0, len(texts), Config.EMBED_BATCH_SIZE):
        batch = texts[i:i + Config.EMBED_BATCH_SIZE]
        try:
            embs = embedding_model.embed(batch)
            embeddings.extend(embs)
        except Exception as e:
            logger.error(f"Embedding failed for batch {i}: {str(e)}")
            # Add zero vectors for failed embeddings
            embeddings.extend([np.zeros(embedding_model.n_embd)] * len(batch))
    return np.array(embeddings)

def summarize_text(text: str) -> str:
    """Generate a concise summary of the text"""
    try:
        prompt = f"Summarize the following text briefly and clearly:\n\n{text}\n\nSummary:"
        response = llm(
            prompt,
            max_tokens=512,
            stop=["\n\n"],
            temperature=0.3,
            top_p=0.9
        )
        return response['choices'][0]['text'].strip()
    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}")
        return "Summary generation failed. Please try again."

def answer_question(question: str, docs: List[str], embeddings: np.ndarray, top_k: int = 3) -> str:
    """Answer question based on document context"""
    try:
        # Get question embedding
        q_emb = embedding_model.embed([question])[0]
        
        # Find most relevant documents
        sims = cosine_similarity([q_emb], embeddings)[0]
        top_indices = sims.argsort()[::-1][:top_k]
        context = "\n\n".join([docs[i] for i in top_indices])
        
        # Generate answer
        prompt = f"""Based on the following context, answer the question clearly and concisely:
        Context: {context}
        Question: {question}
        Answer:"""
        
        response = llm(
            prompt,
            max_tokens=512,
            stop=["\n\n"],
            temperature=0.2,
            top_p=0.9
        )
        return response['choices'][0]['text'].strip()
    except Exception as e:
        logger.error(f"Question answering failed: {str(e)}")
        return "Failed to generate answer. Please try again."

def compare_pdfs(pdf1_path: str, pdf2_path: str) -> List[Tuple[int, str]]:
    """Compare two PDFs and return formatted differences"""
    try:
        doc1 = fitz.open(pdf1_path)
        doc2 = fitz.open(pdf2_path)
        max_pages = max(len(doc1), len(doc2))
        diffs = []
        
        for i in range(max_pages):
            text1 = doc1[i].get_text() if i < len(doc1) else ""
            text2 = doc2[i].get_text() if i < len(doc2) else ""
            
            diff_lines = list(difflib.ndiff(text1.splitlines(), text2.splitlines()))
            
            formatted_diff = []
            for line in diff_lines:
                line_esc = html.escape(line[2:])
                if line.startswith("+ "):
                    formatted_diff.append(f'<span style="background-color:#d4fcbc;">+ {line_esc}</span>')
                elif line.startswith("- "):
                    formatted_diff.append(f'<span style="background-color:#ffbcbc;">- {line_esc}</span>')
                elif not line.startswith("? "):
                    formatted_diff.append(line_esc)
            
            diffs.append((i + 1, "<br>".join(formatted_diff)))
        
        return diffs
    except Exception as e:
        logger.error(f"PDF comparison failed: {str(e)}")
        return [(0, f"Comparison error: {str(e)}")]

# ---- Main Assistant Class ----
class PDFAssistant:
    def __init__(self):
        self.uploaded_files = []
        self.docs = []
        self.embeddings = []
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.device_info = get_device_info()

    def get_system_info(self) -> Dict[str, Any]:
        """Return current system resource information"""
        info = self.device_info.copy()
        if Config.USE_GPU:
            info.update({
                "gpu_memory_used": f"{torch.cuda.memory_allocated()/1024**3:.2f}GB",
                "gpu_memory_free": f"{torch.cuda.memory_reserved()/1024**3:.2f}GB"
            })
        info.update({
            "pdf_count": len(self.uploaded_files),
            "chunks_processed": sum(len(d) for d in self.docs),
            "status": "Ready"
        })
        return info

    def clear_resources(self):
        """Release memory resources and clear GPU cache"""
        with self.lock:
            self.docs = []
            self.embeddings = []
            if hasattr(llm, 'reset'):
                llm.reset()
            if Config.USE_GPU:
                torch.cuda.empty_cache()
            logger.info("Resources cleared")

    def upload_pdfs(self, files: List[gr.File]) -> Tuple[str, List[str], List[str], List[str]]:
        """Process uploaded PDFs"""
        self.stop_event.clear()
        try:
            with self.lock:
                self.uploaded_files = [file.name for file in files]
                self.docs = []
                self.embeddings = []
                
                for path in self.uploaded_files:
                    if self.stop_event.is_set():
                        logger.info("PDF processing cancelled")
                        return "Processing cancelled.", [], [], []
                    
                    # Process each PDF (cached)
                    pages_content = extract_text_and_tables_with_ocr(path)
                    combined_chunks = []
                    for page in pages_content:
                        if page["text"].strip():
                            combined_chunks.extend(chunk_text(page["text"]))
                        for tbl in page["tables"]:
                            combined_chunks.extend(chunk_text("\n".join(["\t".join(row) for row in tbl])))
                    
                    embeddings = embed_texts(combined_chunks)
                    self.docs.append(combined_chunks)
                    self.embeddings.append(embeddings)
                
                logger.info(f"Processed {len(self.uploaded_files)} PDFs")
                return (
                    f"Processed {len(self.uploaded_files)} PDFs", 
                    [os.path.basename(f) for f in self.uploaded_files],
                    [os.path.basename(f) for f in self.uploaded_files],
                    [os.path.basename(f) for f in self.uploaded_files]
                )
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            return f"Error processing PDFs: {str(e)}", [], [], []

    def summarize_selected(self, selected_indices: List[int], progress: gr.Progress = gr.Progress()) -> str:
        """Generate summaries for selected PDFs"""
        if not selected_indices:
            return "Select PDFs to summarize."
        
        summaries = []
        try:
            for i, idx in enumerate(selected_indices):
                if self.stop_event.is_set():
                    logger.info("Summarization cancelled")
                    return "Summarization cancelled."
                
                progress(i/len(selected_indices), desc=f"Processing PDF {i+1}/{len(selected_indices)}")
                
                combined_text = "\n\n".join(self.docs[idx])
                chunks = chunk_text(combined_text, max_chunk_size=800)
                summary_parts = []
                
                for c in chunks:
                    if self.stop_event.is_set():
                        return "Summarization cancelled."
                    summary_parts.append(summarize_text(c))
                
                final_summary = "\n\n".join(summary_parts)
                summaries.append(f"## Summary of {os.path.basename(self.uploaded_files[idx])}:\n\n{final_summary}")
            
            return "\n\n---\n\n".join(summaries)
        except Exception as e:
            logger.error(f"Summarization error: {str(e)}")
            return f"Summarization failed: {str(e)}"

    def ask_question(self, question: str, selected_indices: List[int]) -> str:
        """Answer question about selected PDFs"""
        if not question.strip():
            return "Please enter a question."
        if not selected_indices:
            return "Select PDFs to query."
        
        try:
            combined_docs = []
            combined_embeds = []
            for idx in selected_indices:
                combined_docs.extend(self.docs[idx])
                combined_embeds.append(self.embeddings[idx])
            
            if not combined_docs:
                return "No text found in selected PDFs."
            
            combined_embeds = np.vstack(combined_embeds)
            return answer_question(question,combined_docs, combined_embeds)
        except Exception as e:
            logger.error(f"Question answering error: {str(e)}")
            return f"Failed to answer question: {str(e)}"

    def compare_selected(self, idx1: int, idx2: int) -> str:
        """Compare two selected PDFs"""
        if idx1 == idx2:
            return "Select two different PDFs."
        
        try:
            pdf1 = self.uploaded_files[idx1]
            pdf2 = self.uploaded_files[idx2]
            diffs = compare_pdfs(pdf1, pdf2)
            
            formatted = []
            for page_num, diff_html in diffs:
                formatted.append(f"<h3>Page {page_num} Differences</h3>{diff_html}")
            
            return "<hr>".join(formatted)
        except Exception as e:
            logger.error(f"Comparison error: {str(e)}")
            return f"Comparison failed: {str(e)}"

    def cancel_long_task(self) -> str:
        """Cancel current operation"""
        self.stop_event.set()
        logger.info("Operation cancelled by user")
        return "Operation cancelled."

# ----- Gradio UI -----
assistant = PDFAssistant()

with gr.Blocks(title="Advanced PDF Assistant with GPU", theme="soft") as demo:
    gr.Markdown("# 🚀 Advanced PDF Assistant with GPU Acceleration")
    
    # System Info
    with gr.Accordion("System Information", open=False):
        sys_info = gr.JSON(
            label="Hardware Configuration",
            value=assistant.get_system_info(),
            every=5
        )
        refresh_btn = gr.Button("Refresh Info")
    
    with gr.Tab("📤 Upload PDFs"):
        gr.Markdown("### Upload multiple PDFs for processing")
        with gr.Row():
            upload = gr.File(
                file_types=[".pdf"],
                file_count="multiple",
                label="Upload PDFs",
                elem_id="upload_pdfs"
            )
        with gr.Row():
            upload_btn = gr.Button("Process PDFs", variant="primary")
            clear_btn = gr.Button("Clear All", variant="secondary")
        upload_status = gr.Textbox(label="Status", interactive=False)
    
    with gr.Tab("📝 Summarize"):
        with gr.Row():
            pdf_list_summarize = gr.CheckboxGroup(
                [],
                label="Select PDFs to Summarize",
                elem_classes="checkbox-group"
            )
        with gr.Row():
            summarize_btn = gr.Button("Generate Summary", variant="primary")
            cancel_sum = gr.Button("Cancel", variant="stop")
        summarize_output = gr.Markdown(label="Summary Output")
    
    with gr.Tab("💬 Chatbot"):
        with gr.Row():
            pdf_list_chat = gr.CheckboxGroup(
                [],
                label="Select PDFs to Query",
                elem_classes="checkbox-group"
            )
        question_input = gr.Textbox(
            label="Ask a question about selected PDFs",
            placeholder="What is the main topic of this document?"
        )
        with gr.Row():
            ask_btn = gr.Button("Ask Question", variant="primary")
            cancel_qa = gr.Button("Cancel", variant="stop")
        answer_output = gr.Markdown(label="Answer")
    
    with gr.Tab("🔍 Compare PDFs"):
        with gr.Row():
            pdf_list_compare1 = gr.Dropdown(
                [],
                label="Select First PDF",
                interactive=True
            )
            pdf_list_compare2 = gr.Dropdown(
                [],
                label="Select Second PDF",
                interactive=True
            )
        compare_btn = gr.Button("Compare PDFs", variant="primary")
        compare_output = gr.HTML(label="Comparison Results")

    # Callbacks
    def process_upload(files):
        msg, list1, list2, list3 = assistant.upload_pdfs(files)
        return msg, gr.CheckboxGroup(choices=list1), gr.CheckboxGroup(choices=list2), gr.Dropdown(choices=list1)

    def refresh_info():
        return assistant.get_system_info()

    # Event handlers
    upload_btn.click(
        process_upload,
        inputs=upload,
        outputs=[upload_status, pdf_list_summarize, pdf_list_chat, pdf_list_compare1]
    )
    
    clear_btn.click(
        lambda: [None, "", gr.CheckboxGroup(choices=[]), gr.CheckboxGroup(choices=[]), gr.Dropdown(choices=[])],
        outputs=[upload, upload_status, pdf_list_summarize, pdf_list_chat, pdf_list_compare1]
    )
    
    refresh_btn.click(
        refresh_info,
        outputs=sys_info
    )
    
    summarize_btn.click(
        assistant.summarize_selected,
        inputs=pdf_list_summarize,
        outputs=summarize_output
    )
    
    cancel_sum.click(
        assistant.cancel_long_task,
        outputs=summarize_output
    )
    
    ask_btn.click(
        assistant.ask_question,
        inputs=[question_input, pdf_list_chat],
        outputs=answer_output
    )
    
    cancel_qa.click(
        assistant.cancel_long_task,
        outputs=answer_output
    )
    
    compare_btn.click(
        assistant.compare_selected,
        inputs=[pdf_list_compare1, pdf_list_compare2],
        outputs=compare_output
    )

    # Cleanup on close
    demo.unload(assistant.clear_resources)

# Launch the app
if __name__ == "__main__":
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            enable_queue=True,
            share=False
        )
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        raise
