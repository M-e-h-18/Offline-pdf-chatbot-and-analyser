import os
import fitz  # PyMuPDF
import gradio as gr
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama
import threading
import time
import logging
import torch
from typing import List, Dict, Any, Tuple
import cv2  # OpenCV
import uuid
import json
import io
import networkx as nx
import matplotlib.pyplot as plt

# ----- Configuration -----
class Config:
    LLM_N_CTX = 8192; EMBED_N_CTX = 512; GPU_LAYERS = -1
    N_THREADS = max(1, os.cpu_count() // 2) if not torch.cuda.is_available() else 4
    LLM_REPO = "NousResearch/Hermes-2-Pro-Mistral-7B-GGUF"; LLM_FILENAME = "Hermes-2-Pro-Mistral-7B.Q4_K_M.gguf"
    EMBED_REPO = "CompendiumLabs/bge-large-en-v1.5-gguf"; EMBED_FILENAME = "bge-large-en-v1.5-f16.gguf"
    MAX_TOKENS_CHUNK = 400; OCR_DPI = 150; EMBED_BATCH_SIZE = 8
    DIFF_COLOR = (0, 0, 255); DIFF_THRESHOLD = 5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Loading ---
try:
    USE_GPU = torch.cuda.is_available()
    if USE_GPU: logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else: logger.info("No GPU detected, using CPU")
except Exception: USE_GPU = False
try:
    logger.info(f"Loading LLM: {Config.LLM_FILENAME}")
    llm = Llama.from_pretrained(repo_id=Config.LLM_REPO, filename=Config.LLM_FILENAME, n_ctx=Config.LLM_N_CTX, n_gpu_layers=Config.GPU_LAYERS if USE_GPU else 0, n_threads=Config.N_THREADS, seed=42, verbose=False)
    logger.info("Loading Embedding Model...")
    embedding_model = Llama.from_pretrained(repo_id=Config.EMBED_REPO, filename=Config.EMBED_FILENAME, n_ctx=Config.EMBED_N_CTX, n_gpu_layers=Config.GPU_LAYERS if USE_GPU else 0, n_threads=Config.N_THREADS, seed=42, verbose=False, embedding=True)
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load models: {str(e)}", exc_info=True); raise

# --- Helper and Core Class Logic ---
def llm_call(prompt: str, max_tokens: int, temperature: float = 0.2, stop: List[str] = None) -> str:
    try:
        response = llm(prompt, max_tokens=max_tokens, stop=stop or ["\n\n", "Question:"], temperature=temperature, echo=False)
        return response['choices'][0]['text'].strip()
    except Exception as e:
        logger.error(f"LLM call failed: {e}"); return "Error: Could not generate a response."
def embed_texts(texts: List[str]) -> np.ndarray:
    all_embeddings = []
    for i in range(0, len(texts), Config.EMBED_BATCH_SIZE):
        batch = texts[i:i + Config.EMBED_BATCH_SIZE]
        try: all_embeddings.extend(embedding_model.embed(batch))
        except Exception: all_embeddings.extend([np.zeros(embedding_model.n_embd())] * len(batch))
    return np.array(all_embeddings)

class ProPDFAssistant:
    def __init__(self):
        self.file_data: Dict[str, Dict[str, Any]] = {}; self.lock = threading.Lock()
        self.manager_thread = threading.Thread(target=self._queue_manager_loop, daemon=True)
        self.manager_thread.start()
    
    def _queue_manager_loop(self):
        logger.info("Queue Manager started.")
        while True:
            try:
                queued_id = None
                with self.lock:
                    for fid, data in self.file_data.items():
                        if data.get("status") == "⏳ Queued":
                            queued_id = fid
                            break
                if queued_id:
                    with self.lock:
                        self.file_data[queued_id]["status"] = "⏳ Processing..."
                        enable_ocr = self.file_data[queued_id].get("ocr_enabled", False)
                    logger.info(f"Manager picked up '{self.file_data[queued_id]['basename']}' from queue.")
                    worker = threading.Thread(target=self._process_worker, args=(queued_id, enable_ocr))
                    worker.start()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in queue manager loop: {e}", exc_info=True)
                time.sleep(5)

    def _process_worker(self, file_id: str, enable_ocr: bool):
        def was_removed():
            with self.lock: return file_id not in self.file_data
        
        try:
            with self.lock: basename = self.file_data[file_id]["basename"]
            if was_removed(): return
            
            chunks, doc = [], fitz.open(self.file_data[file_id]["path"])
            for page in doc:
                if was_removed(): logger.warning(f"Processing cancelled for {basename}."); return
                text = page.get_text("text").strip()
                if enable_ocr and (not text or len(text) < 50):
                    pix = page.get_pixmap(dpi=Config.OCR_DPI); img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = pytesseract.image_to_string(img, lang='eng').strip()
                if text: chunks.extend([{'text': " ".join(text.split()[i:i + Config.MAX_TOKENS_CHUNK]), 'source': f"`{basename}`, page {page.number + 1}"} for i in range(0, len(text.split()), Config.MAX_TOKENS_CHUNK)])
            
            if was_removed() or not chunks:
                 with self.lock: 
                     if file_id in self.file_data: self.file_data[file_id].update({"status": "❌ Error: No text"})
                 return

            full_text = "\n".join(c['text'] for c in chunks)
            summary = llm_call(f"System: You are an expert summarizer.\nUser: Provide a concise, professional summary of the document:\n\n{full_text[:7000]}\n\nSummary:", 400)
            if was_removed(): return
            
            entities_str = llm_call(f"System: You are an expert entity extractor. Respond ONLY with a valid JSON object with keys: 'people', 'organizations', 'locations', 'dates'.\nUser: {full_text[:4000]}\n\nJSON:", 1024, stop=["}"]) + "}"
            try: entities = json.loads(entities_str)
            except: entities = {}
            if was_removed(): return

            embeddings = embed_texts([c['text'] for c in chunks])
            if was_removed(): return
            
            with self.lock:
                if file_id in self.file_data:
                    self.file_data[file_id].update({"chunks": chunks, "embeddings": embeddings, "status": "✅ Processed", "summary": summary, "entities": entities})
                    logger.info(f"Successfully processed {basename}.")
        except Exception as e:
            basename_for_log = "unknown"
            with self.lock:
                if file_id in self.file_data: basename_for_log = self.file_data[file_id].get('basename', file_id)
            logger.error(f"Processing failed for {basename_for_log}: {e}", exc_info=True)
            with self.lock:
                if file_id in self.file_data: self.file_data[file_id]["status"] = "❌ Error"
    
    def queue_files_for_processing(self, ids_to_process: List[str], enable_ocr: bool, current_state: Dict):
        if not ids_to_process: gr.Warning("No files selected to process."); return current_state
        with self.lock:
            self.file_data = current_state
            queued_count = 0
            for file_id in ids_to_process:
                if file_id in self.file_data and self.file_data[file_id]["status"] == "[Staged]":
                    self.file_data[file_id]["status"] = "⏳ Queued"
                    self.file_data[file_id]["ocr_enabled"] = enable_ocr
                    queued_count += 1
        if queued_count > 0: gr.Info(f"Added {queued_count} file(s) to the processing queue.")
        return self.file_data
    
    def add_files_and_update_ui(self, files: List[gr.File], current_state: Dict) -> Tuple:
        with self.lock:
            for file_obj in files:
                basename = os.path.basename(file_obj.name)
                if any(d['basename'] == basename for d in current_state.values()): continue
                current_state[str(uuid.uuid4())] = {"basename": basename, "path": file_obj.name, "status": "[Staged]"}
        gr.Info(f"Added {len(files)} new file(s). Select and click 'Process/Queue'.")
        return (current_state,) + self._get_ui_updates_from_state(current_state)

    def remove_files_and_update_ui(self, ids_to_remove: List[str], current_state: Dict) -> Tuple:
        if not ids_to_remove: 
            gr.Warning("No files selected to remove."); return (current_state,) + self._get_ui_updates_from_state(current_state)
        with self.lock:
            for selected_id in ids_to_remove:
                if selected_id in current_state:
                    logger.info(f"User removed file '{current_state[selected_id]['basename']}' with status '{current_state[selected_id]['status']}'")
                    del current_state[selected_id]
        gr.Info(f"Removed {len(ids_to_remove)} file(s).")
        return (current_state,) + self._get_ui_updates_from_state(current_state)

    def _get_ui_updates_from_state(self, current_state: Dict) -> Tuple:
        with self.lock:
            file_choices = [(f"{d.get('status', '[Staged]')} {d.get('basename', 'N/A')}", fid) for fid, d in sorted(current_state.items(), key=lambda item: item[1].get('basename', ''))]
            processed_files = sorted([d['basename'] for d in current_state.values() if d.get('status') == '✅ Processed'])
        return (gr.update(choices=file_choices, value=[]), gr.update(choices=processed_files), gr.update(choices=processed_files), gr.update(choices=processed_files), gr.update(choices=processed_files))
    
    def live_ui_refresher(self, current_state: Dict):
        while True:
            with self.lock: self.file_data = current_state
            yield self.file_data
            time.sleep(2)

    def get_summary_and_entities(self, selected_name: str):
        if not selected_name: return "Select a processed file.", None, None
        with self.lock: file_id = next((fid for fid, d in self.file_data.items() if d.get('basename') == selected_name), None)
        if not file_id or self.file_data.get(file_id, {}).get('status') != '✅ Processed': return "File not processed.", None, None
        summary = self.file_data[file_id].get('summary', "N/A"); entities = self.file_data[file_id].get('entities', {})
        entity_md = "### Extracted Entities\n\n" + ("No entities found." if not entities else "\n".join([f"**{k.capitalize()}:** {', '.join(v)}\n" for k,v in entities.items() if isinstance(v, list) and v]))
        return f"### Summary\n\n{summary}", entity_md, self._build_and_plot_graph(entities)

    def ask_question(self, question: str, chat_history: List[List[str]], selected_names: List[str]):
        if not question: gr.Warning("Please enter a question."); return chat_history, question
        if not selected_names: gr.Warning("Please select PDF(s) to query."); return chat_history, question
        gr.Info("Thinking..."); chat_history.append([question, None]); yield chat_history, ""
        all_chunks, all_embeds = [], []
        with self.lock:
            for name in selected_names:
                file_id = next((fid for fid, d in self.file_data.items() if d.get('basename') == name and d.get('status') == '✅ Processed'), None)
                if file_id and self.file_data[file_id].get('embeddings') is not None and self.file_data[file_id]['embeddings'].size > 0:
                     all_chunks.extend(self.file_data[file_id]['chunks']); all_embeds.append(self.file_data[file_id]['embeddings'])
        if not all_chunks: chat_history[-1][1] = "No text found to answer."; yield chat_history, "" ; return
        sims = cosine_similarity(embedding_model.embed([question]), np.vstack(all_embeds))[0]
        context, sources = "", set([all_chunks[i]['source'] for i in sims.argsort()[::-1][:5] if sims[i] > 0.4])
        for chunk in all_chunks:
            if chunk['source'] in sources: context += f"Source: {chunk['source']}\nContent: {chunk['text']}\n\n"
        history_str = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history[:-1][-2:]])
        prompt = f"System: You are an AI assistant. Answer based ONLY on the provided context. Cite sources.\n\nContext:\n{context}\n\nUser: {question}\n\nAssistant:"
        answer = llm_call(prompt, 512)
        chat_history[-1][1] = f"{answer}\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sorted(list(sources))); yield chat_history, ""

    def _build_and_plot_graph(self, entities: Dict):
        if not entities or not any(v for v in entities.values()): return None
        G = nx.Graph(); all_entities = [i for s in entities.values() if isinstance(s, list) for i in s]
        for et, el in entities.items():
            if isinstance(el, list): [G.add_node(e, type=et) for e in el]
        for i in range(len(all_entities)):
            for j in range(i + 1, len(all_entities)):
                if G.nodes[all_entities[i]]['type'] != G.nodes[all_entities[j]]['type']: G.add_edge(all_entities[i], all_entities[j])
        if G.number_of_nodes() == 0: return None
        plt.style.use('default'); fig, ax = plt.subplots(figsize=(12, 10)); pos = nx.spring_layout(G, k=0.9, iterations=50, seed=42)
        colors = {'people': '#cde4ff', 'organizations': '#d2f7d2', 'locations': '#ffdddd', 'dates': '#fff8c4'}
        nx.draw_networkx(G, pos, ax=ax, node_color=[colors.get(d['type'], '#e0e0e0') for _, d in G.nodes(data=True)], node_size=3000, with_labels=True, font_size=10, edge_color='#cccccc', width=1.0)
        buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0); plt.close(fig)
        return Image.open(buf)
    
    def compare_and_analyze(self, pdf1_name: str, pdf2_name: str, progress=gr.Progress()):
        if not pdf1_name or not pdf2_name: gr.Warning("Please select two PDFs to compare."); return None, None
        if pdf1_name == pdf2_name: gr.Warning("Please select two different PDFs."); return None, None
        with self.lock:
            pdf1_id = next((fid for fid, data in self.file_data.items() if data.get('basename') == pdf1_name and data.get('status') == '✅ Processed'), None)
            pdf2_id = next((fid for fid, data in self.file_data.items() if data.get('basename') == pdf2_name and data.get('status') == '✅ Processed'), None)
        if not (pdf1_id and pdf2_id): gr.Error("One or both PDFs are not processed. Please process them first."); return "Error: Both PDFs must be processed first.", None
        pdf1_path, pdf2_path = self.file_data[pdf1_id]['path'], self.file_data[pdf2_id]['path']
        try:
            progress(0, desc="Converting PDFs to images..."); images1, images2 = convert_from_path(pdf1_path, dpi=Config.OCR_DPI), convert_from_path(pdf2_path, dpi=Config.OCR_DPI)
        except Exception as e: return f"Error converting PDFs to images: {e}", None
        diff_images, diff_texts = [], []
        for i in progress.tqdm(range(max(len(images1), len(images2))), desc="Comparing pages visually..."):
            img1 = images1[i] if i < len(images1) else Image.new('RGB', images2[0].size, (255, 255, 255)); img2 = images2[i] if i < len(images2) else Image.new('RGB', images1[0].size, (255, 255, 255))
            cv_img1, cv_img2 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR), cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
            if cv_img1.shape != cv_img2.shape: h, w, _ = cv_img1.shape; cv_img2 = cv2.resize(cv_img2, (w, h))
            gray1, gray2 = cv2.cvtColor(cv_img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(cv_img2, cv2.COLOR_BGR2GRAY)
            abs_diff = cv2.absdiff(gray1, gray2); _, thresh = cv2.threshold(abs_diff, Config.DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(cv2.dilate(thresh, None, iterations=2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if any(cv2.contourArea(c) > 40 for c in contours):
                highlight_img = cv_img2.copy(); [cv2.rectangle(highlight_img, cv2.boundingRect(c), Config.DIFF_COLOR, 2) for c in contours]
                diff_images.append(Image.fromarray(cv2.cvtColor(np.hstack((cv_img1, highlight_img)), cv2.COLOR_BGR2RGB))); diff_texts.append(f"- **Differences detected on page {i+1}.**")
        progress(0.9, desc="Generating AI analysis of text..."); pdf1_text, pdf2_text = "\n".join([c['text'] for c in self.file_data[pdf1_id]['chunks']]), "\n".join([c['text'] for c in self.file_data[pdf2_id]['chunks']])
        prompt = f"<|im_start|>system\nYou are a meticulous document analyst. Your task is to compare two versions of a document and provide a concise, bulleted list of the key textual differences. Focus on additions, deletions, and significant modifications.<|im_end|><|im_start|>user\n**Doc 1: {pdf1_name}**\n---\n{pdf1_text[:3500]}\n---\n\n**Doc 2: {pdf2_name}**\n---\n{pdf2_text[:3500]}\n---\n\nAnalyze the key differences.<|im_end|><|im_start|>assistant\n"; ai_analysis = llm_call(prompt, 512, stop=["<|im_end|>"])
        report = f"### AI Analysis of Textual Differences\n\n{ai_analysis}\n\n---\n\n" + ("### Visual Change Summary\n\n" + "\n".join(diff_texts) if diff_texts else "### Visual Change Summary\n\nNo significant visual differences were found.")
        return report, diff_images

# ----- UI Construction -----
assistant = ProPDFAssistant()
with gr.Blocks(fill_height=True, title="Pro PDF Assistant") as demo:
    file_state = gr.State({})
    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=450):
            gr.Markdown("# Pro PDF Assistant")
            with gr.Accordion("🗂️ Document Workflow & Controls", open=True):
                gr.Markdown("**Instructions:**\n1. **Add PDFs**.\n2. **(Optional)** Check **Deep OCR** for scans.\n3. Select files and click **Process/Queue**.\n4. You can add or remove files at any time.")
                file_checkboxes = gr.CheckboxGroup(label="File Management List", type="value")
                ocr_checkbox = gr.Checkbox(label="Enable Deep OCR (for scanned PDFs)", value=False)
                with gr.Row():
                    upload_button = gr.UploadButton("Add PDFs", file_types=[".pdf"], file_count="multiple", variant="secondary", size="sm")
                    process_button = gr.Button("Process/Queue Selected", variant="primary", size="sm")
                    refresh_button = gr.Button("🔄 Refresh List", size="sm")
                remove_button = gr.Button("Remove Selected Files", variant="stop")
        with gr.Column(scale=5):
            with gr.Tabs():
                with gr.TabItem("📄 Summary & Entities"):
                    analysis_selector = gr.Dropdown(label="Select a Document to Analyze", interactive=True)
                    with gr.Row(): summary_output = gr.Markdown(); entity_output = gr.Markdown()
                with gr.TabItem("🕸️ Knowledge Graph"):
                    graph_output = gr.Image(label="Entity Relationship Graph", interactive=False)
                with gr.TabItem("💬 Chat"):
                    chat_selector = gr.Dropdown(label="Select PDFs to Chat With", multiselect=True, interactive=True)
                    chatbot = gr.Chatbot(label="Conversation", height=600, show_copy_button=True)
                    with gr.Row(): question_box = gr.Textbox(placeholder="Ask a question...", scale=5, container=False); ask_button = gr.Button("Ask", variant="primary", scale=1)
                    gr.ClearButton([question_box, chatbot])
                with gr.TabItem("↔️ Compare Documents"):
                    with gr.Row(): pdf1_selector = gr.Dropdown(label="Original (V1)", interactive=True); pdf2_selector = gr.Dropdown(label="New (V2)", interactive=True)
                    compare_button = gr.Button("Compare & Analyze", variant="primary")
                    with gr.Row():
                        comparison_report = gr.Markdown(); comparison_gallery = gr.Gallery(label="Visual Differences", height=650, object_fit="contain")
                    gr.ClearButton([comparison_report, comparison_gallery, pdf1_selector, pdf2_selector])

    update_outputs = [file_checkboxes, chat_selector, analysis_selector, pdf1_selector, pdf2_selector]

    # --- DYNAMIC & ROBUST Event Handling Logic ---
    upload_button.upload(fn=assistant.add_files_and_update_ui, inputs=[upload_button, file_state], outputs=[file_state] + update_outputs)
    remove_button.click(fn=assistant.remove_files_and_update_ui, inputs=[file_checkboxes, file_state], outputs=[file_state] + update_outputs)
    process_button.click(fn=assistant.queue_files_for_processing, inputs=[file_checkboxes, ocr_checkbox, file_state], outputs=[file_state])
    
    refresh_button.click(fn=assistant._get_ui_updates_from_state, inputs=[file_state], outputs=update_outputs)
    
    file_state.change(fn=assistant._get_ui_updates_from_state, inputs=[file_state], outputs=update_outputs)
    
    # Start the live UI refresher
    demo.load(fn=assistant.live_ui_refresher, inputs=[file_state], outputs=[file_state])
    
    # Analysis functions
    ask_button.click(assistant.ask_question, [question_box, chatbot, chat_selector], [chatbot, question_box])
    analysis_selector.change(assistant.get_summary_and_entities, analysis_selector, [summary_output, entity_output, graph_output])
    compare_button.click(fn=assistant.compare_and_analyze, inputs=[pdf1_selector, pdf2_selector], outputs=[comparison_report, comparison_gallery], show_progress="full")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7863, share=True, inbrowser=True)
