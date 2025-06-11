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
    MAX_TOKENS_CHUNK = 400; OCR_DPI = 200; EMBED_BATCH_SIZE = 8
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
    
    def add_files_to_state(self, files: List[gr.File], current_state: Dict) -> Dict:
        """This function ONLY modifies the state. It does not return UI updates."""
        with self.lock:
            for file_obj in files:
                basename = os.path.basename(file_obj.name)
                if any(d['basename'] == basename for d in current_state.values()): continue
                current_state[str(uuid.uuid4())] = {"basename": basename, "path": file_obj.name, "status": "[Staged]", "chunks": [], "embeddings": None, "summary": None, "entities": None}
        gr.Info(f"Added {len(files)} new file(s). Check the boxes and click 'Process'.")
        return current_state

    def remove_files_from_state(self, ids_to_remove: List[str], current_state: Dict) -> Dict:
        """This function ONLY modifies the state."""
        if not ids_to_remove: gr.Warning("No files selected to remove."); return current_state
        with self.lock:
            for selected_id in ids_to_remove:
                if selected_id in current_state:
                    if os.path.exists(current_state[selected_id]['path']):
                        try: os.remove(current_state[selected_id]['path'])
                        except: pass
                    del current_state[selected_id]
        gr.Info(f"Removed {len(ids_to_remove)} file(s).")
        return current_state
    
    def _process_worker(self, file_id: str):
        with self.lock: path, basename = self.file_data[file_id]["path"], self.file_data[file_id]["basename"]
        try:
            chunks, doc = [], fitz.open(path)
            for page in doc:
                text = page.get_text("text").strip()
                if not text or len(text) < 50:
                    pix = page.get_pixmap(dpi=Config.OCR_DPI); img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = pytesseract.image_to_string(img, lang='eng').strip()
                if text: chunks.extend([{'text': " ".join(text.split()[i:i + Config.MAX_TOKENS_CHUNK]), 'source': f"`{basename}`, page {page.number + 1}"} for i in range(0, len(text.split()), Config.MAX_TOKENS_CHUNK)])
            full_text = "\n".join(c['text'] for c in chunks)
            summary = llm_call(f"System: You are an expert summarizer.\nUser: Provide a concise, professional summary of the document:\n\n{full_text[:7000]}\n\nSummary:", 400)
            entities_str = llm_call(f"System: You are an expert entity extractor. Respond ONLY with a valid JSON object with keys: 'people', 'organizations', 'locations', 'dates'.\nUser: {full_text[:4000]}\n\nJSON:", 1024)
            try: entities = json.loads(entities_str)
            except: entities = {}
            with self.lock: self.file_data[file_id].update({"chunks": chunks, "embeddings": embed_texts([c['text'] for c in chunks]), "status": "✅ Processed", "summary": summary, "entities": entities})
        except Exception as e:
            logger.error(f"Processing failed for {basename}: {e}"); self.file_data[file_id]["status"] = "❌ Error"
            
    def start_processing(self, ids_to_process: List[str], current_state: Dict):
        """Kicks off processing and immediately updates UI to show 'Processing...' status."""
        if not ids_to_process: gr.Warning("No files selected to process."); return current_state
        gr.Info(f"Starting processing for {len(ids_to_process)} file(s)...")
        with self.lock:
            self.file_data = current_state
            for file_id in ids_to_process:
                if file_id in self.file_data and self.file_data[file_id]["status"] == "[Staged]":
                    self.file_data[file_id]["status"] = "⏳ Processing..."; threading.Thread(target=self._process_worker, args=(file_id,)).start()
        return current_state

    def get_ui_updates(self, current_state: Dict) -> Tuple:
        """The MASTER UI update function. Takes the current state and returns updates for all components."""
        with self.lock:
            self.file_data = current_state # Sync internal state
            # Format choices for the CheckboxGroup: (Display Name, ID)
            file_choices = [(f"{d['status']} {d['basename']}", fid) for fid, d in current_state.items()]
            processed_files = sorted([d['basename'] for d in current_state.values() if d['status'] == '✅ Processed'])
        
        # Return updates for all components that depend on the file list
        return gr.update(choices=file_choices, value=[]), gr.update(choices=processed_files), gr.update(choices=processed_files), gr.update(choices=processed_files), gr.update(choices=processed_files)

    def ask_question(self, question: str, chat_history: List[List[str]], selected_names: List[str]):
        if not question: gr.Warning("Please enter a question."); return chat_history, question
        if not selected_names: gr.Warning("Please select PDF(s) to query."); return chat_history, question
        gr.Info("Thinking..."); chat_history.append([question, None]); yield chat_history, ""
        all_chunks, all_embeds = [], []
        with self.lock:
            for name in selected_names:
                file_id = next((fid for fid, d in self.file_data.items() if d['basename'] == name and d['status'] == '✅ Processed'), None)
                if file_id and self.file_data[file_id]['embeddings'].size > 0: all_chunks.extend(self.file_data[file_id]['chunks']); all_embeds.append(self.file_data[file_id]['embeddings'])
        if not all_chunks: chat_history[-1][1] = "No text found to answer."; yield chat_history, "" ; return
        sims = cosine_similarity(embedding_model.embed([question]), np.vstack(all_embeds))[0]
        context, sources = "", set([all_chunks[i]['source'] for i in sims.argsort()[::-1][:5] if sims[i] > 0.4])
        for chunk in all_chunks:
            if chunk['source'] in sources: context += f"Source: {chunk['source']}\nContent: {chunk['text']}\n\n"
        history_str = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history[:-1][-2:]])
        prompt = f"System: You are an AI assistant. Answer based ONLY on the provided context and conversation history. Cite sources.\n\nHistory:\n{history_str}\n\nContext:\n{context}\n\nUser: {question}\n\nAssistant:"
        answer = llm_call(prompt, 512)
        chat_history[-1][1] = f"{answer}\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sorted(list(sources))); yield chat_history, ""

    def get_summary_and_entities(self, selected_name: str):
        if not selected_name: return "Select a processed file.", None, None
        with self.lock: file_id = next((fid for fid, d in self.file_data.items() if d['basename'] == selected_name), None)
        if not file_id or self.file_data[file_id]['status'] != '✅ Processed': return "File not processed.", None, None
        summary = self.file_data[file_id].get('summary', "N/A"); entities = self.file_data[file_id].get('entities', {})
        entity_md = "### Extracted Entities\n\n" + ("No entities found." if not entities else "\n".join([f"**{k.capitalize()}:** {', '.join(v)}\n" for k,v in entities.items() if isinstance(v, list) and v]))
        return f"### Summary\n\n{summary}", entity_md, self._build_and_plot_graph(entities, selected_name)

    def _build_and_plot_graph(self, entities: Dict, doc_name: str):
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
            pdf1_id = next((fid for fid, data in self.file_data.items() if data['basename'] == pdf1_name), None)
            pdf2_id = next((fid for fid, data in self.file_data.items() if data['basename'] == pdf2_name), None)
        if not (pdf1_id and pdf2_id): gr.Error("One or both selected PDFs not found in the state."); return "Error: PDF data not found.", None
        pdf1_path, pdf2_path = self.file_data[pdf1_id]['path'], self.file_data[pdf2_id]['path']
        try: images1, images2 = convert_from_path(pdf1_path, dpi=Config.OCR_DPI), convert_from_path(pdf2_path, dpi=Config.OCR_DPI)
        except Exception as e: return f"Error converting PDFs to images: {e}", None
        diff_images, diff_texts = [], []
        for i in progress.tqdm(range(max(len(images1), len(images2))), desc="Comparing pages..."):
            img1 = images1[i] if i < len(images1) else Image.new('RGB', images2[0].size, (255, 255, 255))
            img2 = images2[i] if i < len(images2) else Image.new('RGB', images1[0].size, (255, 255, 255))
            cv_img1, cv_img2 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR), cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
            gray1, gray2 = cv2.cvtColor(cv_img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(cv_img2, cv2.COLOR_BGR2GRAY)
            abs_diff = cv2.absdiff(gray1, gray2); _, thresh = cv2.threshold(abs_diff, Config.DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(cv2.dilate(thresh, None, iterations=2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if any(cv2.contourArea(c) > 40 for c in contours):
                highlight_img = cv_img2.copy()
                for c in contours: cv2.rectangle(highlight_img, cv2.boundingRect(c), Config.DIFF_COLOR, 2)
                diff_images.append(Image.fromarray(cv2.cvtColor(np.hstack((cv_img1, highlight_img)), cv2.COLOR_BGR2RGB))); diff_texts.append(f"- Differences detected on page {i+1}.")
        if not diff_texts: return "No significant visual differences found.", None
        progress(0.9, desc="Generating AI analysis..."); pdf1_text, pdf2_text = "\n".join([c['text'] for c in self.file_data[pdf1_id]['chunks']]), "\n".join([c['text'] for c in self.file_data[pdf2_id]['chunks']])
        prompt = f"<|im_start|>system\nYou are an expert document analyst. Compare two documents and summarize key changes.<|im_end|><|im_start|>user\nDoc 1 ({pdf1_name}):\n{pdf1_text[:3500]}\n\nDoc 2 ({pdf2_name}):\n{pdf2_text[:3500]}\n\nAnalysis:<|im_end|><|im_start|>assistant\n"
        ai_analysis = llm_call(prompt, 512)
        return f"### AI Analysis of Differences\n\n{ai_analysis}\n\n---\n\n### Visual Change Summary\n\n" + "\n".join(diff_texts), diff_images

# ----- UI Construction with UX Improvements -----
assistant = ProPDFAssistant()

with gr.Blocks(fill_height=True, title="Pro PDF Assistant") as demo:
    file_state = gr.State({})
    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=450):
            gr.Markdown("# Pro PDF Assistant")
            with gr.Accordion("🗂️ Document Workflow & Controls", open=True):
                gr.Markdown("**Follow these steps to analyze your documents:**\n1. **Add PDFs** using the button below.\n2. **Check the boxes** next to the files you want to work with from the list.\n3. Click **Process Selected**. The status will change to `⏳ Processing...`.\n4. Once finished, the status will become `✅ Processed`.\n5. **Analyze!** Only processed files can be used. They will now appear in the dropdown lists on the right.")
                file_checkboxes = gr.CheckboxGroup(label="File Management List", type="value")
                with gr.Row():
                    upload_button = gr.UploadButton("Step 1: Add PDFs", file_types=[".pdf"], file_count="multiple", variant="secondary", size="sm")
                    process_button = gr.Button("Step 2: Process Selected", variant="primary")
                remove_button = gr.Button("Remove Selected Files", variant="stop", size="sm")
        with gr.Column(scale=5):
            with gr.Tabs():
                with gr.TabItem("💬 Chat"):
                    gr.Markdown("### Chat with Your Documents\nSelect one or more **processed** documents to begin a conversation.")
                    chat_selector = gr.Dropdown(label="Select PDFs to Chat With", multiselect=True, interactive=True)
                    chatbot = gr.Chatbot(label="Conversation", height=600, bubble_full_width=False, show_copy_button=True)
                    with gr.Row(): question_box = gr.Textbox(placeholder="Ask a follow-up question...", scale=5, container=False); ask_button = gr.Button("Ask", variant="primary", scale=1)
                    gr.ClearButton([question_box, chatbot], value="Clear Chat")
                with gr.TabItem("📄 Summary & Entities"):
                    gr.Markdown("### Single Document Analysis\nSelect a single **processed** document to view its AI-generated summary and extracted key entities.")
                    analysis_selector = gr.Dropdown(label="Select a Document to Analyze", interactive=True)
                    with gr.Row(): summary_output = gr.Markdown(elem_id="summary_output"); entity_output = gr.Markdown(elem_id="entity_output")
                with gr.TabItem("🕸️ Knowledge Graph"):
                    gr.Markdown("### Entity Relationship Graph\nThis graph visualizes the connections between entities found in the selected document.")
                    graph_output = gr.Image(label="Entity Relationship Graph", interactive=False)
                with gr.TabItem("↔️ Compare Documents"):
                    gr.Markdown("### Compare Two Documents\nSelect two **processed** documents from the lists below to find visual and textual differences.")
                    with gr.Row(): pdf1_selector = gr.Dropdown(label="Original Document (V1)", interactive=True); pdf2_selector = gr.Dropdown(label="New Document (V2)", interactive=True)
                    compare_button = gr.Button("Compare & Analyze Differences", variant="primary")
                    with gr.Row():
                        with gr.Column(scale=1): comparison_report = gr.Markdown()
                        with gr.Column(scale=1): comparison_gallery = gr.Gallery(label="Visual Differences", height=650, object_fit="contain")
                    gr.ClearButton([comparison_report, comparison_gallery, pdf1_selector, pdf2_selector], value="Clear Comparison")

    dummy_for_poll = gr.Textbox(visible=False, elem_id="dummy_for_poll")
    demo.load(None, None, dummy_for_poll, js="() => { setInterval(() => { const e = document.getElementById('dummy_for_poll'); if (e) { e.value = Math.random(); e.dispatchEvent(new Event('input')) } }, 2000) }")

    # This list defines all the UI components that need to be updated when the state changes.
    update_outputs = [file_checkboxes, chat_selector, analysis_selector, pdf1_selector, pdf2_selector]

    upload_button.upload(fn=assistant.add_files_to_state, inputs=[upload_button, file_state], outputs=[file_state])
    remove_button.click(fn=assistant.remove_files_from_state, inputs=[file_checkboxes, file_state], outputs=[file_state])
    process_button.click(fn=assistant.start_processing, inputs=[file_checkboxes, file_state], outputs=[file_state])

    # The master UI update trigger. This runs whenever the backend state changes for any reason.
    file_state.change(fn=assistant.get_ui_updates, inputs=[file_state], outputs=update_outputs)
    
    # The polling mechanism  just triggers the same master UI update function by changing the state.
    dummy_for_poll.input(lambda s: s, [file_state], [file_state])
    
    # Action handlers 
    ask_button.click(assistant.ask_question, [question_box, chatbot, chat_selector], [chatbot, question_box])
    analysis_selector.change(assistant.get_summary_and_entities, analysis_selector, [summary_output, entity_output, graph_output])
    compare_button.click(fn=assistant.compare_and_analyze, inputs=[pdf1_selector, pdf2_selector], outputs=[comparison_report, comparison_gallery], show_progress="full")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7865, share=True, inbrowser=True)
