#🚀 Advanced PDF Assistant with GPU Acceleration
Using llama cpp (for multiple pdfs)

A powerful AI-powered PDF assistant that leverages GPU acceleration for fast document processing, summarization, question-answering, and comparison. Built with Gradio for an intuitive web interface and powered by local LLMs via llama-cpp-python.

## ✨ Features

- **📤 Multi-PDF Upload & Processing**: Upload and process multiple PDFs simultaneously
- **🧠 AI-Powered Summarization**: Generate concise summaries of your documents
- **💬 Intelligent Q&A**: Ask questions about your documents and get contextual answers
- **🔍 PDF Comparison**: Compare two PDFs and highlight differences
- **⚡ GPU Acceleration**: Automatic GPU detection and acceleration for faster processing
- **🔄 OCR Support**: Fallback OCR for scanned documents using Tesseract
- **📊 System Monitoring**: Real-time system resource monitoring
- **🎯 Efficient Caching**: Smart caching system for improved performance

## 🛠️ Installation

### Prerequisites

#### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils
```

**macOS:**
```bash
brew install tesseract poppler
```

**Windows:**
- Download Tesseract from: [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- Download Poppler from: [oschwartz10612/poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases/)

### Python Dependencies

1. **Clone the repository:**
```bash
git clone https://github.com/M-e-h-18/Offline-pdf-chatbot-and-analyser.git
cd Offline-pdf-chatbot-and-analyser
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### GPU Support (Optional but Recommended)

For NVIDIA GPU acceleration:

1. **Install CUDA-enabled PyTorch:**
```bash
pip install torch>=2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

2. **Install llama-cpp-python with CUDA support:**
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

## 🚀 Usage

### Starting the Application

```bash
python app.py
```

The application will:
1. Automatically download required AI models on first run
2. Launch a web interface at `http://localhost:7860`
3. Display system information including GPU availability

### Using the Interface

#### 1. Upload PDFs
- Go to the "Upload PDFs" tab
- Select multiple PDF files
- Click "Process PDFs" to extract text and prepare for analysis

#### 2. Generate Summaries
- Navigate to the "Summarize" tab
- Select PDFs you want to summarize
- Click "Generate Summary" for AI-powered summaries

#### 3. Ask Questions
- Use the "Chatbot" tab
- Select relevant PDFs
- Ask questions about the content
- Get contextual answers based on document content

#### 4. Compare PDFs
- Go to "Compare PDFs" tab
- Select two different PDFs
- View highlighted differences between documents

## ⚙️ Configuration

The application can be configured by modifying the `Config` class in `app.py`:

```python
class Config:
    MAX_TOKENS_CHUNK = 1000      # Chunk size for processing
    OCR_DPI = 200                # OCR resolution
    OCR_LANG = 'eng'             # Tesseract language
    EMBED_BATCH_SIZE = 4         # Embedding batch size
    LLM_N_CTX = 2048            # Context window size
    CACHE_SIZE = 5               # Number of PDFs to cache
    GPU_LAYERS = -1              # GPU layers (-1 = all)
    N_THREADS = 4                # CPU threads
```

## 🔧 Models

The application automatically downloads these models on first run:

- **LLM Model**: Hermes-2-Pro-Mistral-7B (Quantized)
- **Embedding Model**: BGE-Large-EN-v1.5 (f16)

Models are stored in the `models/` directory and only downloaded once.

## 📊 System Requirements

### Minimum Requirements
- **RAM**: 8GB
- **Storage**: 10GB free space (for models)
- **CPU**: 4+ cores recommended

### Recommended for GPU Acceleration
- **GPU**: NVIDIA GPU with 6GB+ VRAM
- **RAM**: 16GB+
- **CUDA**: 11.8 or higher

## 🐛 Troubleshooting

### Common Issues

1. **Models not downloading:**
   - Check internet connection
   - Ensure sufficient disk space
   - Check `models/` directory permissions

2. **OCR not working:**
   - Verify Tesseract installation: `tesseract --version`
   - Install additional language packs if needed

3. **GPU not detected:**
   - Install CUDA toolkit
   - Verify GPU compatibility: `nvidia-smi`
   - Reinstall PyTorch with CUDA support

4. **Out of memory errors:**
   - Reduce `EMBED_BATCH_SIZE` in config
   - Process fewer PDFs simultaneously
   - Reduce `LLM_N_CTX` size

### Performance Tips

- **Use GPU acceleration** for 3-5x speed improvement
- **Process PDFs in batches** for large collections
- **Clear resources** regularly using the clear button
- **Monitor system resources** via the system info panel

## 📝 Logging

The application logs to both console and `pdf_assistant.log` file. Check logs for debugging information.



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Gradio](https://gradio.app/) for the web interface
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for LLM inference
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF processing
- [Tesseract](https://github.com/tesseract-ocr/tesseract) for OCR capabilities
- [Hugging Face](https://huggingface.co/) for model hosting




**Note**: This application processes PDFs locally on your machine. No data is sent to external servers, ensuring your document privacy and security.
