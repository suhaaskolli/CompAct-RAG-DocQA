# Doc-QA + CompAct Pipeline

A fast, modern, and robust document question-answering pipeline with optional context compaction, designed for Apple Silicon and quantized LLMs (GGUF format).

---

## Overview
This project lets you:
- Upload or index PDFs and text files
- Build a FAISS index for fast retrieval
- Ask questions and get answers from a local LLM (llama.cpp/llama-cpp-python)
- **Compact** retrieved context with a summarizer LLM before answering
- Run everything locally, optimized for Apple Silicon

---

## Features
- **PDF & text support**
- **Fast retrieval** with FAISS and sentence-transformers
- **Quantized LLMs** (GGUF) for efficient inference
- **Compaction**: Summarize context before answering for better results on long docs
- **Flask web app**: Upload a PDF, ask multiple questions, choose compaction
- **CLI scripts** for batch and automated use

---

## About the CompAct Model

**CompAct** is a specialized LLM trained to summarize and compress large retrieved contexts into concise, information-rich summaries. This helps the answer LLM focus on the most relevant information, especially for long or complex documents.

- **When to use:**
  - Enable compaction for long PDFs, multi-document queries, or when you want more focused answers.
  - Disable compaction for short/simple documents or when you want to see the full retrieved context.

---

## Requirements
- Python 3.10
- Apple Silicon (M1/M2/M3) recommended for best performance
- Also works on Intel Macs, Windows, and Linux
- At least 8GB RAM (16GB+ recommended for 8B models)
- macOS 12+, Windows 10+, or modern Linux

> On non-Apple hardware, LLM inference may be slower unless you have a fast CPU or compatible GPU (CUDA/AVX2/AVX512).

---

## Installation
1. **Clone the repo:**
   ```bash
   git clone https://github.com/suhaaskolli/CompAct-RAG-DocQA.git
   cd docqa4
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download a quantized LLaMA model**:
   - Download a quantized GGUF version of Llama (e.g., `llama-3-8b-instruct.Q4_K_M.gguf`) from [Llama on Hugging Face](https://huggingface.co/NoelJacob/Meta-Llama-3-8B-Instruct-Q4_K_M-GGUF) or your preferred source.
   - Place it in `models/llama-3-8b-instruct.Q4_K_M.gguf`

5. **Download the CompAct compaction model**:
   - Download a quantized GGUF version of CompAct (e.g., `CompAct-7b.Q4_K_S.gguf`) from [CompAct on Hugging Face](https://huggingface.co/mradermacher/CompAct-7b-GGUF) or your preferred source.
   - Place it in `models/compact7b/CompAct-7b.Q4_K_S.gguf`

---

## Quickstart

### **Place all the PDFs you want into the data/ folder**

### **Index all PDFs in a folder:**
```bash
python build_index.py --docs data/*.pdf --index-dir index
```

### **Ask a question (no compaction):**
```bash
python qa_compact.py --question "What is the capital of France?" --index-dir index --model-path models/llama-3-8b-instruct.Q4_K_M.gguf
```

### **Ask a question (with compaction):**
```bash
python qa_compact.py --question "Who are the main characters in Of Mice and Men?" --index-dir index --model-path models/llama-3-8b-instruct.Q4_K_M.gguf --compact-model models/compact7b/CompAct-7b.Q4_K_S.gguf
```

---

## Flask Web App
1. **Start the app:**
   ```bash
   python app.py
   ```
2. **Open your browser:**
   - Go to [http://localhost:5000](http://localhost:5000)
   - Upload a PDF, ask questions, and choose whether to use compaction

---

## Testing
Run all tests:
```bash
pytest -v tests/test_pipeline.py
```
Run a specific test:
```bash
pytest -v tests/test_pipeline.py::test_summary_budget
```

---

## Model Notes
- **Quantized GGUF models** are required (e.g., Llama-3 8B Q4_K_M)
- Download from Hugging Face and place in the `models/` directory
- Compaction model is optional; if not provided, context is passed directly to the answer LLM

---

## Troubleshooting
- **Model not found:** Check your `--model-path` and file location
- **RAM errors:** Use a smaller model or quantization (Q4_K_M, Q4_K_S)
- **Slow inference:** Quantized models are much faster
- **CompAct folder missing:** No longer needed; all logic is in `qa_compact.py` and `app.py`
- **DeprecationWarnings:** These are from faiss/numpy and can be ignored

---

## Credits
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [sentence-transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [CompAct](https://github.com/dmis-lab/CompAct)
- [pypdf](https://pypdf.readthedocs.io/)
- [Bootstrap](https://getbootstrap.com/) (for Flask UI)

---

## License

This project is licensed under the [MIT License](LICENSE).

---