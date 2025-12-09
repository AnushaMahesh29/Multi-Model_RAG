# Multi-Modal RAG QA System

A RAG system that processes PDF documents with text, tables, and images, then answers questions using LLM.

## Features

- Extracts text, tables, and images from PDFs
- OCR for images using Tesseract
- Vector search with FAISS
- Answer generation via Groq API
- Streamlit web UI and CLI

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Setup

1. Get Groq API key from https://console.groq.com/

2. Set environment variable:
```bash
set GROQ_API_KEY=your-api-key-here
```

## Usage

**Run Streamlit App:**
```bash
streamlit run src/app/streamlit_app.py
```

**Run CLI:**
```bash
python src/app/cli_app.py
```

## Tech Stack

- PyMuPDF, Camelot - PDF processing
- Tesseract - OCR
- SentenceTransformers, CLIP - Embeddings
- FAISS - Vector search
- Groq - LLM API
- Streamlit - Web UI
