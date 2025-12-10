#!/bin/bash

echo "Starting Multi-Modal RAG QA System..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found!"
    echo "Please run: python -m venv venv"
    echo "Then: source venv/bin/activate"
    echo "Then: pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if GROQ_API_KEY is set
if [ -z "$GROQ_API_KEY" ]; then
    echo "WARNING: GROQ_API_KEY environment variable not set!"
    echo "You can set it in the Streamlit sidebar or run:"
    echo "export GROQ_API_KEY='your-api-key-here'"
    echo ""
fi

# Create required directories
mkdir -p data/raw
mkdir -p data/intermediate/images
mkdir -p data/index

# Run Streamlit
echo "Starting Streamlit app..."
streamlit run src/app/streamlit_app.py
