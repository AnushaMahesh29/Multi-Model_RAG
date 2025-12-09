@echo off
echo Starting Multi-Modal RAG QA System...
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then: venv\Scripts\activate
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate

REM Check if GROQ_API_KEY is set
if "%GROQ_API_KEY%"=="" (
    echo WARNING: GROQ_API_KEY environment variable not set!
    echo You can set it in the Streamlit sidebar or run:
    echo set GROQ_API_KEY=your-api-key-here
    echo.
)

REM Create required directories
if not exist "data\raw" mkdir data\raw
if not exist "data\intermediate\images" mkdir data\intermediate\images
if not exist "data\index" mkdir data\index

REM Run Streamlit
echo Starting Streamlit app...
streamlit run src\app\streamlit_app.py

pause
