@echo off
echo Installing packages in Python 3.12 virtual environment...
echo This may take 10-15 minutes due to large packages like torch and scipy.
echo.

venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements_exact.txt

echo.
echo Installation complete!
echo.
echo To run the Streamlit app, use:
echo venv\Scripts\streamlit.exe run src/app/streamlit_app.py
pause
