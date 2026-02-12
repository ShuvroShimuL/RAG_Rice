@echo off
REM RAG_Rice Setup Script for Windows
REM This script automates the initial setup of the RAG_Rice project

echo ========================================
echo RAG_Rice Project Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo X Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Create virtual environment
echo [1/7] Creating virtual environment...
if not exist "rag_env" (
    python -m venv rag_env
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)
echo.

REM Activate virtual environment
echo [2/7] Activating virtual environment...
call rag_env\Scripts\activate.bat
echo [OK] Virtual environment activated
echo.

REM Upgrade pip
echo [3/7] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo [OK] Pip upgraded
echo.

REM Install requirements
echo [4/7] Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt
echo [OK] Dependencies installed
echo.

REM Create .env file if it doesn't exist
echo [5/7] Setting up environment variables...
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo [OK] Created .env from .env.example
        echo [!] Please edit .env and add your Groq API key!
    ) else (
        echo [!] .env.example not found. Please create .env manually.
    )
) else (
    echo [OK] .env file already exists
)
echo.

REM Create necessary directories
echo [6/7] Creating project directories...
if not exist "data\pdfs" mkdir data\pdfs
if not exist "data\processed" mkdir data\processed
if not exist "data\vector_store" mkdir data\vector_store
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "notebooks" mkdir notebooks
if not exist "static\css" mkdir static\css
if not exist "static\js" mkdir static\js
if not exist "static\images" mkdir static\images
if not exist "templates" mkdir templates
if not exist "tests" mkdir tests

REM Create .gitkeep files
echo. > data\pdfs\.gitkeep
echo. > data\processed\.gitkeep
echo. > data\vector_store\.gitkeep
echo. > models\.gitkeep
echo. > logs\.gitkeep

echo [OK] Directories created
echo.

REM Copy config file
echo [7/7] Setting up configuration...
if not exist "config\config.yaml" (
    if not exist "config" mkdir config
    if exist "config.yaml" (
        copy config.yaml config\config.yaml >nul
        echo [OK] Configuration file copied to config\
    ) else (
        echo [!] config.yaml not found. Please add it to config\ directory.
    )
) else (
    echo [OK] Configuration file already exists
)
echo.

REM Final instructions
echo ========================================
echo Setup Complete! ✨
echo ========================================
echo.
echo Next steps:
echo 1. Edit .env and add your Groq API key
echo 2. Add your agricultural PDF documents to data\pdfs\
echo 3. Run: python src\document_processor.py
echo 4. Run: python src\ml_integration.py
echo 5. Start the app: python app.py
echo.
echo For development, activate the virtual environment:
echo   rag_env\Scripts\activate
echo.
echo To start the application:
echo   python app.py
echo.
echo Happy farming! 🌾
echo.
pause
