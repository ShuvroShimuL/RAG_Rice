#!/bin/bash
# RAG_Rice Setup Script
# This script automates the initial setup of the RAG_Rice project

echo "========================================"
echo "RAG_Rice Project Setup"
echo "========================================"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Create virtual environment
echo "[1/7] Creating virtual environment..."
if [ ! -d "rag_env" ]; then
    python3 -m venv rag_env
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "[2/7] Activating virtual environment..."
source rag_env/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "[3/7] Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ Pip upgraded"
echo ""

# Install requirements
echo "[4/7] Installing dependencies..."
echo "This may take a few minutes..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create .env file if it doesn't exist
echo "[5/7] Setting up environment variables..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✓ Created .env from .env.example"
        echo "⚠️  Please edit .env and add your Groq API key!"
    else
        echo "⚠️  .env.example not found. Please create .env manually."
    fi
else
    echo "✓ .env file already exists"
fi
echo ""

# Create necessary directories
echo "[6/7] Creating project directories..."
mkdir -p data/pdfs
mkdir -p data/processed
mkdir -p data/vector_store
mkdir -p models
mkdir -p logs
mkdir -p notebooks
mkdir -p static/css
mkdir -p static/js
mkdir -p static/images
mkdir -p templates
mkdir -p tests

# Create .gitkeep files
touch data/pdfs/.gitkeep
touch data/processed/.gitkeep
touch data/vector_store/.gitkeep
touch models/.gitkeep
touch logs/.gitkeep

echo "✓ Directories created"
echo ""

# Copy config file
echo "[7/7] Setting up configuration..."
if [ ! -f "config/config.yaml" ]; then
    mkdir -p config
    if [ -f "config.yaml" ]; then
        cp config.yaml config/config.yaml
        echo "✓ Configuration file copied to config/"
    else
        echo "⚠️  config.yaml not found. Please add it to config/ directory."
    fi
else
    echo "✓ Configuration file already exists"
fi
echo ""

# Final instructions
echo "========================================"
echo "Setup Complete! ✨"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your Groq API key"
echo "2. Add your agricultural PDF documents to data/pdfs/"
echo "3. Run: python src/document_processor.py"
echo "4. Run: python src/ml_integration.py"
echo "5. Start the app: python app.py"
echo ""
echo "For development, activate the virtual environment:"
echo "  source rag_env/bin/activate"
echo ""
echo "To start the application:"
echo "  python app.py"
echo ""
echo "Happy farming! 🌾"
