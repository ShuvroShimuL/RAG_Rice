# Getting Started with RAG_Rice

Welcome to RAG_Rice! This guide will help you set up and run the agricultural chatbot system.

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed ([Download Python](https://www.python.org/downloads/))
- **Git** installed ([Download Git](https://git-scm.com/downloads))
- **Groq API Key** ([Get one free](https://console.groq.com/))
- **4GB+ RAM** recommended
- **Internet connection** for API calls

## 🚀 Quick Start (5 minutes)

### Option 1: Automated Setup (Recommended)

**On Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**On Windows:**
```bash
setup.bat
```

Then follow the on-screen instructions!

### Option 2: Manual Setup

Follow the detailed steps below for manual configuration.

---

## 📦 Step-by-Step Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/RAG_Rice.git
cd RAG_Rice
```

### Step 2: Create Virtual Environment

**Linux/Mac:**
```bash
python3 -m venv rag_env
source rag_env/bin/activate
```

**Windows:**
```bash
python -m venv rag_env
rag_env\Scripts\activate
```

You should see `(rag_env)` in your terminal prompt.

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install all required packages including Flask, Groq, LangChain, ChromaDB, etc.

### Step 4: Configure Environment Variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your favorite text editor:
   ```bash
   nano .env  # or use notepad, vim, code, etc.
   ```

3. Add your Groq API key:
   ```env
   GROQ_API_KEY=your_actual_groq_api_key_here
   ```

4. Save and close the file.

### Step 5: Set Up Configuration

1. Ensure `config/config.yaml` exists with proper settings
2. Review and modify settings as needed:
   ```bash
   nano config/config.yaml
   ```

### Step 6: Prepare Data Directories

Create necessary directories (if not already created):
```bash
mkdir -p data/pdfs data/processed data/vector_store
mkdir -p models logs notebooks
```

### Step 7: Add Agricultural Documents

Place your agricultural PDF documents in the `data/pdfs/` folder:
```bash
cp /path/to/your/pdfs/*.pdf data/pdfs/
```

### Step 8: Process Documents

Generate embeddings from your PDF documents:
```bash
python src/document_processor.py
```

This will:
- Extract text from PDFs
- Create text chunks
- Generate embeddings
- Store in vector database

Expected output:
```
Processing documents...
✓ Processed: rice_cultivation_guide.pdf
✓ Processed: brri_varieties.pdf
✓ Created 450 embeddings
✓ Saved to vector database
```

### Step 9: Train ML Models

Train or load the yield prediction models:
```bash
python src/ml_integration.py
```

This will:
- Load historical dataset
- Train XGBoost model
- Save trained models

Expected output:
```
Loading dataset...
✓ Loaded 5000 records
Training model...
✓ Model trained (R² = 0.85)
✓ Saved to models/rice_yield_model.pkl
```

### Step 10: Start the Application

Launch the Flask web application:
```bash
python app.py
```

Expected output:
```
 * Running on http://0.0.0.0:5000/
 * Running on http://127.0.0.1:5000/
```

### Step 11: Access the Application

Open your web browser and navigate to:
```
http://localhost:5000
```

You should see the RAG_Rice web interface!

---

## 🎯 First-Time Usage

### Using the Chat Interface

1. Click on the **"Chat"** tab
2. Type your question, for example:
   ```
   What is the best time to plant Boro rice?
   ```
3. Press Enter or click "Send"
4. Wait for the AI-generated response

### Using Yield Prediction

1. Click on the **"Predict"** tab
2. Fill in the form:
   - **Region**: Select your region (e.g., Dhaka)
   - **Rice Variety**: Choose variety (e.g., BRRI dhan28)
   - **Season**: Select season (Boro, Aus, Aman)
   - **Temperature**: Enter average temperature (°C)
   - **Rainfall**: Enter rainfall amount (mm)
3. Click "Predict Yield"
4. View the predicted yield result

---

## 🔧 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'groq'"

**Solution:** Activate virtual environment and install requirements:
```bash
source rag_env/bin/activate  # or rag_env\Scripts\activate on Windows
pip install -r requirements.txt
```

### Problem: "GROQ_API_KEY not found"

**Solution:** 
1. Check that `.env` file exists in project root
2. Verify it contains `GROQ_API_KEY=your_key_here`
3. Restart the application

### Problem: "No documents found in vector database"

**Solution:** Run document processing:
```bash
python src/document_processor.py
```

### Problem: "Address already in use"

**Solution:** Another application is using port 5000. Either:
1. Stop the other application, or
2. Change port in `config/config.yaml`:
   ```yaml
   flask:
     port: 8080  # Use a different port
   ```

### Problem: Can't find config.yaml

**Solution:** Ensure `config/config.yaml` exists:
```bash
cp config.yaml config/config.yaml  # If config.yaml is in root
```

---

## 📚 Next Steps

Once everything is running:

1. **Explore Notebooks** - Open Jupyter notebooks for data analysis:
   ```bash
   jupyter notebook
   ```

2. **Run Tests** - Verify system functionality:
   ```bash
   pytest tests/
   ```

3. **Read Documentation** - Check README.md for detailed information

4. **Customize** - Modify templates, add features, improve models

5. **Contribute** - See CONTRIBUTING.md for guidelines

---

## 🆘 Getting Help

If you encounter issues:

1. **Check logs**: Look in `logs/app.log` for error messages
2. **GitHub Issues**: Open an issue on the repository
3. **Contact**: Email the development team
   - Student: shamimul.shimul@northsouth.edu
   - Supervisor: shahnewaz.siddique@northsouth.edu

---

## 📝 Quick Reference

### Activate Virtual Environment
```bash
# Linux/Mac
source rag_env/bin/activate

# Windows
rag_env\Scripts\activate
```

### Run Application
```bash
python app.py
```

### Run API Server
```bash
python api.py
```

### Process Documents
```bash
python src/document_processor.py
```

### Train Models
```bash
python src/ml_integration.py
```

### Run Tests
```bash
pytest tests/
```

### Deactivate Virtual Environment
```bash
deactivate
```

---

## 🎓 Learning Resources

- **Flask Documentation**: https://flask.palletsprojects.com/
- **Groq API Docs**: https://console.groq.com/docs
- **LangChain Docs**: https://python.langchain.com/
- **XGBoost Tutorial**: https://xgboost.readthedocs.io/

---

**Congratulations! 🎉 You're now ready to use RAG_Rice!**

Happy farming! 🌾
