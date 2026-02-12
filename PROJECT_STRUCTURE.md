# RAG_Rice Project Structure

Complete overview of the project's file and folder organization.

## 📁 Directory Structure

```
RAG_Rice/
│
├── config/                        # Configuration files
│   └── config.yaml               # Main system configuration
│
├── data/                          # Data storage
│   ├── pdfs/                      # Source PDF documents
│   │   ├── .gitkeep              
│   │   └── *.pdf                 # Agricultural manuals (not in git)
│   ├── processed/                 # Processed text data
│   │   ├── .gitkeep
│   │   └── *.txt                 # Extracted text (not in git)
│   ├── vector_store/              # ChromaDB vector database
│   │   ├── .gitkeep
│   │   └── chroma.sqlite3        # Vector database (not in git)
│   └── merged_dataset_final.csv   # Historical yield data (not in git)
│
├── logs/                          # Application logs
│   ├── .gitkeep
│   └── app.log                   # Runtime logs (not in git)
│
├── models/                        # Trained ML models
│   ├── .gitkeep
│   ├── rice_yield_model.pkl       # XGBoost model (not in git)
│   ├── region_encoder.pkl         # Label encoder (not in git)
│   └── rice_types.pkl             # Rice type encoder (not in git)
│
├── notebooks/                     # Jupyter notebooks
│   ├── .ipynb_checkpoints/        # Notebook checkpoints (not in git)
│   ├── data_exploration.ipynb     # Dataset analysis
│   ├── model_training.ipynb       # ML model development
│   ├── rag_testing.ipynb          # RAG system experiments
│   └── evaluation.ipynb           # System evaluation
│
├── rag_env/                       # Virtual environment (not in git)
│   ├── bin/ (or Scripts/ on Windows)
│   ├── lib/
│   └── ...
│
├── src/                           # Source code
│   ├── __init__.py               # Package initialization
│   ├── config.py                  # Configuration loader
│   ├── document_processor.py      # PDF processing & embeddings
│   ├── rag_system.py              # RAG implementation
│   ├── ml_integration.py          # ML model interface
│   ├── advanced_features.py       # Advanced RAG features
│   ├── evaluation_system.py       # Metrics & evaluation
│   └── utils.py                   # Utility functions (if exists)
│
├── static/                        # Web static assets
│   ├── css/
│   │   ├── main.css              # Main stylesheet
│   │   └── chat.css              # Chat-specific styles
│   ├── js/
│   │   ├── main.js               # Main JavaScript
│   │   ├── chat.js               # Chat functionality
│   │   └── prediction.js         # Prediction form logic
│   └── images/
│       ├── logo.png              # Application logo
│       ├── favicon.ico           # Browser icon
│       └── ...
│
├── templates/                     # Jinja2 HTML templates
│   ├── base.html                 # Base template
│   ├── index.html                # Landing page
│   ├── chat.html                 # Chat interface
│   ├── prediction.html           # Yield prediction page
│   ├── history.html              # Conversation history
│   └── error.html                # Error page
│
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── conftest.py               # Pytest configuration
│   ├── test_document_processor.py
│   ├── test_rag_system.py
│   ├── test_ml_integration.py
│   ├── test_api.py
│   └── test_app.py
│
├── .env                          # Environment variables (not in git)
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore rules
├── .gitkeep                      # Template for empty dirs
│
├── api.py                        # REST API implementation
├── app.py                        # Flask web application
│
├── requirements.txt              # Python dependencies
├── setup.sh                      # Linux/Mac setup script
├── setup.bat                     # Windows setup script
│
├── README.md                     # Main documentation
├── GETTING_STARTED.md            # Quick start guide
├── CONTRIBUTING.md               # Contribution guidelines
├── CHANGELOG.md                  # Version history
├── LICENSE                       # License information
└── PROJECT_STRUCTURE.md          # This file
```

---

## 📄 File Descriptions

### Root Level Files

| File | Purpose | Status |
|------|---------|--------|
| `api.py` | REST API endpoints for programmatic access | Required |
| `app.py` | Main Flask web application entry point | Required |
| `requirements.txt` | Python package dependencies | Required |
| `.env` | Environment variables (API keys, secrets) | Required (not in git) |
| `.env.example` | Template for environment variables | In git |
| `.gitignore` | Files to exclude from version control | In git |
| `setup.sh` | Automated setup script (Linux/Mac) | Optional |
| `setup.bat` | Automated setup script (Windows) | Optional |

### Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Main project documentation |
| `GETTING_STARTED.md` | Step-by-step setup guide |
| `CONTRIBUTING.md` | Contribution guidelines |
| `CHANGELOG.md` | Version history and changes |
| `LICENSE` | License information |
| `PROJECT_STRUCTURE.md` | This file - project organization |

### Configuration

| Path | Purpose |
|------|---------|
| `config/config.yaml` | Main system configuration (models, paths, settings) |

### Source Code (`src/`)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `__init__.py` | Package initialization | - |
| `config.py` | Load and validate configuration | `load_config()` |
| `document_processor.py` | Process PDFs and create embeddings | `process_documents()`, `create_embeddings()` |
| `rag_system.py` | Retrieval-Augmented Generation pipeline | `query()`, `retrieve_documents()`, `generate_response()` |
| `ml_integration.py` | Machine learning model interface | `predict_yield()`, `train_model()`, `load_model()` |
| `advanced_features.py` | Advanced RAG capabilities | `multi_query_expansion()`, `rerank_documents()` |
| `evaluation_system.py` | System metrics and evaluation | `evaluate_retrieval()`, `calculate_metrics()` |

### Web Application

#### Flask App (`app.py`)
- **Routes**: `/`, `/chat`, `/predict`, `/history`
- **Functions**: `index()`, `chat()`, `predict_yield()`, `history()`
- **Session Management**: User sessions, conversation history

#### API Server (`api.py`)
- **Endpoints**: `/api/chat`, `/api/predict`, `/api/health`
- **Authentication**: API key validation (if enabled)
- **Response Format**: JSON

#### Templates (`templates/`)
- **Base Template**: `base.html` - Common layout, navigation
- **Chat Interface**: `chat.html` - Real-time messaging UI
- **Prediction Form**: `prediction.html` - Input form for yield prediction
- **History**: `history.html` - Past conversations and predictions

#### Static Assets (`static/`)
- **CSS**: Stylesheets for UI design
- **JavaScript**: Client-side interactivity
- **Images**: Logos, icons, illustrations

### Data Files

| Directory | Contents | Tracked in Git? |
|-----------|----------|-----------------|
| `data/pdfs/` | Source PDF documents | ❌ No (too large) |
| `data/processed/` | Extracted text files | ❌ No |
| `data/vector_store/` | ChromaDB vector database | ❌ No (binary) |
| `data/*.csv` | Datasets (e.g., merged_dataset_final.csv) | ❌ No (large files) |

### Models

| File | Purpose | Size | Tracked? |
|------|---------|------|----------|
| `rice_yield_model.pkl` | XGBoost regression model | ~5-50 MB | ❌ No |
| `region_encoder.pkl` | Region label encoder | <1 MB | ❌ No |
| `rice_types.pkl` | Rice variety encoder | <1 MB | ❌ No |

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `data_exploration.ipynb` | Analyze dataset, visualize patterns |
| `model_training.ipynb` | Train and tune ML models |
| `rag_testing.ipynb` | Test RAG retrieval and generation |
| `evaluation.ipynb` | Evaluate system performance |

### Tests (`tests/`)

| Test File | Tests |
|-----------|-------|
| `test_document_processor.py` | PDF extraction, chunking, embeddings |
| `test_rag_system.py` | Document retrieval, response generation |
| `test_ml_integration.py` | Yield prediction, model loading |
| `test_api.py` | API endpoints, responses |
| `test_app.py` | Flask routes, templates |

---

## 🔄 Data Flow

### 1. Document Processing Flow
```
PDF Files (data/pdfs/)
    ↓
document_processor.py
    ↓
Text Chunks (data/processed/)
    ↓
Embeddings Generation
    ↓
Vector Database (data/vector_store/)
```

### 2. RAG Query Flow
```
User Query (web/API)
    ↓
app.py / api.py
    ↓
rag_system.py
    ↓
Vector Search (ChromaDB)
    ↓
LLM Generation (Groq)
    ↓
Response (web/API)
```

### 3. ML Prediction Flow
```
User Input (Region, Variety, etc.)
    ↓
app.py / api.py
    ↓
ml_integration.py
    ↓
Load Model (models/)
    ↓
Predict Yield
    ↓
Return Prediction
```

---

## 📦 Dependencies

Key packages used in the project:

### Core ML/AI
- `numpy`, `pandas` - Data manipulation
- `scikit-learn` - ML utilities
- `xgboost` - Gradient boosting

### LLM & RAG
- `groq` - LLM API client
- `langchain` - RAG framework
- `chromadb` - Vector database
- `sentence-transformers` - Embeddings

### Web Framework
- `flask` - Web application
- `flask-cors` - CORS support
- `flask-session` - Session management

### Document Processing
- `PyPDF2`, `pdfplumber` - PDF extraction

### Utilities
- `pyyaml` - Configuration
- `python-dotenv` - Environment variables

---

## 🔒 Security Considerations

### Files NOT in Git (Sensitive)
- `.env` - Contains API keys
- `data/` - May contain proprietary documents
- `models/` - Trained models (IP)
- `logs/` - May contain user data
- `rag_env/` - Large virtual environment

### Files IN Git (Public)
- Source code (`src/`, `app.py`, `api.py`)
- Configuration templates (`.env.example`)
- Documentation
- Tests

---

## 🚀 Deployment Considerations

### For Development
- Keep `DEBUG=true` in `.env`
- Use local virtual environment
- Access via `localhost:5000`

### For Production
- Set `DEBUG=false`
- Use production WSGI server (Gunicorn, uWSGI)
- Configure proper logging
- Set up reverse proxy (Nginx, Apache)
- Use environment variables for secrets
- Enable HTTPS

---

## 📝 Notes

- **Empty directories** use `.gitkeep` files to be tracked by Git
- **Large files** (models, data) are stored locally, not in Git
- **Virtual environment** is always excluded from version control
- **Notebooks** are included for development reference
- **Tests** help maintain code quality

---

For questions about the project structure, see the [CONTRIBUTING.md](CONTRIBUTING.md) guide or contact the development team.
