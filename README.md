# RAG_Rice: Context-Aware Agricultural Chatbot

**Integrating Document Retrieval with Predictive Analytics for Rice Farming**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-Academic-green)
![Status](https://img.shields.io/badge/Status-Research%20Project-orange)

## 📋 Overview

RAG_Rice is an intelligent agricultural advisory system that combines **Retrieval-Augmented Generation (RAG)** technology with **machine learning-based yield prediction** to provide comprehensive rice cultivation guidance. The system transforms domain-specific agricultural manuals into searchable knowledge and integrates real-time predictive analytics to deliver evidence-based, personalized farming advice.

### Key Features

- 🤖 **AI-Powered Q&A**: Natural language interface for farmer inquiries
- 📚 **Document-Grounded Responses**: Retrieves accurate information from agricultural manuals
- 📊 **Yield Prediction**: XGBoost-based forecasting using historical climate and crop data
- 🌾 **Rice Variety Support**: Specialized knowledge for different rice types
- 🎯 **Context-Aware**: Delivers personalized advice based on farmer context
- 💻 **Web Interface**: User-friendly platform for conversational interaction

## 🎓 Academic Context

This project is part of a Directed Research at the **Department of Electrical and Computer Engineering, North South University**, supervised by **Dr. Shahnewaz Siddique**.

**Research Focus**: Addressing critical information gaps in rural agriculture by providing farmers with immediate access to evidence-based agricultural advice combined with predictive analytics.

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────┐
│               User Interface (Web)              │
└──────────────────────┬──────────────────────────┘
                       │
       ┌───────────────┴──────────────┐
       │                              │
┌──────▼──────────┐          ┌────────▼─────────┐
│   RAG System    │          │  ML Integration  │
│                 │          │                  │
│ • LLM (LLaMA)   │          │ • XGBoost Model  │
│ • Vector DB     │          │ • Yield Predict  │
│ • Embeddings    │          │ • Rice Varieties │
└──────┬──────────┘          └────────┬─────────┘
       │                              │
┌──────▼──────────┐          ┌────────▼─────────┐
│  Document       │          │   Historical     │
│  Processing     │          │   Dataset        │
│                 │          │                  │
│ • PDF Parser    │          │ • Climate Data   │
│ • Chunking      │          │ • Yield Records  │
│ • Embeddings    │          │ • Region Info    │
└─────────────────┘          └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- Internet connection (for Groq API access)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/RAG_Rice.git
   cd RAG_Rice
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

   Get your Groq API key from: https://console.groq.com/

5. **Prepare the data directories**
   ```bash
   mkdir -p data/pdfs
   mkdir -p data/processed
   mkdir -p data/vector_store
   ```

6. **Add your agricultural PDF documents**
   
   Place your agricultural manuals and documents in `data/pdfs/`

## 📖 Usage

### Running the System

#### Method 1: Web Application (Recommended)

1. **Start the Flask application**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   - Open your browser and navigate to: `http://localhost:5000`
   - You'll see the main interface with:
     - Chat interface for asking questions
     - Yield prediction form
     - Document management

3. **Using the Web Interface**
   - **Chat Tab**: Ask natural language questions about rice cultivation
   - **Predict Tab**: Enter parameters for yield prediction
   - **History Tab**: View past conversations and predictions

#### Method 2: REST API

1. **Start the API server**
   ```bash
   python api.py
   ```

2. **API Endpoints**

   **Chat Endpoint:**
   ```bash
   curl -X POST http://localhost:5000/api/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the best planting time for Boro rice?"}'
   ```

   **Yield Prediction Endpoint:**
   ```bash
   curl -X POST http://localhost:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{
       "region": "Dhaka",
       "variety": "BRRI dhan28",
       "season": "Boro",
       "temperature": 25,
       "rainfall": 150
     }'
   ```

#### Method 3: Python Scripts

For development and testing:

**Process Documents:**
```bash
python src/document_processor.py
```

**Train ML Models:**
```bash
python src/ml_integration.py
```

**Test RAG System:**
```bash
python src/rag_system.py
```

### Example Interactions

**Q: "What is the best planting time for Boro rice?"**
- System retrieves relevant sections from agricultural manuals
- Provides season-specific recommendations
- Considers regional climate patterns

**Q: "Predict yield for BRRI dhan28 in Dhaka region"**
- Uses XGBoost model with historical data
- Considers variety-specific characteristics
- Provides yield forecast with confidence metrics

### Jupyter Notebooks

Explore the system interactively:

```bash
jupyter notebook
```

Then open:
- `notebooks/data_exploration.ipynb` - Analyze datasets
- `notebooks/model_training.ipynb` - Train and evaluate ML models
- `notebooks/rag_testing.ipynb` - Test RAG components

## 📂 Project Structure

```
RAG_Rice/
│
├── config/                        # Configuration files
│   └── config.yaml               # Main configuration
│
├── data/                          # Data directory
│   ├── pdfs/                      # Agricultural PDF documents
│   ├── processed/                 # Processed documents
│   ├── vector_store/              # Vector embeddings database
│   └── merged_dataset_final.csv   # Historical yield dataset
│
├── logs/                          # Application logs
│   └── app.log                   # Runtime logs
│
├── models/                        # Trained ML models
│   ├── rice_yield_model.pkl       # XGBoost yield prediction model
│   ├── region_encoder.pkl         # Region label encoder
│   └── rice_types.pkl             # Rice variety encoder
│
├── notebooks/                     # Jupyter notebooks
│   ├── data_exploration.ipynb     # Dataset analysis
│   ├── model_training.ipynb       # ML model development
│   └── rag_testing.ipynb          # RAG system experiments
│
├── rag_env/                       # Virtual environment (git ignored)
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── config.py                  # Configuration loader
│   ├── document_processor.py      # PDF processing & embedding
│   ├── rag_system.py              # RAG implementation
│   ├── ml_integration.py          # ML model integration
│   ├── advanced_features.py       # Advanced features
│   └── evaluation_system.py       # System evaluation
│
├── static/                        # Web static files
│   ├── css/                       # Stylesheets
│   ├── js/                        # JavaScript files
│   └── images/                    # Image assets
│
├── templates/                     # HTML templates
│   ├── index.html                 # Main page
│   ├── chat.html                  # Chat interface
│   └── prediction.html            # Yield prediction page
│
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── test_rag_system.py         # RAG tests
│   └── test_ml_integration.py     # ML tests
│
├── .env                          # Environment variables
├── .gitignore                    # Git ignore rules
├── api.py                        # REST API endpoints
├── app.py                        # Flask web application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── LICENSE                       # License file
```

## ⚙️ Configuration

The system is configured via `config/config.yaml`:

```yaml
models:
  llm_model: "llama-3.1-8b-instant"        # Groq LLM model
  embedding_model: "all-MiniLM-L6-v2"      # Sentence transformer

vector_db:
  collection_name: "rice_advisory"          # Chroma collection name
  persist_directory: "./data/vector_store"  # Vector DB location

document_processing:
  chunk_size: 1000                          # Text chunk size
  chunk_overlap: 200                        # Overlap between chunks

data:
  pdfs_directory: "./data/pdfs"
  processed_directory: "./data/processed"
  ml_dataset: "./data/merged_dataset_final.csv"

ml_models:
  yield_prediction: "./models/rice_yield_model.pkl"
  region_encoder: "./models/region_encoder.pkl"
  rice_types: "./models/rice_types.pkl"

groq:
  temperature: 0.3                          # LLM creativity (0-1)
  max_tokens: 1000                          # Max response length

flask:
  host: "0.0.0.0"                          # Server host
  port: 5000                                # Server port
  debug: true                               # Debug mode
```

### Customization

- **Change LLM Model**: Modify `models.llm_model` (see Groq documentation)
- **Adjust Chunk Size**: Modify `document_processing.chunk_size` for longer/shorter contexts
- **Temperature**: Lower for factual responses, higher for creative answers
- **Flask Settings**: Modify `flask` section for server configuration

## 🔧 Components

### 1. Web Application (`app.py`)
- Flask-based web server
- User-friendly chat interface
- Yield prediction form
- Session management
- Real-time response streaming

### 2. REST API (`api.py`)
- RESTful API endpoints
- JSON request/response handling
- Authentication (if implemented)
- CORS support for cross-origin requests
- Rate limiting capabilities

### 3. Document Processor (`src/document_processor.py`)
- Extracts text from agricultural PDF manuals
- Splits documents into semantic chunks
- Generates embeddings using sentence transformers
- Stores vectors in ChromaDB for efficient retrieval

### 4. RAG System (`src/rag_system.py`)
- Implements Retrieval-Augmented Generation pipeline
- Retrieves relevant document chunks based on query
- Generates context-aware responses using LLaMA 3.1
- Integrates with Groq API for fast inference

### 5. ML Integration (`src/ml_integration.py`)
- XGBoost regression model for yield prediction
- Features: region, rice variety, climate data, historical yields
- Model persistence and loading
- Prediction API for real-time forecasting

### 6. Advanced Features (`src/advanced_features.py`)
- Multi-query expansion for better retrieval
- Re-ranking of retrieved documents
- Conversation history management
- Response synthesis optimization

### 7. Evaluation System (`src/evaluation_system.py`)
- Retrieval accuracy metrics
- Response quality assessment
- System performance benchmarking
- A/B testing framework

### 8. Frontend Components
- **Templates** (`templates/`): HTML templates using Jinja2
- **Static Assets** (`static/`): CSS, JavaScript, images
- **Interactive UI**: Real-time chat, forms, visualizations

## 📊 Dataset

The system uses a comprehensive historical dataset (`merged_dataset_final.csv`) containing:

- **Yield Records**: Historical crop yields across seasons
- **Climate Variables**: Temperature, rainfall, humidity patterns
- **Rice Varieties**: BRRI varieties and local cultivars
- **Regional Data**: District-wise agricultural statistics
- **Temporal Data**: Multi-year seasonal patterns

### Dataset Format
```csv
Year, Season, District, Variety, Temperature, Rainfall, Yield, ...
```

## 🔍 Technical Details

### RAG Pipeline

1. **Query Processing**
   - User query → Embedding generation
   - Semantic search in vector database
   - Retrieve top-k relevant chunks

2. **Context Assembly**
   - Combine retrieved chunks
   - Add system instructions
   - Format for LLM consumption

3. **Response Generation**
   - LLM (LLaMA 3.1) generates response
   - Grounds answer in retrieved documents
   - Maintains conversational context

### ML Model

- **Algorithm**: XGBoost Regressor
- **Input Features**: Region, variety, climate, historical yield
- **Target**: Rice yield (tons/hectare)
- **Evaluation Metrics**: MAE, RMSE, R²
- **Cross-validation**: 5-fold stratified

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

Run evaluation:
```bash
python evaluation_system.py
```

## 📈 Performance Metrics

- **Retrieval Precision**: Document relevance accuracy
- **Response Quality**: Factual correctness and coherence
- **Prediction Accuracy**: Yield forecast MAE/RMSE
- **Inference Speed**: Average response time
- **System Availability**: Uptime and reliability

## 🤝 Contributing

This is an academic research project. For questions or collaboration:

1. Contact the research team
2. Submit issues via GitHub
3. Follow academic citation guidelines

## 📄 License

This project is developed for academic research purposes at North South University.

**Academic Use**: Free for educational and research purposes  
**Commercial Use**: Requires permission from the authors and institution

## 📞 Contact

**Student Researcher**: M Shamimul Haque Mondal Shimul  
**Student ID**: 2122085642  
**Email**: shamimul.shimul@northsouth.edu

**Faculty Supervisor**: Dr. Shahnewaz Siddique  
**Department**: Electrical and Computer Engineering  
**Institution**: North South University, Dhaka, Bangladesh

## 🙏 Acknowledgments

Special thanks to:
- **Dr. Shahnewaz Siddique** - Project Supervisor

## 📚 References

This project builds upon:
- **Groq API**: Fast LLM inference
- **LangChain**: RAG framework components
- **ChromaDB**: Vector database
- **Sentence Transformers**: Document embeddings
- **XGBoost**: Gradient boosting framework
- **BRRI**: Bangladesh Rice Research Institute data

## 🔮 Future Work

- [ ] Multilingual support (Bengali language interface)
- [ ] Mobile application development
- [ ] Real-time weather API integration
- [ ] Satellite imagery analysis
- [ ] Farmer community features
- [ ] Offline mode for rural areas
- [ ] Voice-based interaction
- [ ] SMS gateway integration

## 📊 Research Publications

*If applicable, list publications, conference presentations, or papers related to this work*

---

## 🚨 Important Notes

⚠️ **API Key Security**: Never commit your `.env` file with API keys  
⚠️ **Data Privacy**: Ensure farmer data is handled according to privacy regulations  
⚠️ **Model Updates**: Regularly update models with new agricultural research  
⚠️ **Accuracy**: Always validate predictions with domain experts

---

**Version**: 1.0  
**Last Updated**: January 2025  
**Project Status**: Active Research

---

### Quick Commands Reference

```bash
# Setup
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
pip install -r requirements.txt

# Configuration
cp .env.example .env
# Edit .env with your Groq API key

# Process documents (first time)
python src/document_processor.py

# Train ML models
python src/ml_integration.py

# Start web application
python app.py

# Start API server
python api.py

# Run tests
pytest tests/

# Evaluate performance
python src/evaluation_system.py

# Launch Jupyter notebooks
jupyter notebook
```

---

**Made with 🌾 for Bangladesh's Farmers**

*Empowering sustainable agriculture through AI*
