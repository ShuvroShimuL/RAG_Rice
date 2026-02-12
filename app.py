import streamlit as st
import os
from dotenv import load_dotenv
from src.rag_system import RiceAdvisoryRAG
from src.document_processor import DocumentProcessor
import logging

# Steup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging

# Load environment variables from .env file
load_dotenv()

st.set_page_config(
    page_title="Rice Advisory RAG System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for visual enhancement
st.markdown("""
    <style>
    .main { background-color: #f5f7e9; }
    .stChatMessage { border-radius: 10px; padding: 10px; margin: 5px; }
    .user-message { background-color: #e6f3ff; }
    .assistant-message { background-color: #d4edda; }
    .stSidebar { background-color: #e8f5e9; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; }
    .stExpander { background-color: #ffffff; border-radius: 8px; }
    .prediction-box { background-color: #fff3cd; padding: 15px; border-radius: 8px; }
    .welcome-text { font-size: 1.2em; color: #2e7d32; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """Load or initialize the Rice Advisory RAG system"""
    return RiceAdvisoryRAG()

def display_message(role, message, sources=None):
    """Helper to display chat messages and optional sources"""
    with st.chat_message(role):
        st.write(message)
        if role == "assistant" and sources:
            with st.expander("📚 Sources"):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**Source {i}:** {source.get('source_file', 'Unknown')}")
                    if 'page' in source:
                        st.markdown(f"*Page: {source['page']}*")
                    st.text(source.get('content_preview', '')[:300] + "…")

def sidebar_controls(system):
    """Sidebar for configuration and document processing"""
    st.sidebar.title("🌾 Rice Advisory Controls")
    st.sidebar.markdown("Configure settings and input data for yield predictions.")

    if st.sidebar.button("Process Agricultural PDFs", key="process_pdfs"):
        with st.spinner("Processing documents and building vector store..."):
            success = system.process_documents(force_rebuild=True)
        if success:
            st.sidebar.success("Documents processed successfully!")
        else:
            st.sidebar.error("Failed to process documents. Check logs.")

    st.sidebar.markdown("---")
    st.sidebar.header("Yield Prediction Inputs")
    rice_types = [
        "Amon Broadcast", "Amon HYV", "Amon L.T",
        "Aus HYV", "Aus Local", "Boro HYV",
        "Boro Hybrid", "Boro Local"
    ]
    selected_rice_type = st.sidebar.selectbox("Select Rice Type", options=rice_types)
    region = st.sidebar.text_input("Region", "Dhaka")
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=10.0, max_value=45.0, value=25.0)
    rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=3000.0, value=1200.0)
    humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
    cultivation_area = st.sidebar.number_input("Cultivation Area (acres)", min_value=0.1, max_value=1000.0, value=1.0)

    ml_features = {
        "region": region,
        "rice_type": selected_rice_type,
        "temperature": temperature,
        "rainfall": rainfall,
        "humidity": humidity,
        "cultivation_area": cultivation_area
    }

    return ml_features

def main():
    st.title("🌾 Rice Advisory RAG System")
    st.markdown("""
        <div class="welcome-text">
        Welcome to the Rice Advisory System! Ask questions about rice cultivation, pest management, or yield predictions.
        This system uses agricultural documents and machine learning to provide practical advice for farmers.
        </div>
    """, unsafe_allow_html=True)

    # Check Groq API key
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not found! Please add it to your environment or .env file.")
        st.stop()

    # Initialize or load the system once
    system = initialize_system()

    # Sidebar inputs
    ml_features = sidebar_controls(system)

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        display_message(msg["role"], msg["content"], msg.get("sources"))

    # Input box for user query
    user_input = st.chat_input("Ask about rice cultivation...")

    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        display_message("user", user_input)

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                # Detect if the query relates to yield prediction
                include_ml = any(
                    kw in user_input.lower() for kw in ["yield", "predict", "production", "forecast", "how much"]
                )

                # Query the system
                result = system.query(
                    question=user_input,
                    include_ml_prediction=include_ml,
                    ml_features=ml_features if include_ml else None
                )

                # Handle errors
                if "error" in result:
                    answer = "Sorry, I encountered an error: " + result["error"]
                    sources = []
                else:
                    answer = result.get("response", "Sorry, I couldn't generate an answer.")
                    sources = result.get("sources", [])

                # Display response
                st.write(answer)

                # Display yield prediction if available
                if include_ml and "yield_prediction" in result:
                    prediction = result["yield_prediction"]
                    st.markdown("### 📊 Yield Prediction Results")
                    st.metric("Predicted Yield", f"{prediction.get('predicted_yield', 'N/A')} tons/acre")
                    st.metric("Confidence", f"{prediction.get('confidence', 0):.0%}")

                    if prediction.get("recommendations"):
                        st.markdown("💡 **Recommendations:**")
                        for rec in prediction["recommendations"]:
                            st.markdown(f"- {rec}")

                # Display sources in expandable section
                if sources:
                    with st.expander("📚 View Sources", expanded=False):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}:** {source.get('source_file', 'Unknown')}")
                            if 'page' in source:
                                st.markdown(f"*Page: {source['page']}*")
                            st.text(source.get("content_preview", "")[:300] + "…")

                # Save assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

if __name__ == "__main__":
    main()
