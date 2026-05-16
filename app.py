import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import tempfile

# RAG imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="DataPipeline Pro", page_icon="⚡", layout="wide")

# ============================================================================
# CONSTANTS
# ============================================================================
MODEL_PATH = "model.pkl"
FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
TARGET_NAMES = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "rag_history" not in st.session_state:
    st.session_state.rag_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0
if "sample_chunks" not in st.session_state:
    st.session_state.sample_chunks = []
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None
if "uploaded_model" not in st.session_state:
    st.session_state.uploaded_model = None

# ============================================================================
# CACHED FUNCTIONS
# ============================================================================
@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "model.pkl not found. Add your trained model to the project root."
        )
    return joblib.load(path)


@st.cache_resource
def load_embeddings(model_name: str):
    """Load embedding model."""
    return HuggingFaceEmbeddings(model_name=model_name)


@st.cache_resource
def load_llm():
    """Load FLAN-T5 model for text generation."""
    try:
        hf_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=512,
        )
        return HuggingFacePipeline(model=hf_pipeline)
    except Exception as e:
        st.error(f"Failed to load LLM: {e}")
        return None

# ============================================================================
# RAG FUNCTIONS
# ============================================================================
def process_document(uploaded_file, chunk_size: int = 500, chunk_overlap: int = 50):
    """Load and chunk document."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Load document
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)

        documents = loader.load()

        # Chunk text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents(documents)

        # Clean up temp file
        os.unlink(tmp_path)

        return chunks
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return []


def create_vector_db(chunks, embeddings):
    """Create FAISS vector database."""
    try:
        if not chunks:
            st.warning("No chunks to process.")
            return None
        
        db = FAISS.from_documents(chunks, embeddings)
        return db
    except Exception as e:
        st.error(f"Error creating vector database: {e}")
        return None


def retrieve_and_generate(query: str, vector_db, llm):
    """Retrieve relevant chunks and generate answer."""
    try:
        if vector_db is None:
            st.warning("No vector database loaded. Please upload a document first.")
            return None, []

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        )

        # Retrieve docs
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.invoke(query)

        # Generate answer
        answer = qa_chain.run(query)

        return answer, retrieved_docs
    except Exception as e:
        st.error(f"Error in retrieval or generation: {e}")
        return None, []


# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
st.sidebar.title("Configuration")

# File upload
st.sidebar.subheader("Document Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload PDF or TXT file",
    type=["pdf", "txt"],
    help="Upload a document for RAG processing"
)

# Embedding model selection
st.sidebar.subheader("Embedding Model")
embedding_model = st.sidebar.selectbox(
    "Choose embedding model",
    ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
    help="HuggingFace embedding models (no API key needed)"
)

# LLM selection
st.sidebar.subheader("Language Model")
llm_choice = st.sidebar.selectbox(
    "Choose LLM",
    ["FLAN-T5 (Recommended)", "OpenAI (Optional - Requires API Key)"],
    help="FLAN-T5 is free and works offline"
)

# ============================================================================
# MAIN INTERFACE
# ============================================================================
st.title("DataPipeline Pro - ML + RAG System")
st.write("Universal ML Prediction & Document Intelligence Platform")

# Create tabs
tab1, tab2, tab3 = st.tabs(["ML Prediction", "RAG System", "RAG History"])

# ============================================================================
# TAB 1: ML PREDICTION (GENERIC - ANY DATA)
# ============================================================================
with tab1:
    st.header("Universal ML Prediction")
    st.write("Upload your data and model, or use the pre-loaded Iris model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Input")
        data_option = st.radio("Choose data source:", ["Manual Input", "Upload CSV"], horizontal=True)
        
        if data_option == "Upload CSV":
            csv_file = st.file_uploader("Upload CSV file", type=["csv"], key="data_upload")
            if csv_file:
                df = pd.read_csv(csv_file)
                st.session_state.uploaded_data = df
                st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns")
                st.write(df.head(3))
        else:
            st.info("Use Manual Input to enter features manually")
    
    with col2:
        st.subheader("Model Input")
        model_option = st.radio("Choose model source:", ["Pre-loaded (Iris)", "Upload Custom"], horizontal=True)
        
        if model_option == "Upload Custom":
            model_file = st.file_uploader("Upload .pkl model", type=["pkl"], key="model_upload")
            if model_file:
                st.session_state.uploaded_model = model_file
                st.success("Model uploaded")
        else:
            st.info("Using pre-trained Iris model (4 features)")
    
    st.divider()
    
    # Prediction Input
    if data_option == "Manual Input":
        if model_option == "Pre-loaded (Iris)":
            st.subheader("Iris Flower Prediction (4 Features)")
            
            with st.form("iris_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=15.0, value=5.1, step=0.1)
                    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=15.0, value=1.4, step=0.1)
                
                with col2:
                    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
                    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

                predict_clicked = st.form_submit_button("Predict")

            if predict_clicked:
                try:
                    model = load_model(MODEL_PATH)

                    user_input = pd.DataFrame(
                        [[sepal_length, sepal_width, petal_length, petal_width]],
                        columns=FEATURES,
                    )

                    numeric_input = user_input.astype(float)
                    prediction = model.predict(numeric_input)[0]

                    predicted_label = TARGET_NAMES.get(int(prediction), f"Class {prediction}")
                    st.success(f"Prediction: **{predicted_label}**")

                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(numeric_input)[0]
                        prob_df = pd.DataFrame(
                            {
                                "Class": [TARGET_NAMES.get(i, f"Class {i}") for i in range(len(probs))],
                                "Probability": probs,
                            }
                        )
                        st.subheader("Prediction Confidence")
                        st.dataframe(prob_df, use_container_width=True)
                except FileNotFoundError as exc:
                    st.error(str(exc))
                except ValueError as exc:
                    st.error(f"Invalid input: {exc}")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")
        
        else:
            st.warning("Upload a custom model to make predictions")
    
    elif st.session_state.uploaded_data is not None:
        st.subheader("Custom Data Prediction")
        df = st.session_state.uploaded_data
        
        st.write(f"Columns available: {list(df.columns)}")
        
        selected_features = st.multiselect(
            "Select feature columns for prediction:",
            df.columns,
            help="Choose which columns to use as features"
        )
        
        if selected_features and model_option == "Upload Custom":
            if st.session_state.uploaded_model:
                with st.spinner("Loading model..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
                            tmp.write(st.session_state.uploaded_model.read())
                            tmp_path = tmp.name
                        
                        custom_model = joblib.load(tmp_path)
                        os.unlink(tmp_path)
                        
                        X = df[selected_features].astype(float)
                        predictions = custom_model.predict(X)
                        
                        result_df = df.copy()
                        result_df["Prediction"] = predictions
                        
                        st.success("Predictions completed!")
                        st.dataframe(result_df)
                        
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            "Download Results (CSV)",
                            csv,
                            "predictions.csv",
                            "text/csv"
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please upload a custom model")
        elif selected_features and model_option == "Pre-loaded (Iris)":
            if len(selected_features) == 4:
                try:
                    model = load_model(MODEL_PATH)
                    X = df[selected_features].astype(float)
                    predictions = model.predict(X)
                    
                    result_df = df.copy()
                    result_df["Prediction"] = predictions
                    result_df["Prediction_Label"] = result_df["Prediction"].map(TARGET_NAMES)
                    
                    st.success("Predictions completed!")
                    st.dataframe(result_df)
                    
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        "Download Results (CSV)",
                        csv,
                        "predictions.csv",
                        "text/csv"
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning(f"Pre-loaded model requires exactly 4 features, you selected {len(selected_features)}")
    else:
        st.info("Upload a CSV file or use Manual Input to get started")

# ============================================================================
# TAB 2: RAG SYSTEM
# ============================================================================
with tab2:
    st.header("RAG System - Document Intelligence")
    
    if uploaded_file is None:
        st.info("Please upload a PDF or TXT file in the sidebar to get started.")
    else:
        st.success(f"File loaded: {uploaded_file.name}")
        
        # Process document
        if st.button("Process Document", key="process_btn"):
            with st.spinner("Processing document..."):
                chunks = process_document(uploaded_file)
                
                if chunks:
                    st.session_state.num_chunks = len(chunks)
                    st.session_state.sample_chunks = [doc.page_content[:200] for doc in chunks[:3]]
                    
                    # Create embeddings and vector DB
                    embeddings = load_embeddings(embedding_model)
                    vector_db = create_vector_db(chunks, embeddings)
                    st.session_state.vector_db = vector_db
                    
                    st.success(f"Document processed! Created {len(chunks)} chunks.")

        # Display pipeline info
        if st.session_state.num_chunks > 0:
            st.subheader("Pipeline Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Chunks", st.session_state.num_chunks)
            
            with col2:
                st.metric("Embedding Model", embedding_model.split("/")[-1])
            
            with col3:
                st.metric("LLM", "FLAN-T5 (free)" if "FLAN-T5" in llm_choice else "OpenAI")

            # Show sample chunks
            with st.expander("View Sample Chunks"):
                for i, chunk in enumerate(st.session_state.sample_chunks, 1):
                    st.write(f"Chunk {i}:")
                    st.text(chunk + "...")
                    st.divider()

        # Query section
        st.subheader("Ask Your Document")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            user_query = st.text_input("Enter your question:", placeholder="What is this document about?")
        with col2:
            search_btn = st.button("Search & Generate", key="search_btn")

        if search_btn:
            if not user_query.strip():
                st.warning("Please enter a question.")
            elif st.session_state.vector_db is None:
                st.warning("Please process a document first.")
            else:
                with st.spinner("Generating answer..."):
                    # Load LLM
                    llm = load_llm()
                    
                    if llm is not None:
                        answer, retrieved_docs = retrieve_and_generate(
                            user_query,
                            st.session_state.vector_db,
                            llm
                        )
                        
                        if answer:
                            # Display answer
                            st.subheader("Generated Answer")
                            st.write(answer)
                            
                            # Display retrieved chunks
                            with st.expander("Retrieved Chunks"):
                                for i, doc in enumerate(retrieved_docs, 1):
                                    st.write(f"Chunk {i}:")
                                    st.text(doc.page_content)
                                    st.divider()
                            
                            # Store in history
                            st.session_state.rag_history.append({
                                "query": user_query,
                                "answer": answer,
                                "chunks": [doc.page_content for doc in retrieved_docs],
                                "model": embedding_model.split("/")[-1],
                            })
                            
                            st.success("Answer saved to history!")

# ============================================================================
# TAB 3: RAG HISTORY
# ============================================================================
with tab3:
    st.header("RAG Query History")
    
    if not st.session_state.rag_history:
        st.info("No queries yet. Ask a question in the RAG System tab!")
    else:
        # Clear history button
        if st.button("Clear History"):
            st.session_state.rag_history = []
            st.rerun()
        
        st.write(f"Total Queries: {len(st.session_state.rag_history)}")
        st.divider()
        
        # Display history
        for idx, entry in enumerate(st.session_state.rag_history, 1):
            with st.expander(f"Query {idx}: {entry['query'][:50]}..."):
                st.subheader("Question")
                st.write(entry["query"])
                
                st.subheader("Answer")
                st.write(entry["answer"])
                
                st.subheader("Retrieved Chunks")
                for i, chunk in enumerate(entry["chunks"], 1):
                    st.write(f"Chunk {i}:")
                    st.text(chunk)
                    st.divider()
                
                st.caption(f"Model used: {entry['model']}")

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
**DataPipeline Pro - Universal ML + RAG**
- Tab 1: Generic ML prediction - upload your data and model, or use pre-loaded Iris
- Tab 2: RAG-based document intelligence with retrieval and generation
- Tab 3: Query history and tracking
""")
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import tempfile

# RAG imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="DataPipeline Pro", page_icon="⚡", layout="wide")

# ============================================================================
# CONSTANTS
# ============================================================================
MODEL_PATH = "model.pkl"
FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
TARGET_NAMES = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "rag_history" not in st.session_state:
    st.session_state.rag_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0
if "sample_chunks" not in st.session_state:
    st.session_state.sample_chunks = []
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None
if "uploaded_model" not in st.session_state:
    st.session_state.uploaded_model = None

# ============================================================================
# CACHED FUNCTIONS
# ============================================================================
@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "model.pkl not found. Add your trained model to the project root."
        )
    return joblib.load(path)


@st.cache_resource
def load_embeddings(model_name: str):
    """Load embedding model."""
    return HuggingFaceEmbeddings(model_name=model_name)


@st.cache_resource
def load_llm():
    """Load FLAN-T5 model for text generation."""
    try:
        hf_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=512,
        )
        return HuggingFacePipeline(model=hf_pipeline)
    except Exception as e:
        st.error(f"Failed to load LLM: {e}")
        return None

# ============================================================================
# RAG FUNCTIONS
# ============================================================================
def process_document(uploaded_file, chunk_size: int = 500, chunk_overlap: int = 50):
    """Load and chunk document."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Load document
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)

        documents = loader.load()

        # Chunk text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents(documents)

        # Clean up temp file
        os.unlink(tmp_path)

        return chunks
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return []


def create_vector_db(chunks, embeddings):
    """Create FAISS vector database."""
    try:
        if not chunks:
            st.warning("No chunks to process.")
            return None
        
        db = FAISS.from_documents(chunks, embeddings)
        return db
    except Exception as e:
        st.error(f"Error creating vector database: {e}")
        return None


def retrieve_and_generate(query: str, vector_db, llm):
    """Retrieve relevant chunks and generate answer."""
    try:
        if vector_db is None:
            st.warning("No vector database loaded. Please upload a document first.")
            return None, []

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        )

        # Retrieve docs
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.invoke(query)

        # Generate answer
        answer = qa_chain.run(query)

        return answer, retrieved_docs
    except Exception as e:
        st.error(f"Error in retrieval or generation: {e}")
        return None, []


# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
st.sidebar.title("⚙️ Configuration")

# File upload
st.sidebar.subheader("📄 Document Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload PDF or TXT file",
    type=["pdf", "txt"],
    help="Upload a document for RAG processing"
)

# Embedding model selection
st.sidebar.subheader("🧠 Embedding Model")
embedding_model = st.sidebar.selectbox(
    "Choose embedding model",
    ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
    help="HuggingFace embedding models (no API key needed)"
)

# LLM selection
st.sidebar.subheader("🤖 Language Model")
llm_choice = st.sidebar.selectbox(
    "Choose LLM",
    ["FLAN-T5 (Recommended)", "OpenAI (Optional - Requires API Key)"],
    help="FLAN-T5 is free and works offline"
)

# ============================================================================
# MAIN INTERFACE
# ============================================================================
st.title("DataPipeline Pro - ML + RAG System")
st.write("Universal ML Prediction & Document Intelligence Platform")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 ML Prediction", "🔍 RAG System", "📚 RAG History"])

# ============================================================================
# TAB 1: ML PREDICTION (GENERIC - ANY DATA)
# ============================================================================
with tab1:
    st.header("🤖 Universal ML Prediction")
    st.write("Upload your data and model, or use the pre-loaded Iris model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Data Input")
        data_option = st.radio("Choose data source:", ["Manual Input", "Upload CSV"], horizontal=True)
        
        if data_option == "Upload CSV":
            csv_file = st.file_uploader("Upload CSV file", type=["csv"], key="data_upload")
            if csv_file:
                df = pd.read_csv(csv_file)
                st.session_state.uploaded_data = df
                st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
                st.write(df.head(3))
        else:
            st.info("👉 Use Manual Input to enter features manually")
    
    with col2:
        st.subheader("🤖 Model Input")
        model_option = st.radio("Choose model source:", ["Pre-loaded (Iris)", "Upload Custom"], horizontal=True)
        
        if model_option == "Upload Custom":
            model_file = st.file_uploader("Upload .pkl model", type=["pkl"], key="model_upload")
            if model_file:
                st.session_state.uploaded_model = model_file
                st.success("✅ Model uploaded")
        else:
            st.info("ℹ️ Using pre-trained Iris model (4 features)")
    
    st.divider()
    
    # Prediction Input
    if data_option == "Manual Input":
        if model_option == "Pre-loaded (Iris)":
            st.subheader("🎯 Iris Flower Prediction (4 Features)")
            
            with st.form("iris_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=15.0, value=5.1, step=0.1)
                    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=15.0, value=1.4, step=0.1)
                
                with col2:
                    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
                    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

                predict_clicked = st.form_submit_button("🎯 Predict")

            if predict_clicked:
                try:
                    model = load_model(MODEL_PATH)

                    user_input = pd.DataFrame(
                        [[sepal_length, sepal_width, petal_length, petal_width]],
                        columns=FEATURES,
                    )

                    numeric_input = user_input.astype(float)
                    prediction = model.predict(numeric_input)[0]

                    predicted_label = TARGET_NAMES.get(int(prediction), f"Class {prediction}")
                    st.success(f"✅ Prediction: **{predicted_label}**")

                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(numeric_input)[0]
                        prob_df = pd.DataFrame(
                            {
                                "Class": [TARGET_NAMES.get(i, f"Class {i}") for i in range(len(probs))],
                                "Probability": probs,
                            }
                        )
                        st.subheader("Prediction Confidence")
                        st.dataframe(prob_df, use_container_width=True)
                except FileNotFoundError as exc:
                    st.error(str(exc))
                except ValueError as exc:
                    st.error(f"Invalid input: {exc}")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")
        
        else:
            st.warning("⚠️ Upload a custom model to make predictions")
    
    elif st.session_state.uploaded_data is not None:
        st.subheader("📊 Custom Data Prediction")
        df = st.session_state.uploaded_data
        
        st.write(f"**Columns available:** {list(df.columns)}")
        
        selected_features = st.multiselect(
            "Select feature columns for prediction:",
            df.columns,
            help="Choose which columns to use as features"
        )
        
        if selected_features and model_option == "Upload Custom":
            if st.session_state.uploaded_model:
                with st.spinner("Loading model..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
                            tmp.write(st.session_state.uploaded_model.read())
                            tmp_path = tmp.name
                        
                        custom_model = joblib.load(tmp_path)
                        os.unlink(tmp_path)
                        
                        X = df[selected_features].astype(float)
                        predictions = custom_model.predict(X)
                        
                        result_df = df.copy()
                        result_df["Prediction"] = predictions
                        
                        st.success("✅ Predictions completed!")
                        st.dataframe(result_df)
                        
                        # Download results
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results (CSV)",
                            csv,
                            "predictions.csv",
                            "text/csv"
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("⚠️ Please upload a custom model")
        elif selected_features and model_option == "Pre-loaded (Iris)":
            if len(selected_features) == 4:
                try:
                    model = load_model(MODEL_PATH)
                    X = df[selected_features].astype(float)
                    predictions = model.predict(X)
                    
                    result_df = df.copy()
                    result_df["Prediction"] = predictions
                    result_df["Prediction_Label"] = result_df["Prediction"].map(TARGET_NAMES)
                    
                    st.success("✅ Predictions completed!")
                    st.dataframe(result_df)
                    
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results (CSV)",
                        csv,
                        "predictions.csv",
                        "text/csv"
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning(f"⚠️ Pre-loaded model requires exactly 4 features, you selected {len(selected_features)}")
    else:
        st.info("👆 Upload a CSV file or use Manual Input to get started")

# ============================================================================
# TAB 2: RAG SYSTEM
# ============================================================================
with tab2:
    st.header("🔍 RAG System - Document Intelligence")
    
    if uploaded_file is None:
        st.info("📤 Please upload a PDF or TXT file in the sidebar to get started.")
    else:
        st.success(f"✅ File loaded: {uploaded_file.name}")
        
        # Process document
        if st.button("📥 Process Document", key="process_btn"):
            with st.spinner("Processing document..."):
                chunks = process_document(uploaded_file)
                
                if chunks:
                    st.session_state.num_chunks = len(chunks)
                    st.session_state.sample_chunks = [doc.page_content[:200] for doc in chunks[:3]]
                    
                    # Create embeddings and vector DB
                    embeddings = load_embeddings(embedding_model)
                    vector_db = create_vector_db(chunks, embeddings)
                    st.session_state.vector_db = vector_db
                    
  DataPipeline Pro - Universal ML + RAG**
- 📊 Tab 1: Generic ML prediction - upload your data and model, or use pre-loaded Iris
        # Display pipeline info
        if st.session_state.num_chunks > 0:
            st.subheader("📊 Pipeline Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Chunks", st.session_state.num_chunks)
            
            with col2:
                st.metric("Embedding Model", embedding_model.split("/")[-1])
            
            with col3:
                st.metric("LLM", "FLAN-T5 (free)" if "FLAN-T5" in llm_choice else "OpenAI")

            # Show sample chunks
            with st.expander("👀 View Sample Chunks"):
                for i, chunk in enumerate(st.session_state.sample_chunks, 1):
                    st.write(f"**Chunk {i}:**")
                    st.text(chunk + "...")
                    st.divider()

        # Query section
        st.subheader("❓ Ask Your Document")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            user_query = st.text_input("Enter your question:", placeholder="What is this document about?")
        with col2:
            search_btn = st.button("🔍 Search & Generate", key="search_btn")

        if search_btn:
            if not user_query.strip():
                st.warning("⚠️ Please enter a question.")
            elif st.session_state.vector_db is None:
                st.warning("⚠️ Please process a document first.")
            else:
                with st.spinner("Generating answer..."):
                    # Load LLM
                    llm = load_llm()
                    
                    if llm is not None:
                        answer, retrieved_docs = retrieve_and_generate(
                            user_query,
                            st.session_state.vector_db,
                            llm
                        )
                        
                        if answer:
                            # Display answer
                            st.subheader("📝 Generated Answer")
                            st.write(answer)
                            
                            # Display retrieved chunks
                            with st.expander("📖 Retrieved Chunks"):
                                for i, doc in enumerate(retrieved_docs, 1):
                                    st.write(f"**Chunk {i}:**")
                                    st.text(doc.page_content)
                                    st.divider()
                            
                            # Store in history
                            st.session_state.rag_history.append({
                                "query": user_query,
                                "answer": answer,
                                "chunks": [doc.page_content for doc in retrieved_docs],
                                "model": embedding_model.split("/")[-1],
                            })
                            
                            st.success("✅ Answer saved to history!")

# ============================================================================
# TAB 3: RAG HISTORY
# ============================================================================
with tab3:
    st.header("📚 RAG Query History")
    
    if not st.session_state.rag_history:
        st.info("No queries yet. Ask a question in the RAG System tab!")
    else:
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.rag_history = []
            st.rerun()
        
        st.write(f"**Total Queries:** {len(st.session_state.rag_history)}")
        st.divider()
        
        # Display history
        for idx, entry in enumerate(st.session_state.rag_history, 1):
            with st.expander(f"Query {idx}: {entry['query'][:50]}..."):
                st.subheader("❓ Query")
                st.write(entry["query"])
                
                st.subheader("📝 Answer")
                st.write(entry["answer"])
                
                st.subheader("📖 Retrieved Chunks")
                for i, chunk in enumerate(entry["chunks"], 1):
                    st.write(f"**Chunk {i}:**")
                    st.text(chunk)
                    st.divider()
                
                st.caption(f"Model used: {entry['model']}")

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
**PredictFlow ML + RAG System**
- 📊 Tab 1: Traditional ML prediction with Iris dataset
- 🔍 Tab 2: RAG-based document intelligence with retrieval and generation
- 📚 Tab 3: Query history and tracking
""")

