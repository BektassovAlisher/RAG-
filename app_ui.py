import streamlit as st
import os
from rag import initialize_vectorstore, build_rag_chain, PDF_PATH

st.set_page_config(
    page_title="Virus Knowledge Base RAG",
    page_icon="🦠",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
        color : #1a1a1a;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 5px solid #4caf50;
    }
    .message-label {
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #1a1a1a;
    }
    .message-content {
        color: #1a1a1a;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore_initialized" not in st.session_state:
    st.session_state.vectorstore_initialized = False

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

with st.sidebar:
    st.title("⚙️ Settings")
    
    st.markdown("---")
    
    st.subheader("📚 Vector Store")
    
    persist_dir = st.text_input(
        "Persist Directory",
        value="chroma_db",
        help="Directory where vector store is saved"
    )
    
    k_value = st.slider(
        "Number of Retrieved Documents (k)",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of relevant chunks to retrieve"
    )
    
    force_recreate = st.checkbox(
        "Force Recreate Vector Store",
        value=False,
        help="Recreate vector store even if it exists"
    )
    
    if st.button("🔄 Initialize/Reload Vector Store", type="primary"):
        with st.spinner("Initializing vector store..."):
            try:
                initialize_vectorstore(
                    PDF_PATH,
                    persist_dir=persist_dir,
                    force_recreate=force_recreate
                )
                st.session_state.rag_chain = build_rag_chain(
                    persist_dir=persist_dir,
                    k=k_value
                )
                st.session_state.vectorstore_initialized = True
                st.success("✅ Vector store initialized successfully!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.vectorstore_initialized = False
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    st.subheader("ℹ️ About")
    st.markdown("""
    This is a RAG (Retrieval-Augmented Generation) system for querying information about computer viruses.
    
    **Features:**
    - PDF document processing
    - Vector embeddings with HuggingFace
    - Semantic search with ChromaDB
    - LLM-powered answers with Ollama
    
    **Usage:**
    1. Initialize the vector store (first time only)
    2. Ask questions about viruses
    3. Get AI-powered answers based on the PDF
    """)

st.title("🦠 Virus Knowledge Base RAG System")
st.markdown("Ask questions about computer viruses from the loaded PDF document")

if not st.session_state.vectorstore_initialized:
    st.warning("⚠️ Please initialize the vector store using the sidebar settings.")
    st.info("👈 Click the **Initialize/Reload Vector Store** button in the sidebar to get started.")
else:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="message-label">👤 You</div>
                    <div class="message-content">{message["content"]}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message assistant-message">
                    <div class="message-label">🤖 Assistant</div>
                    <div class="message-content">{message["content"]}</div>
                </div>
            """, unsafe_allow_html=True)
    
    if prompt := st.chat_input("Ask a question about viruses..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-label">👤 You</div>
                <div class="message-content">{prompt}</div>
            </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Thinking..."):
            try:
                if st.session_state.rag_chain is None:
                    st.session_state.rag_chain = build_rag_chain(
                        persist_dir=persist_dir,
                        k=k_value
                    )
                
                response = st.session_state.rag_chain.invoke(prompt)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <div class="message-label">🤖 Assistant</div>
                        <div class="message-content">{response}</div>
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <small>Powered by LangChain, Ollama, ChromaDB, and HuggingFace Embeddings</small>
    </div>
""", unsafe_allow_html=True)