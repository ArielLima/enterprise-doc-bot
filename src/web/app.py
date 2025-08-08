import os
import sys
from pathlib import Path
import streamlit as st
from typing import List, Dict, Any

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.shared.config import load_config
from src.search.vector_store import VectorStore, KeywordSearch
from src.search.retriever import HybridRetriever, SearchResult
from src.chat.llm_client import OllamaClient, ChatBot
from src.shared.logging_utils import setup_logging, get_logger

# Setup
st.set_page_config(
    page_title="Enterprise Doc Bot",
    page_icon="🤖",
    layout="wide"
)

logger = get_logger(__name__)


@st.cache_resource
def load_system():
    """Load and cache system components"""
    config_path = os.getenv('DOCBOT_CONFIG_PATH', 'config.yaml')
    config = load_config(Path(config_path))
    
    # Initialize search components
    vector_store = VectorStore(persist_dir=config.storage.persist_dir)
    if not vector_store.documents:
        st.error("No documents found in vector store. Please run ingestion first.")
        st.stop()
    
    keyword_search = KeywordSearch()
    keyword_search.add_documents(vector_store.documents)
    
    retriever = HybridRetriever(vector_store, keyword_search)
    
    # Initialize LLM
    llm_client = OllamaClient(config.llm)
    if not llm_client.is_available():
        st.error(f"Ollama is not available at {config.llm.base_url}")
        st.stop()
    
    chatbot = ChatBot(llm_client, config.app.system_prompt)
    
    return config, retriever, chatbot, vector_store


def display_search_results(results: List[SearchResult]):
    """Display search results in sidebar"""
    with st.sidebar:
        st.subheader("📚 Retrieved Sources")
        
        for i, result in enumerate(results, 1):
            with st.expander(f"#{i} - {result.document.source} (Score: {result.score:.3f})"):
                st.write(f"**Method:** {result.retrieval_method}")
                st.write(f"**Type:** {result.document.doc_type}")
                
                # Show metadata
                metadata = result.document.metadata
                if 'file_path' in metadata:
                    st.write(f"**File:** {metadata['file_path']}")
                if 'page_title' in metadata:
                    st.write(f"**Page:** {metadata['page_title']}")
                
                st.write("**Content Preview:**")
                preview = result.document.content[:300] + "..." if len(result.document.content) > 300 else result.document.content
                st.text(preview)


def display_stats(vector_store: VectorStore):
    """Display system statistics"""
    stats = vector_store.get_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", stats["total_documents"])
    
    with col2:
        st.metric("Index Size", stats["index_size"])
    
    with col3:
        st.metric("Embedding Dim", stats["embedding_dimension"])
    
    with col4:
        st.metric("Model", stats["model_name"], delta=None)
    
    # Document types breakdown
    if stats["document_types"]:
        st.subheader("📊 Document Types")
        for doc_type, count in stats["document_types"].items():
            st.write(f"**{doc_type.title()}:** {count}")
    
    # Sources breakdown
    if stats["sources"]:
        st.subheader("📁 Sources")
        for source, count in stats["sources"].items():
            st.write(f"**{source}:** {count}")


def main():
    # Load system components
    config, retriever, chatbot, vector_store = load_system()
    
    # App title
    st.title("🤖 Enterprise Doc Bot")
    st.markdown("*Ask questions about your codebase and documentation*")
    
    # Sidebar with stats and settings
    with st.sidebar:
        st.title("⚙️ System Info")
        display_stats(vector_store)
        
        st.divider()
        
        # Search settings
        st.subheader("🔍 Search Settings")
        max_results = st.slider("Max Results", 3, 20, 5)
        vector_weight = st.slider("Vector Weight", 0.0, 1.0, 0.7, 0.1)
        keyword_weight = 1.0 - vector_weight
        
        # Update retriever weights
        retriever.vector_weight = vector_weight
        retriever.keyword_weight = keyword_weight
        
        st.write(f"Keyword Weight: {keyword_weight:.1f}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            chatbot.clear_history()
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 View Sources"):
                    for source in message["sources"]:
                        st.write(f"**{source['source']}** (Score: {source['score']:.3f})")
                        st.text(source['preview'])
                        st.divider()
    
    # Accept user input
    if prompt := st.chat_input("Ask a question about your codebase..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get context and generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                context, search_results = retriever.get_context_for_query(
                    prompt, 
                    max_tokens=4000,
                    k=max_results
                )
            
            # Display search results in sidebar
            display_search_results(search_results)
            
            with st.spinner("Generating response..."):
                response = chatbot.chat(prompt, context=context)
            
            st.markdown(response.content)
            
            # Prepare sources for display
            sources_info = []
            for result in search_results:
                preview = result.document.content[:200] + "..." if len(result.document.content) > 200 else result.document.content
                sources_info.append({
                    "source": result.document.source,
                    "score": result.score,
                    "preview": preview.replace('\n', ' ')
                })
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response.content,
            "sources": sources_info
        })


if __name__ == "__main__":
    main()