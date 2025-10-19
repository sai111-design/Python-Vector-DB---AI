import streamlit as st
import requests
import time

# Configure page
st.set_page_config(
    page_title="RAG Demo - Day 15",
    page_icon="🔍",
    layout="wide"
)

API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except:
        return False, {}

def upload_document(uploaded_file):
    """Upload document to API"""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def query_documents(query: str, top_k: int = 5):
    """Query the RAG system"""
    try:
        payload = {"query": query, "top_k": top_k}
        response = requests.post(f"{API_BASE_URL}/query", json=payload)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def main():
    st.title("🔍 RAG Demo Application - Day 15 Capstone")
    
    # Check API health
    is_healthy, health_data = check_api_health()
    
    if not is_healthy:
        st.error("⚠️ Backend API is not running. Please start the FastAPI server first.")
        st.code("uvicorn main:app --reload", language="bash")
        return
    
    st.success(f"✅ API is healthy - {health_data.get('documents_in_db', 0)} documents in database")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📚 Upload Documents")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'md'],
            help="Upload PDF, TXT, or MD files"
        )
        
        if uploaded_file is not None:
            if st.button("Upload Document"):
                with st.spinner("Processing document..."):
                    success, result = upload_document(uploaded_file)
                
                if success:
                    st.success(f"✅ {result['message']}")
                    st.json(result)
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('error', 'Upload failed')}")
    
    # Main query interface
    st.header("💬 Ask Questions")
    
    query = st.text_input(
        "Enter your question:",
        placeholder="What is the main topic discussed in the documents?"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_button = st.button("🔍 Search", type="primary")
    with col2:
        top_k = st.selectbox("Results", [3, 5, 10], index=1)
    
    if search_button and query.strip():
        with st.spinner("Searching..."):
            success, result = query_documents(query, top_k)
        
        if success:
            st.subheader("📝 Answer")
            st.write(result['answer'])
            
            st.subheader("📚 Sources")
            for i, source in enumerate(result['sources'], 1):
                with st.expander(f"Source {i}: {source['filename']} (Score: {source['distance']:.3f})"):
                    st.text(source['content_preview'])
        else:
            st.error(f"❌ Query failed: {result.get('error', 'Unknown error')}")
    
    # Display system info
    with st.expander("🔧 System Information"):
        if is_healthy:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", health_data.get('documents_in_db', 0))
            with col2:
                st.metric("Embedding Model", "all-MiniLM-L6-v2")

if __name__ == "__main__":
    main()