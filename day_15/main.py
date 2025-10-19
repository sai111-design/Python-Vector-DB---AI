import os
import logging
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import PyPDF2
print("main.py loaded")
import io
from pydantic import BaseModel
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="RAG Capstone API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    query_id: str

# RAG Pipeline Class
class RAGPipeline:
    def __init__(self):
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(
                name="documents",
                embedding_function=self.embedding_function
            )
        except:
            self.collection = self.chroma_client.create_collection(
                name="documents",
                embedding_function=self.embedding_function
            )
        
        logger.info("RAG Pipeline initialized successfully")
    
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            raise HTTPException(status_code=400, detail="Failed to extract PDF text")
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Break at sentence boundaries
            if end < len(text):
                last_period = chunk.rfind('.')
                if last_period > start + chunk_size // 2:
                    end = start + last_period + 1
                    chunk = text[start:end]
            
            chunks.append(chunk.strip())
            start = end - overlap
            
        return chunks
    
    async def add_document(self, filename: str, content: str) -> str:
        """Add document to vector database"""
        try:
            doc_id = str(uuid.uuid4())
            chunks = self.chunk_text(content)
            
            chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "filename": filename,
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                } for i in range(len(chunks))
            ]
            
            self.collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=chunk_ids
            )
            
            logger.info(f"Added document {filename} with {len(chunks)} chunks")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise HTTPException(status_code=500, detail="Failed to add document")
    
    def retrieve_relevant_docs(self, query: str, top_k: int = 5) -> List[dict]:
        """Retrieve relevant documents"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            retrieved_docs = []
            for i in range(len(results['documents'][0])):
                retrieved_docs.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                })
            
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def generate_answer(self, query: str, context_docs: List[dict]) -> str:
        """Generate answer using context"""
        if not context_docs:
            return "I couldn't find relevant information to answer your question."
        
        context = "\n\n".join([doc["content"] for doc in context_docs])
        
        answer = f"""Based on the retrieved documents, here's what I found:

{context[:2000]}...

This information is from {len(context_docs)} relevant document chunks that match your query: "{query}"

For a complete implementation, integrate with OpenAI GPT or another LLM for more sophisticated answers."""
        
        return answer

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

# API Endpoints
@app.get("/")
async def root():
    return {"message": "RAG Capstone API - Day 15 Task", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    try:
        collection_count = rag_pipeline.collection.count()
        return {
            "status": "healthy",
            "documents_in_db": collection_count,
            "embedding_model": "all-MiniLM-L6-v2"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document"""
    try:
        content = await file.read()
        
        if file.filename.lower().endswith('.pdf'):
            text_content = rag_pipeline.extract_text_from_pdf(content)
        elif file.filename.lower().endswith(('.txt', '.md')):
            text_content = content.decode('utf-8')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        doc_id = await rag_pipeline.add_document(file.filename, text_content)
        
        return {
            "message": f"Document '{file.filename}' uploaded successfully",
            "doc_id": doc_id,
            "text_length": len(text_content),
            "chunks_created": len(rag_pipeline.chunk_text(text_content))
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload document")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system"""
    try:
        query_id = str(uuid.uuid4())
        relevant_docs = rag_pipeline.retrieve_relevant_docs(request.query, request.top_k)
        answer = rag_pipeline.generate_answer(request.query, relevant_docs)
        
        sources = [
            {
                "filename": doc["metadata"]["filename"],
                "chunk_index": doc["metadata"]["chunk_index"],
                "distance": doc["distance"],
                "content_preview": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"]
            }
            for doc in relevant_docs
        ]
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            query_id=query_id
        )
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail="Failed to query documents")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
