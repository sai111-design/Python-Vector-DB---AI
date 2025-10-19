import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Optional, List, Dict, Any
import os
from functools import lru_cache

class VectorDatabase:
    def __init__(self, collection_name: str = "documents", model_name: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.Client()
        self.collection_name = collection_name
        self.model = SentenceTransformer(model_name)
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        try:
            collection = self.client.get_collection(name=self.collection_name)
        except Exception:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Document embeddings for vector search"}
            )
        return collection

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()

    def insert_document(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        try:
            embedding = self.generate_embeddings([text])

            self.collection.add(
                embeddings=embedding,
                documents=[text],
                ids=[doc_id],
                metadatas=[metadata] if metadata else None
            )
            return True
        except Exception as e:
            print(f"Error inserting document {doc_id}: {str(e)}")
            return False

    def search_documents(self, query: str, n_results: int = 5, 
                        metadata_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            query_embedding = self.generate_embeddings([query])

            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=metadata_filter,
                include=["documents", "metadatas", "distances"]
            )

            return {
                "documents": results["documents"][0] if results["documents"] else [],
                "ids": results["ids"][0] if results["ids"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else []
            }
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return {"documents": [], "ids": [], "distances": [], "metadatas": []}

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        try:
            result = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas", "embeddings"]
            )

            if not result["ids"]:
                return None

            return {
                "id": result["ids"][0],
                "document": result["documents"][0],
                "metadata": result["metadatas"][0] if result["metadatas"] and result["metadatas"][0] else None,
                "embedding_dimension": len(result["embeddings"][0]) if result["embeddings"] else 0
            }
        except Exception as e:
            print(f"Error retrieving document {doc_id}: {str(e)}")
            return None

    def delete_document(self, doc_id: str) -> bool:
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            print(f"Error deleting document {doc_id}: {str(e)}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        try:
            count = self.collection.count()
            return {
                "name": self.collection.name,
                "document_count": count,
                "metadata": self.collection.metadata
            }
        except Exception as e:
            print(f"Error getting collection info: {str(e)}")
            return {"name": self.collection_name, "document_count": 0, "metadata": None}

    def batch_insert_documents(self, documents: List[Dict[str, Any]]) -> List[bool]:
        results = []
        for doc in documents:
            success = self.insert_document(
                doc_id=doc["doc_id"],
                text=doc["text"],
                metadata=doc.get("metadata")
            )
            results.append(success)
        return results

# Global database instance
_db_instance: Optional[VectorDatabase] = None

@lru_cache()
def get_database() -> VectorDatabase:
    global _db_instance
    if _db_instance is None:
        _db_instance = VectorDatabase()
    return _db_instance
