import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import logging
from pgvector.psycopg2 import register_vector
import psycopg2

logger = logging.getLogger(__name__)

class VectorOperations:
    """Handle vector operations with PostgreSQL and pgvector"""

    def __init__(self, connection_string: str, model_name: str = "all-MiniLM-L6-v2"):
        self.connection_string = connection_string
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        self._connection = None
        self._cursor = None

    def connect(self):
        """Establish database connection with pgvector support"""
        try:
            self._connection = psycopg2.connect(self.connection_string)
            register_vector(self._connection)
            self._cursor = self._connection.cursor()
            logger.info("Vector operations connection established")
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

    def disconnect(self):
        """Close database connection"""
        if self._cursor:
            self._cursor.close()
        if self._connection:
            self._connection.close()
        logger.info("Vector operations connection closed")

    def create_embeddings_table(self, table_name: str = "document_embeddings"):
        """Create table for storing document embeddings"""
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            document_id VARCHAR(255) UNIQUE NOT NULL,
            content TEXT NOT NULL,
            embedding VECTOR({self.embedding_dimension}),
            metadata JSONB DEFAULT '{{}}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        try:
            self._cursor.execute(create_table_query)
            self._connection.commit()
            logger.info(f"Table {table_name} created successfully")
        except Exception as e:
            logger.error(f"Failed to create table {table_name}: {e}")
            self._connection.rollback()
            raise

    def create_vector_index(self, table_name: str = "document_embeddings", 
                          index_type: str = "hnsw"):
        """Create vector index for efficient similarity search"""

        if index_type.lower() == "hnsw":
            # HNSW index for approximate nearest neighbor search
            index_query = f"""
            CREATE INDEX IF NOT EXISTS {table_name}_embedding_hnsw_idx 
            ON {table_name} 
            USING hnsw (embedding vector_cosine_ops) 
            WITH (m = 16, ef_construction = 64);
            """
        elif index_type.lower() == "ivfflat":
            # IVFFlat index
            index_query = f"""
            CREATE INDEX IF NOT EXISTS {table_name}_embedding_ivfflat_idx 
            ON {table_name} 
            USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100);
            """
        else:
            raise ValueError("Index type must be 'hnsw' or 'ivfflat'")

        try:
            self._cursor.execute(index_query)
            self._connection.commit()
            logger.info(f"{index_type.upper()} index created successfully")
        except Exception as e:
            logger.error(f"Failed to create {index_type} index: {e}")
            self._connection.rollback()
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text"""
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def insert_document(self, document_id: str, content: str, 
                       metadata: Dict[str, Any] = None, 
                       table_name: str = "document_embeddings") -> bool:
        """Insert document with embedding into database"""
        try:
            embedding = self.generate_embedding(content)
            metadata_json = metadata or {}

            insert_query = f"""
            INSERT INTO {table_name} (document_id, content, embedding, metadata)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (document_id) DO UPDATE SET
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata,
                updated_at = CURRENT_TIMESTAMP;
            """

            self._cursor.execute(insert_query, (
                document_id, content, embedding, psycopg2.extras.Json(metadata_json)
            ))
            self._connection.commit()
            logger.info(f"Document {document_id} inserted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to insert document {document_id}: {e}")
            self._connection.rollback()
            return False

    def search_similar_documents(self, query_text: str, limit: int = 5,
                               distance_metric: str = "cosine",
                               table_name: str = "document_embeddings",
                               metadata_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity"""

        try:
            query_embedding = self.generate_embedding(query_text)

            # Choose distance operator
            distance_ops = {
                "cosine": "<=>",
                "l2": "<->", 
                "inner_product": "<#>"
            }

            if distance_metric not in distance_ops:
                raise ValueError(f"Unknown distance metric: {distance_metric}")

            operator = distance_ops[distance_metric]

            # Build query with optional metadata filtering
            base_query = f"""
            SELECT 
                document_id,
                content,
                metadata,
                embedding {operator} %s AS distance,
                created_at
            FROM {table_name}
            """

            where_conditions = []
            params = [query_embedding]

            if metadata_filter:
                for key, value in metadata_filter.items():
                    where_conditions.append(f"metadata->>%s = %s")
                    params.extend([key, str(value)])

            if where_conditions:
                base_query += " WHERE " + " AND ".join(where_conditions)

            final_query = base_query + f" ORDER BY embedding {operator} %s LIMIT %s;"
            params.extend([query_embedding, limit])

            self._cursor.execute(final_query, params)
            results = self._cursor.fetchall()

            # Format results
            formatted_results = []
            for row in results:
                formatted_results.append({
                    "document_id": row[0],
                    "content": row[1],
                    "metadata": row[2],
                    "distance": float(row[3]),
                    "created_at": row[4].isoformat() if row[4] else None
                })

            logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise

    def get_document(self, document_id: str, 
                    table_name: str = "document_embeddings") -> Optional[Dict[str, Any]]:
        """Retrieve specific document by ID"""
        try:
            query = f"""
            SELECT document_id, content, metadata, created_at, updated_at
            FROM {table_name}
            WHERE document_id = %s;
            """

            self._cursor.execute(query, (document_id,))
            result = self._cursor.fetchone()

            if result:
                return {
                    "document_id": result[0],
                    "content": result[1],
                    "metadata": result[2],
                    "created_at": result[3].isoformat() if result[3] else None,
                    "updated_at": result[4].isoformat() if result[4] else None
                }
            return None

        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            raise

    def delete_document(self, document_id: str, 
                       table_name: str = "document_embeddings") -> bool:
        """Delete document by ID"""
        try:
            delete_query = f"DELETE FROM {table_name} WHERE document_id = %s;"
            self._cursor.execute(delete_query, (document_id,))
            deleted_count = self._cursor.rowcount
            self._connection.commit()

            if deleted_count > 0:
                logger.info(f"Document {document_id} deleted successfully")
                return True
            else:
                logger.warning(f"Document {document_id} not found")
                return False

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            self._connection.rollback()
            return False

    def get_collection_stats(self, table_name: str = "document_embeddings") -> Dict[str, Any]:
        """Get statistics about the document collection"""
        try:
            stats_query = f"""
            SELECT 
                COUNT(*) as total_documents,
                AVG(LENGTH(content)) as avg_content_length,
                MIN(created_at) as first_document,
                MAX(created_at) as last_document
            FROM {table_name};
            """

            self._cursor.execute(stats_query)
            result = self._cursor.fetchone()

            return {
                "total_documents": result[0],
                "avg_content_length": float(result[1]) if result[1] else 0,
                "first_document": result[2].isoformat() if result[2] else None,
                "last_document": result[3].isoformat() if result[3] else None,
                "embedding_dimension": self.embedding_dimension,
                "model_name": self.model.model_name
            }

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise

    def batch_insert_documents(self, documents: List[Dict[str, Any]], 
                             table_name: str = "document_embeddings") -> Dict[str, int]:
        """Batch insert multiple documents"""
        successful = 0
        failed = 0

        for doc in documents:
            try:
                success = self.insert_document(
                    document_id=doc["document_id"],
                    content=doc["content"],
                    metadata=doc.get("metadata", {}),
                    table_name=table_name
                )
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Batch insert failed for document {doc.get('document_id', 'unknown')}: {e}")
                failed += 1

        return {"successful": successful, "failed": failed}

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
