-- Initialize pgvector extension and create tables
-- This script runs automatically when the PostgreSQL container starts

-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the main embeddings table
CREATE TABLE IF NOT EXISTS document_embeddings (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255) UNIQUE NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(384),  -- Default dimension for all-MiniLM-L6-v2
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for updated_at
DROP TRIGGER IF EXISTS update_document_embeddings_updated_at ON document_embeddings;
CREATE TRIGGER update_document_embeddings_updated_at
    BEFORE UPDATE ON document_embeddings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create HNSW index for efficient similarity search
CREATE INDEX IF NOT EXISTS document_embeddings_embedding_hnsw_idx 
ON document_embeddings 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- Insert sample data for testing
INSERT INTO document_embeddings (document_id, content, metadata) 
VALUES 
    ('sample_1', 'PostgreSQL is a powerful open-source relational database system', '{"category": "database", "type": "sample"}'),
    ('sample_2', 'pgvector extension enables vector similarity search in PostgreSQL', '{"category": "vector", "type": "sample"}'),
    ('sample_3', 'Machine learning models can generate embeddings for text data', '{"category": "ml", "type": "sample"}'),
    ('sample_4', 'FastAPI is a modern web framework for building APIs with Python', '{"category": "web", "type": "sample"}'),
    ('sample_5', 'Vector databases are essential for AI and machine learning applications', '{"category": "ai", "type": "sample"}')
ON CONFLICT (document_id) DO NOTHING;

-- Create a function to get collection statistics
CREATE OR REPLACE FUNCTION get_collection_stats()
RETURNS TABLE (
    total_documents INTEGER,
    avg_content_length NUMERIC,
    first_document TIMESTAMP,
    last_document TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::INTEGER as total_documents,
        AVG(LENGTH(content)) as avg_content_length,
        MIN(created_at) as first_document,
        MAX(created_at) as last_document
    FROM document_embeddings;
END;
$$ LANGUAGE plpgsql;

-- Create a function for similarity search with metadata filtering
CREATE OR REPLACE FUNCTION search_similar_documents(
    query_embedding VECTOR,
    result_limit INTEGER DEFAULT 5,
    distance_threshold REAL DEFAULT 1.0,
    metadata_filter JSONB DEFAULT NULL
)
RETURNS TABLE (
    document_id VARCHAR,
    content TEXT,
    metadata JSONB,
    distance REAL,
    created_at TIMESTAMP
) AS $$
BEGIN
    IF metadata_filter IS NULL THEN
        RETURN QUERY
        SELECT 
            de.document_id,
            de.content,
            de.metadata,
            (de.embedding <=> query_embedding) as distance,
            de.created_at
        FROM document_embeddings de
        WHERE (de.embedding <=> query_embedding) < distance_threshold
        ORDER BY de.embedding <=> query_embedding
        LIMIT result_limit;
    ELSE
        RETURN QUERY
        SELECT 
            de.document_id,
            de.content,
            de.metadata,
            (de.embedding <=> query_embedding) as distance,
            de.created_at
        FROM document_embeddings de
        WHERE (de.embedding <=> query_embedding) < distance_threshold
        AND de.metadata @> metadata_filter
        ORDER BY de.embedding <=> query_embedding
        LIMIT result_limit;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Create view for document statistics
CREATE OR REPLACE VIEW document_stats AS
SELECT 
    COUNT(*) as total_documents,
    AVG(LENGTH(content)) as avg_content_length,
    MIN(created_at) as oldest_document,
    MAX(created_at) as newest_document,
    COUNT(DISTINCT metadata->>'category') as unique_categories
FROM document_embeddings;

-- Grant permissions (in case of restricted users)
GRANT ALL PRIVILEGES ON document_embeddings TO postgres;
GRANT ALL PRIVILEGES ON document_stats TO postgres;

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'pgvector database initialization completed successfully';
    RAISE NOTICE 'Sample data inserted, indexes created, functions defined';
END $$;
