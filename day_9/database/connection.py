import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
import psycopg2
from pgvector.psycopg2 import register_vector
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:sai%4012345%23@localhost:5432/vectordb"
)

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,
    echo=os.getenv("DB_ECHO", "false").lower() == "true"
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

class DatabaseConnection:
    """Database connection manager for pgvector operations"""

    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string or DATABASE_URL
        self._connection = None
        self._cursor = None

    def connect(self) -> psycopg2.extensions.connection:
        """Establish connection to PostgreSQL with pgvector support"""
        try:
            self._connection = psycopg2.connect(self.connection_string)
            register_vector(self._connection)
            self._cursor = self._connection.cursor()
            logger.info("Successfully connected to PostgreSQL with pgvector")
            return self._connection
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def execute_query(self, query: str, params: tuple = None):
        """Execute a SQL query"""
        if not self._cursor:
            raise RuntimeError("Database connection not established")

        try:
            if params:
                self._cursor.execute(query, params)
            else:
                self._cursor.execute(query)
            return self._cursor.fetchall()
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            self._connection.rollback()
            raise

    def execute_non_query(self, query: str, params: tuple = None):
        """Execute non-query SQL (INSERT, UPDATE, DELETE)"""
        if not self._cursor:
            raise RuntimeError("Database connection not established")

        try:
            if params:
                self._cursor.execute(query, params)
            else:
                self._cursor.execute(query)
            self._connection.commit()
        except Exception as e:
            logger.error(f"Non-query execution failed: {e}")
            self._connection.rollback()
            raise

    def close(self):
        """Close database connection"""
        if self._cursor:
            self._cursor.close()
        if self._connection:
            self._connection.close()
        logger.info("Database connection closed")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def get_db():
    """Dependency for FastAPI to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize database with pgvector extension"""
    with DatabaseConnection() as db:
        # Enable pgvector extension
        db.execute_non_query("CREATE EXTENSION IF NOT EXISTS vector;")
        logger.info("pgvector extension enabled")

        # Create tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created")

def check_pgvector_installed():
    """Check if pgvector extension is available"""
    with DatabaseConnection() as db:
        result = db.execute_query(
            "SELECT * FROM pg_available_extensions WHERE name = 'vector';"
        )
        return len(result) > 0

def get_database_info():
    """Get database and pgvector information"""
    with DatabaseConnection() as db:
        # Get PostgreSQL version
        pg_version = db.execute_query("SELECT version();")[0][0]

        # Check pgvector extension
        pgvector_info = db.execute_query(
            "SELECT * FROM pg_extension WHERE extname = 'vector';"
        )

        return {
            "postgresql_version": pg_version,
            "pgvector_enabled": len(pgvector_info) > 0,
            "connection_string": DATABASE_URL.replace(DATABASE_URL.split('@')[0].split('://')[1], '***')
        }
