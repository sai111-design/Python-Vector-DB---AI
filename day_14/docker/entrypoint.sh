#!/bin/bash
# Day 14 - RAG Pipeline Docker Entrypoint

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Day 14 - RAG Pipeline Container Starting${NC}"
echo "=================================="

# Function to log with timestamp
log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Wait for dependencies
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3

    log "${YELLOW}⏳ Waiting for $service at $host:$port...${NC}"
    # (Add actual wait logic here if needed)
}

    if [ ! -w "/app/logs" ]; then
        log "${RED}❌ Logs directory not writable${NC}"
        exit 1
    fi

    # Initialize application
    log "${BLUE}🔧 Initializing RAG Pipeline...${NC}"

    # Pre-download models to avoid runtime permission issues
    log "${YELLOW}📥 Pre-downloading sentence transformer model...${NC}"

    # Set cache directories for current user
    export TRANSFORMERS_CACHE=/app/models/transformers
    export HF_HOME=/app/models/huggingface
    export HF_DATASETS_CACHE=/app/models/datasets

    # Pre-download the model with proper cache directory
python3 -c "import os; os.makedirs('/app/models/sentence-transformers', exist_ok=True); from sentence_transformers import SentenceTransformer; print('Downloading model...'); model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/app/models/sentence-transformers'); print('Model downloaded successfully')"


log "${GREEN}✅ Model download completed${NC}"
log "${GREEN}✅ Initialization complete${NC}"
log "${BLUE}🌟 Starting RAG Pipeline API Server...${NC}"

# Start the application
exec "$@"
