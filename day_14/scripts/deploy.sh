#!/bin/bash
# Day 14 - Deployment Script for RAG Pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="rag-pipeline"
IMAGE_TAG=${1:-"latest"}
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log "${BLUE}🚀 Day 14 - RAG Pipeline Deployment${NC}"
log "======================================"

# Check prerequisites
log "${YELLOW}🔍 Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    log "${RED}❌ Docker is not installed${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    log "${RED}❌ Docker Compose is not installed${NC}"
    exit 1
fi

log "${GREEN}✅ Prerequisites satisfied${NC}"

# Build Docker image
log "${YELLOW}🏗️  Building Docker image...${NC}"

docker build \
    --build-arg BUILD_DATE="$BUILD_DATE" \
    --build-arg VCS_REF="$VCS_REF" \
    --build-arg APP_VERSION="$IMAGE_TAG" \
    -t "${APP_NAME}:${IMAGE_TAG}" \
    -t "${APP_NAME}:latest" \
    .

if [ $? -eq 0 ]; then
    log "${GREEN}✅ Docker image built successfully${NC}"
else
    log "${RED}❌ Docker image build failed${NC}"
    exit 1
fi

# Create necessary directories
log "${YELLOW}📁 Creating directories...${NC}"
mkdir -p data/chroma_db logs

# Start services
log "${YELLOW}🚀 Starting services with Docker Compose...${NC}"

export BUILD_DATE VCS_REF APP_VERSION=$IMAGE_TAG

docker-compose down --remove-orphans
docker-compose up -d

if [ $? -eq 0 ]; then
    log "${GREEN}✅ Services started successfully${NC}"
else
    log "${RED}❌ Failed to start services${NC}"
    exit 1
fi

# Wait for services to be ready
log "${YELLOW}⏳ Waiting for services to be ready...${NC}"

# Wait for API to be healthy
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
        log "${GREEN}✅ RAG Pipeline API is ready${NC}"
        break
    fi

    attempt=$((attempt + 1))
    log "   Attempt $attempt/$max_attempts..."
    sleep 10
done

if [ $attempt -eq $max_attempts ]; then
    log "${RED}❌ API failed to become ready${NC}"
    docker-compose logs rag-api
    exit 1
fi

# Run health checks
log "${YELLOW}🏥 Running health checks...${NC}"

# API Health
api_health=$(curl -s http://localhost:8000/health | jq -r '.status' 2>/dev/null || echo "error")
if [ "$api_health" = "healthy" ]; then
    log "${GREEN}✅ API Health: $api_health${NC}"
else
    log "${RED}❌ API Health: $api_health${NC}"
fi

# Redis Health  
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    log "${GREEN}✅ Redis: Connected${NC}"
else
    log "${RED}❌ Redis: Connection failed${NC}"
fi

# PostgreSQL Health
if docker-compose exec -T postgres pg_isready > /dev/null 2>&1; then
    log "${GREEN}✅ PostgreSQL: Ready${NC}"
else
    log "${RED}❌ PostgreSQL: Not ready${NC}"
fi

# Display service URLs
log "${BLUE}🌐 Service URLs:${NC}"
log "   • RAG API: http://localhost:8000"
log "   • API Docs: http://localhost:8000/docs"
log "   • Nginx: http://localhost:80"
log "   • Prometheus: http://localhost:9090"
log "   • Grafana: http://localhost:3000 (admin/admin123)"

# Test API endpoint
log "${YELLOW}🧪 Testing API endpoint...${NC}"

test_response=$(curl -s -X POST "http://localhost:8000/query" \
    -H "Content-Type: application/json" \
    -d '{"text": "What is artificial intelligence?", "max_results": 3}' || echo "error")

if echo "$test_response" | jq '.answer' > /dev/null 2>&1; then
    log "${GREEN}✅ API test successful${NC}"
else
    log "${RED}❌ API test failed${NC}"
    log "Response: $test_response"
fi

log "${GREEN}🎉 Deployment completed successfully!${NC}"
log ""
log "${BLUE}Next steps:${NC}"
log "  • Monitor services: docker-compose logs -f"
log "  • Scale API: docker-compose up -d --scale rag-api=3"
log "  • Stop services: docker-compose down"
log "  • View metrics: open http://localhost:9090"
