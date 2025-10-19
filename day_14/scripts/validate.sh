#!/bin/bash
# Day 14 - Deployment Validation Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log "${BLUE}🔍 Day 14 - RAG Pipeline Deployment Validation${NC}"
log "=================================================="

# Check Docker setup
log "${YELLOW}Checking Docker setup...${NC}"
if command -v docker &> /dev/null; then
    log "${GREEN}✅ Docker: $(docker --version)${NC}"
else
    log "${RED}❌ Docker not found${NC}"
    exit 1
fi

if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
    log "${GREEN}✅ Docker Compose: Available${NC}"
else
    log "${RED}❌ Docker Compose not found${NC}"
    exit 1
fi

# Validate Dockerfile
log "${YELLOW}Validating Dockerfile...${NC}"
if [ -f "Dockerfile" ]; then
    log "${GREEN}✅ Dockerfile exists${NC}"
    # Check for multi-stage build
    if grep -q "FROM.*as.*builder" Dockerfile && grep -q "FROM.*as.*production" Dockerfile; then
        log "${GREEN}✅ Multi-stage build detected${NC}"
    else
        log "${YELLOW}⚠️  Single-stage build (consider multi-stage for production)${NC}"
    fi

    # Check for non-root user
    if grep -q "USER " Dockerfile; then
        log "${GREEN}✅ Non-root user configured${NC}"
    else
        log "${YELLOW}⚠️  Running as root (security risk)${NC}"
    fi

    # Check for health check
    if grep -q "HEALTHCHECK" Dockerfile; then
        log "${GREEN}✅ Health check configured${NC}"
    else
        log "${YELLOW}⚠️  No health check configured${NC}"
    fi
else
    log "${RED}❌ Dockerfile not found${NC}"
    exit 1
fi

# Validate docker-compose.yml
log "${YELLOW}Validating docker-compose.yml...${NC}"
if [ -f "docker-compose.yml" ]; then
    log "${GREEN}✅ docker-compose.yml exists${NC}"

    # Validate compose file
    if docker-compose config > /dev/null 2>&1 || docker compose config > /dev/null 2>&1; then
        log "${GREEN}✅ docker-compose.yml syntax valid${NC}"
    else
        log "${RED}❌ docker-compose.yml syntax invalid${NC}"
        exit 1
    fi

    # Check for essential services
    services=$(docker-compose config --services 2>/dev/null || docker compose config --services 2>/dev/null)
    for service in rag-api redis postgres nginx; do
        if echo "$services" | grep -q "$service"; then
            log "${GREEN}✅ Service '$service' configured${NC}"
        else
            log "${YELLOW}⚠️  Service '$service' not found${NC}"
        fi
    done
else
    log "${RED}❌ docker-compose.yml not found${NC}"
    exit 1
fi

# Check environment file
log "${YELLOW}Checking environment configuration...${NC}"
if [ -f ".env.example" ]; then
    log "${GREEN}✅ .env.example exists${NC}"
else
    log "${YELLOW}⚠️  .env.example not found${NC}"
fi

if [ -f ".env" ]; then
    log "${GREEN}✅ .env file exists${NC}"

    # Check for required environment variables
    required_vars=("APP_ENV" "OPENAI_API_KEY" "POSTGRES_PASSWORD")
    for var in "${required_vars[@]}"; do
        if grep -q "^$var=" .env; then
            log "${GREEN}✅ Environment variable '$var' set${NC}"
        else
            log "${YELLOW}⚠️  Environment variable '$var' not set${NC}"
        fi
    done
else
    log "${YELLOW}⚠️  .env file not found (using .env.example)${NC}"
    if [ -f ".env.example" ]; then
        cp .env.example .env
        log "${BLUE}📋 Created .env from template - please update values${NC}"
    fi
fi

# Validate Kubernetes manifests
log "${YELLOW}Validating Kubernetes manifests...${NC}"
if [ -d "k8s" ]; then
    log "${GREEN}✅ k8s directory exists${NC}"

    for file in k8s/*.yaml; do
        if [ -f "$file" ]; then
            if command -v kubectl &> /dev/null; then
                if kubectl apply --dry-run=client -f "$file" > /dev/null 2>&1; then
                    log "${GREEN}✅ $(basename $file) syntax valid${NC}"
                else
                    log "${YELLOW}⚠️  $(basename $file) syntax issues${NC}"
                fi
            else
                log "${BLUE}ℹ️  kubectl not available, skipping K8s validation${NC}"
                break
            fi
        fi
    done
else
    log "${YELLOW}⚠️  k8s directory not found${NC}"
fi

# Check monitoring configuration
log "${YELLOW}Checking monitoring setup...${NC}"
if [ -f "monitoring/prometheus.yml" ]; then
    log "${GREEN}✅ Prometheus configuration exists${NC}"
else
    log "${YELLOW}⚠️  Prometheus configuration not found${NC}"
fi

if [ -d "monitoring/grafana" ]; then
    log "${GREEN}✅ Grafana configuration exists${NC}"
else
    log "${YELLOW}⚠️  Grafana configuration not found${NC}"
fi

# Check scripts
log "${YELLOW}Checking deployment scripts...${NC}"
for script in scripts/deploy.sh scripts/test.sh; do
    if [ -f "$script" ] && [ -x "$script" ]; then
        log "${GREEN}✅ $script exists and is executable${NC}"
    elif [ -f "$script" ]; then
        log "${YELLOW}⚠️  $script exists but not executable${NC}"
        chmod +x "$script"
        log "${BLUE}📋 Made $script executable${NC}"
    else
        log "${YELLOW}⚠️  $script not found${NC}"
    fi
done

# Validate nginx configuration
log "${YELLOW}Checking nginx configuration...${NC}"
if [ -f "nginx/nginx.conf" ] && [ -f "nginx/default.conf" ]; then
    log "${GREEN}✅ Nginx configuration files exist${NC}"

    # Check if nginx config is valid (if nginx is available)
    if command -v nginx &> /dev/null; then
        if nginx -t -c nginx/nginx.conf > /dev/null 2>&1; then
            log "${GREEN}✅ Nginx configuration syntax valid${NC}"
        else
            log "${YELLOW}⚠️  Nginx configuration syntax issues${NC}"
        fi
    fi
else
    log "${YELLOW}⚠️  Nginx configuration files not found${NC}"
fi

# Final summary
log ""
log "${BLUE}📊 VALIDATION SUMMARY${NC}"
log "========================"

# Count files
total_files=$(find . -type f \( -name "*.py" -o -name "*.yml" -o -name "*.yaml" -o -name "Dockerfile*" -o -name "*.conf" -o -name "*.sh" \) | wc -l)
log "Total configuration files: $total_files"

# Check essential components
components=("Dockerfile" "docker-compose.yml" "src/main.py" "nginx/nginx.conf" "k8s/deployment.yaml")
present=0
for component in "${components[@]}"; do
    if [ -f "$component" ]; then
        present=$((present + 1))
    fi
done

log "Essential components: $present/${#components[@]}"

if [ $present -eq ${#components[@]} ]; then
    log "${GREEN}🎉 All essential components present!${NC}"
    log "${GREEN}✅ Day 14 setup is ready for deployment${NC}"

    log ""
    log "${BLUE}🚀 Next steps:${NC}"
    log "  1. Update .env with your configuration"
    log "  2. Run: ./scripts/deploy.sh"
    log "  3. Test: ./scripts/test.sh"
    log "  4. Monitor: http://localhost:3000"

    exit 0
else
    log "${RED}❌ Some essential components missing${NC}"
    exit 1
fi
